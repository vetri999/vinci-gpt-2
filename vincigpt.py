"""
vincigpt.py — A complete GPT in one file: train on TinyStories or infer from any model.

Architecture: LLaMA-family (RMSNorm, RoPE, SwiGLU, GQA, Flash Attention, KV cache)
Module tree matches HuggingFace LLaMA exactly — load open weights by stripping "model." prefix.

Run:  python vincigpt.py              (interactive: pick train or infer)
      python vincigpt.py train        (auto-train on TinyStories)
      python vincigpt.py infer        (generate from checkpoint or open model)
"""

import os, sys, json, math, time, argparse
import subprocess
import urllib.request
from contextlib import nullcontext
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def _ensure_numpy_torch_compat():
    """Fail fast if NumPy/PyTorch are mismatched (common on fresh machines)."""
    try:
        major = int(np.__version__.split('.', 1)[0])
    except Exception:
        return
    if major >= 2:
        try:
            torch.from_numpy(np.array([1], dtype=np.int64))
        except Exception as e:
            raise RuntimeError(
                "NumPy >= 2 detected, but this PyTorch build cannot use NumPy. "
                "Fix: install a compatible pair, e.g. `python -m pip install 'numpy<2'` "
                "or upgrade PyTorch to a build compiled for NumPy 2."
            ) from e

_ensure_numpy_torch_compat()


# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent
DATA_DIR       = BASE_DIR / "data"
TOKENIZER_PATH = DATA_DIR / "tokenizer.json"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

# ─── Config ───────────────────────────────────────────────────────────────────
@dataclass
class GPTConfig:
    vocab_size:        int   = 32_000
    hidden_size:       int   = 768       # d_model. 768 for 1hr H100 sweet spot (~110M params)
    num_hidden_layers: int   = 12
    num_attention_heads: int = 12
    num_key_value_heads: int = 12        # < num_attention_heads → GQA
    intermediate_size: int   = 2048      # SwiGLU hidden dim (≈ 8/3 × hidden_size)
    max_position_embeddings: int = 512   # context window (TinyStories avg story < 256 tok)
    rope_theta:        float = 10_000.0
    rms_norm_eps:      float = 1e-6
    tie_word_embeddings: bool = True
    dropout:           float = 0.0

    @property
    def head_dim(self) -> int:
        assert self.hidden_size % self.num_attention_heads == 0
        return self.hidden_size // self.num_attention_heads

    @property
    def num_kv_groups(self) -> int:
        assert self.num_attention_heads % self.num_key_value_heads == 0
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def kv_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim

# ─── RMSNorm ──────────────────────────────────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

# ─── RoPE ─────────────────────────────────────────────────────────────────────
def precompute_rope(dim, max_len, theta=10_000.0, device=None):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    pos = torch.arange(max_len, device=device).float()
    angles = torch.outer(pos, freqs)
    return torch.cos(angles), torch.sin(angles)

def apply_rope(x, cos, sin, offset=0):
    T = x.shape[2]
    c = cos[offset:offset+T].unsqueeze(0).unsqueeze(0)
    s = sin[offset:offset+T].unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1*c - x2*s, x1*s + x2*c], dim=-1).flatten(-2)

# ─── Attention ────────────────────────────────────────────────────────────────
class Attention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.n_kv_groups = config.num_kv_groups
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.kv_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.kv_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        cos, sin = precompute_rope(config.head_dim, config.max_position_embeddings, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x, mask=None, use_cache=False, kv_cache=None):
        B, T, D = x.shape
        Q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        offset = kv_cache[0].shape[2] if kv_cache is not None else 0
        Q = apply_rope(Q, self.rope_cos, self.rope_sin, offset)
        K = apply_rope(K, self.rope_cos, self.rope_sin, offset)
        if kv_cache is not None:
            K = torch.cat([kv_cache[0], K], dim=2)
            V = torch.cat([kv_cache[1], V], dim=2)
        new_cache = (K, V) if use_cache else None
        if self.n_kv_groups > 1:
            K = K.repeat_interleave(self.n_kv_groups, dim=1)
            V = V.repeat_interleave(self.n_kv_groups, dim=1)
        is_causal = (mask is None) and (kv_cache is None)
        drop_p = self.config.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask, dropout_p=drop_p, is_causal=is_causal)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.o_proj(out), new_cache

# ─── MLP (SwiGLU) ────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

# ─── Block ────────────────────────────────────────────────────────────────────
class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = MLP(config)

    def forward(self, x, mask=None, use_cache=False, kv_cache=None):
        h, cache = self.self_attn(self.input_layernorm(x), mask, use_cache, kv_cache)
        x = x + h
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, cache

# ─── GPT ──────────────────────────────────────────────────────────────────────
class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        scale = 1.0 / math.sqrt(2 * config.num_hidden_layers)
        for layer in self.layers:
            layer.self_attn.o_proj.weight.data *= scale
            layer.mlp.down_proj.weight.data *= scale

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x, mask=None, use_cache=False, kv_caches=None):
        h = self.embed_tokens(x)
        new_caches = []
        for i, layer in enumerate(self.layers):
            lc = kv_caches[i] if kv_caches else None
            h, cache = layer(h, mask, use_cache, lc)
            if use_cache: new_caches.append(cache)
        h = self.norm(h)
        if self.config.tie_word_embeddings:
            logits = h @ self.embed_tokens.weight.T
        else:
            logits = self.lm_head(h)
        return logits, new_caches if use_cache else None

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_pretrained(cls, model_name, device="cpu"):
        from safetensors.torch import load_file
        from huggingface_hub import snapshot_download
        model_path = Path(snapshot_download(model_name))
        with open(model_path / "config.json") as f:
            hf = json.load(f)
        config = GPTConfig(
            vocab_size=hf.get("vocab_size", 32000),
            hidden_size=hf.get("hidden_size", 4096),
            num_hidden_layers=hf.get("num_hidden_layers", 32),
            num_attention_heads=hf.get("num_attention_heads", 32),
            num_key_value_heads=hf.get("num_key_value_heads", hf.get("num_attention_heads", 32)),
            intermediate_size=hf.get("intermediate_size", 14336),
            max_position_embeddings=hf.get("max_position_embeddings", 8192),
            rope_theta=hf.get("rope_theta", 10000.0),
            tie_word_embeddings=hf.get("tie_word_embeddings", True),
        )
        model = cls(config)
        state = {}
        for wf in sorted(model_path.glob("*.safetensors")):
            state.update(load_file(wf, device=device))
        mapped = {}
        for k, v in state.items():
            key = k.replace("model.", "", 1) if k.startswith("model.") else k
            if key == "lm_head.weight" and config.tie_word_embeddings: continue
            mapped[key] = v
        model.load_state_dict(mapped, strict=False)
        model.to(device)
        print(f"Loaded {model_name}: {model.count_parameters():,} params")
        return model


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

# Hyperparams optimized for 1hr on single H100 80GB (vast.ai ~$2/hr)
BATCH_SIZE      = 64             # large micro-batch → high GPU utilization
GRAD_ACCUM      = 4              # effective batch = 64×4×512 = 131K tokens
MAX_STEPS       = 10_000         # fits in ~40min actual training time
WARMUP_STEPS    = 300
MAX_LR          = 6e-4           # higher LR for ~110M model
MIN_LR          = 6e-5
WEIGHT_DECAY    = 0.1
GRAD_CLIP       = 1.0
EVAL_INTERVAL   = 500
EVAL_STEPS      = 20
LOG_INTERVAL    = 10
SAVE_INTERVAL   = 2000

class DataLoader:
    def __init__(self, path, batch_size, seq_len):
        self.data = np.memmap(path, dtype=np.uint16, mode='r')
        self.bs, self.sl = batch_size, seq_len
        self.n = len(self.data)

    def get_batch(self, device):
        ix = torch.randint(0, self.n - self.sl - 1, (self.bs,))
        x = torch.stack([torch.from_numpy(self.data[i:i+self.sl].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(self.data[i+1:i+1+self.sl].astype(np.int64)) for i in ix])
        return x.to(device), y.to(device)

def build_eot_mask(x, eot_id):
    B, T = x.shape
    story_id = torch.cumsum((x == eot_id).long(), dim=1)
    same = story_id.unsqueeze(2) == story_id.unsqueeze(1)
    causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
    mask = torch.zeros(B, 1, T, T, device=x.device)
    mask.masked_fill_(~(same & causal).unsqueeze(1), float('-inf'))
    return mask

def get_lr(step):
    if step < WARMUP_STEPS: return MAX_LR * (step + 1) / WARMUP_STEPS
    if step >= MAX_STEPS: return MIN_LR
    p = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    return MIN_LR + 0.5 * (MAX_LR - MIN_LR) * (1 + math.cos(math.pi * p))

@torch.no_grad()
def evaluate(model, loader, config, eot_id, device):
    model.eval()
    loss = sum(
        F.cross_entropy(
            model((x := loader.get_batch(device))[0], mask=build_eot_mask(x[0], eot_id))[0].view(-1, config.vocab_size),
            x[1].view(-1)
        ).item() for _ in range(EVAL_STEPS)
    ) / EVAL_STEPS
    model.train()
    return loss

def _download_url(url: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    # Prefer wget (fast). Fallback to urllib if wget is missing.
    try:
        subprocess.run(["wget", "-q", "-O", str(out_path), url], check=True)
    except FileNotFoundError:
        with urllib.request.urlopen(url, timeout=60) as r, open(out_path, "wb") as f:
            while True:
                chunk = r.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Download failed: {url} (wget exit {e.returncode})") from e

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"Download produced empty file: {out_path}")


def download_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    need = []
    for s in ["train", "valid"]:
        p = DATA_DIR / f"TinyStoriesV2-GPT4-{s}.txt"
        if not p.exists() or p.stat().st_size == 0:
            need.append((s, p))

    if not need:
        print("Data already downloaded")
        return

    print("Downloading TinyStories...")
    for s, p in need:
        url = f"https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-{s}.txt"
        _download_url(url, p)
    print("Download complete")

def train_tokenizer(vocab_size=32_000):
    if TOKENIZER_PATH.exists():
        from tokenizers import Tokenizer
        return Tokenizer.from_file(str(TOKENIZER_PATH))
    print("Training BPE tokenizer...")
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=vocab_size-1, special_tokens=["<|eot|>"], show_progress=True)
    tok.train([str(DATA_DIR / "TinyStoriesV2-GPT4-train.txt")], trainer)
    tok.save(str(TOKENIZER_PATH))
    print(f"Tokenizer saved ({vocab_size} vocab)")
    return tok

def create_binary(tokenizer, split):
    path = DATA_DIR / f"{split}.bin"
    if path.exists(): return print(f"{split}.bin exists")
    suffix = "train" if split == "train" else "valid"
    eot_id = tokenizer.token_to_id("<|eot|>")
    print(f"Creating {split}.bin...")
    tokens, buf = [], []
    with open(DATA_DIR / f"TinyStoriesV2-GPT4-{suffix}.txt") as f:
        for line in f:
            line = line.strip()
            if line == "<|endoftext|>":
                if buf:
                    tokens.extend(tokenizer.encode(" ".join(buf)).ids)
                    tokens.append(eot_id)
                    buf = []
            elif line:
                buf.append(line)
        # If file doesn't end with <|endoftext|>, flush the last story
        if buf:
            tokens.extend(tokenizer.encode(" ".join(buf)).ids)
            tokens.append(eot_id)
    np.array(tokens, dtype=np.uint16).tofile(path)
    print(f"{split}.bin: {len(tokens):,} tokens")

def save_checkpoint(model, optimizer, config, step, best_val, path):
    raw = model._orig_mod if hasattr(model, '_orig_mod') else model
    torch.save({"model": raw.state_dict(), "optimizer": optimizer.state_dict(),
                "config": asdict(config), "step": step, "best_val_loss": best_val}, path)
    print(f"  Checkpoint: {path.name} (step {step})")

def train_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs="?")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()

    # Data pipeline
    download_data()
    tokenizer = train_tokenizer()
    create_binary(tokenizer, "train")
    create_binary(tokenizer, "val")
    eot_id = tokenizer.token_to_id("<|eot|>")

    # Model — pad vocab to multiple of 64 for tensor cores
    vocab = ((tokenizer.get_vocab_size() + 63) // 64) * 64
    config = GPTConfig(vocab_size=vocab)
    model = GPT(config).to(device)
    n_params = model.count_parameters()
    print(f"Model: {n_params:,} params ({config.num_hidden_layers}L, {config.hidden_size}D, {config.num_attention_heads}H)")

    if device.type == "cuda" and os.environ.get("VGPT_NO_COMPILE", "0") != "1":
        try:
            model = torch.compile(model)
            print("torch.compile enabled")
        except Exception as e:
            print(f"torch.compile failed ({e}); continuing without compile")

    # Optimizer — separate decay/no-decay
    decay = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    nodecay = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW([
        {"params": decay, "weight_decay": WEIGHT_DECAY},
        {"params": nodecay, "weight_decay": 0.0},
    ], lr=MAX_LR, betas=(0.9, 0.95), eps=1e-8, fused=device.type == "cuda")

    # Data loaders
    train_loader = DataLoader(str(DATA_DIR / "train.bin"), BATCH_SIZE, config.max_position_embeddings)
    val_loader = DataLoader(str(DATA_DIR / "val.bin"), BATCH_SIZE, config.max_position_embeddings)
    tok_per_step = BATCH_SIZE * GRAD_ACCUM * config.max_position_embeddings

    # Resume
    start_step, best_val = 0, float('inf')
    if args.resume and (CHECKPOINT_DIR / "latest.pt").exists():
        ckpt = torch.load(CHECKPOINT_DIR / "latest.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"] + 1
        best_val = ckpt.get("best_val_loss", float('inf'))
        print(f"Resumed from step {start_step}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = BASE_DIR / "train_log.jsonl"
    print(f"\nTraining: {MAX_STEPS} steps, {tok_per_step:,} tok/step, {GRAD_ACCUM} accum\n")

    model.train()
    for step in range(start_step, MAX_STEPS):
        t0 = time.time()
        lr = get_lr(step)
        for pg in optimizer.param_groups: pg["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0

        for _ in range(GRAD_ACCUM):
            X, Y = train_loader.get_batch(device)
            mask = build_eot_mask(X, eot_id)
            with amp_ctx:
                logits, _ = model(X, mask=mask)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), Y.view(-1)) / GRAD_ACCUM
            loss.backward()
            total_loss += loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if device.type == "cuda": torch.cuda.synchronize()
        dt = time.time() - t0
        tps = tok_per_step / dt
        flops = 6 * n_params * tok_per_step / dt
        mfu = flops / (990e12) * 100 if device.type == "cuda" else 0

        if step % LOG_INTERVAL == 0:
            print(f"step {step:>5d} | loss {total_loss:.4f} | lr {lr:.2e} | "
                  f"{dt*1000:.0f}ms | {tps:,.0f} tok/s | MFU {mfu:.1f}%")
            with open(log_path, "a") as f:
                f.write(json.dumps({"step": step, "loss": total_loss, "lr": lr, "mfu": mfu}) + "\n")

        if step > 0 and step % EVAL_INTERVAL == 0:
            vl = evaluate(model, val_loader, config, eot_id, device)
            best = " (best!)" if vl < best_val else ""
            print(f"  val_loss {vl:.4f}{best}")
            if vl < best_val:
                best_val = vl
                save_checkpoint(model, optimizer, config, step, best_val, CHECKPOINT_DIR / "best.pt")

        if step > 0 and step % SAVE_INTERVAL == 0:
            save_checkpoint(model, optimizer, config, step, best_val, CHECKPOINT_DIR / "latest.pt")

    save_checkpoint(model, optimizer, config, MAX_STEPS - 1, best_val, CHECKPOINT_DIR / "latest.pt")
    print(f"\nDone. Best val loss: {best_val:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

def sample_token(logits, temperature=0.8, top_k=50, top_p=0.9):
    logits = logits / temperature
    if top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[..., -1]] = float('-inf')
    if top_p < 1.0:
        sorted_l, sorted_i = torch.sort(logits, descending=True)
        cum = torch.cumsum(F.softmax(sorted_l, dim=-1), dim=-1)
        remove = cum - F.softmax(sorted_l, dim=-1) >= top_p
        sorted_l[remove] = float('-inf')
        logits = sorted_l.scatter(-1, sorted_i, sorted_l)
    return torch.multinomial(F.softmax(logits, dim=-1), 1)

@torch.no_grad()
def generate(model, tokens, max_tokens=200, temperature=0.8, top_k=50, top_p=0.9,
             eot_id=None, use_cache=True):
    model.eval()
    cfg = model._orig_mod.config if hasattr(model, '_orig_mod') else model.config
    gen_amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if tokens.is_cuda else nullcontext()
    caches = None
    for _ in range(max_tokens):
        x = tokens[:, -1:] if (use_cache and caches) else tokens[:, -cfg.max_position_embeddings:]
        with gen_amp_ctx:
            logits, new_c = model(x, use_cache=use_cache, kv_caches=caches)
        if use_cache: caches = new_c
        tok = sample_token(logits[:, -1, :].squeeze(0), temperature, top_k, top_p)
        tokens = torch.cat([tokens, tok.unsqueeze(0)], dim=1)
        if eot_id is not None and tok.item() == eot_id: break
        if use_cache and tokens.shape[1] >= cfg.max_position_embeddings: break
    return tokens

def infer_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs="?")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--no_cache", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Interactive source selection if no CLI args
    if not args.model and not args.checkpoint:
        print("\n[1] Our pretrained model  [2] Open source model (LLaMA, Mistral, etc.)")
        choice = input("Select: ").strip()
        if choice == "2":
            args.model = input("Model name (e.g. meta-llama/Llama-3.2-1B): ").strip()
        else:
            args.checkpoint = str(CHECKPOINT_DIR / "best.pt")

    if not args.prompt:
        args.prompt = input("Prompt: ").strip() or "Once upon a time"

    print(f"Device: {device}")

    if args.model:
        model = GPT.from_pretrained(args.model, device=str(device))
        try:
            from transformers import AutoTokenizer
        except Exception as e:
            raise RuntimeError(
                "Open-source model inference requires `transformers`. Install: "
                "python -m pip install transformers huggingface-hub safetensors"
            ) from e
        hf_tok = AutoTokenizer.from_pretrained(args.model)
        tokens = torch.tensor([hf_tok.encode(args.prompt)], dtype=torch.long, device=device)
        eot_id = hf_tok.eos_token_id
        decode_fn = lambda t: hf_tok.decode(t.squeeze(0).tolist(), skip_special_tokens=True)
    else:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        config = GPTConfig(**ckpt["config"])
        model = GPT(config).to(device)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded checkpoint: step {ckpt['step']}, val_loss {ckpt.get('best_val_loss', 'N/A')}")
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(TOKENIZER_PATH))
        tokens = torch.tensor([tok.encode(args.prompt).ids], dtype=torch.long, device=device)
        eot_id = tok.token_to_id("<|eot|>")
        decode_fn = lambda t: tok.decode(t.squeeze(0).tolist())

    print(f"Prompt: {args.prompt}\nGenerating...\n" + "─" * 60)
    out = generate(model, tokens, args.max_tokens, args.temperature, args.top_k, args.top_p,
                   eot_id, not args.no_cache)
    print(decode_fn(out))
    print("─" * 60 + f"\n{out.shape[1] - tokens.shape[1]} tokens generated")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else None
    if mode not in ("train", "infer", None):
        print("Usage: python vincigpt.py [train|infer]"); sys.exit(1)
    if mode is None:
        print("\n╔══════════════════════════════════╗")
        print("║         vinciGPT v1.0            ║")
        print("╠══════════════════════════════════╣")
        print("║  [1] Train on TinyStories        ║")
        print("║  [2] Generate text (infer)        ║")
        print("╚══════════════════════════════════╝")
        mode = {"1": "train", "2": "infer"}.get(input("\nSelect: ").strip())
    if mode == "train": train_main()
    elif mode == "infer": infer_main()
    else: print("Invalid selection")
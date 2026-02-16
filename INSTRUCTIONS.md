# vinciGPT — Train on vast.ai in 1 Hour

## Cost Estimate
- **GPU**: 1× H100 SXM 80GB (~$2-3/hr on vast.ai)
- **Total**: ~$3 for 1 hour (including setup overhead)
- **What you get**: ~110M param model trained ~2.5 epochs on TinyStories → coherent children's stories

## Quick Reference (copy-paste commands)

```bash
# After SSH into vast.ai instance:
cd /workspace
pip install tokenizers --break-system-packages -q
wget -q https://YOUR_UPLOAD_URL/vincigpt.py    # or use scp (see Step 5)
python vincigpt.py train
# ...wait ~50 min... done. Model saved to /workspace/checkpoints/best.pt
python vincigpt.py infer --prompt "Once upon a time"
```

---

## Step-by-Step Guide

### Step 1: Create vast.ai Account
1. Go to [vast.ai](https://vast.ai)
2. Create account → go to **Billing** → add **$5 credits** (minimum, enough for ~2 hours)

### Step 2: Add SSH Key
1. On your local machine, generate a key if you don't have one:
   ```bash
   ssh-keygen -t ed25519 -C "vastai"
   cat ~/.ssh/id_ed25519.pub
   ```
2. On vast.ai: go to **Account** → **SSH Keys** → paste your public key → Save

### Step 3: Rent a GPU
1. Go to **Templates** → select **PyTorch** template (comes with CUDA + PyTorch preinstalled)
2. Go to **Search** page (GPU marketplace)
3. Set filters:
   - **GPU**: H100 SXM (preferred) or H100 PCIe
   - **GPU RAM**: 80 GB
   - **Disk**: 50 GB (TinyStories + tokenizer + checkpoints)
   - **Reliability**: > 0.95
   - **Secure Cloud**: ✅ (recommended)
4. Sort by **price** → pick cheapest H100 (~$2-3/hr)
5. Click **RENT** → confirm

### Step 4: Connect via SSH
1. Go to **Instances** → wait for status to show **Running**
2. Click the **SSH button** (terminal icon) → copy the SSH command
3. On your local machine:
   ```bash
   ssh -p PORT root@HOST_IP     # paste the command from vast.ai
   ```

### Step 5: Upload vincigpt.py
**Option A — SCP (recommended):**
```bash
# From your LOCAL machine (not the instance):
scp -P PORT vincigpt.py root@HOST_IP:/workspace/
```

**Option B — Copy-paste via terminal:**
```bash
# On the instance, paste the file content:
cat > /workspace/vincigpt.py << 'ENDOFFILE'
# ... paste entire vincigpt.py content here ...
ENDOFFILE
```

**Option C — Via Jupyter:**
1. Click the **Jupyter button** on your instance → opens browser
2. Upload vincigpt.py through Jupyter file browser

### Step 6: Install Dependencies
```bash
cd /workspace
pip install tokenizers --break-system-packages -q
```
PyTorch and numpy come preinstalled with the PyTorch template.

### Step 7: Train
```bash
python vincigpt.py train
```

What happens automatically:
1. **Downloads TinyStories** (~500MB, 2-5 min)
2. **Trains BPE tokenizer** (32K vocab, 5-8 min)
3. **Creates binary files** (tokenized train/val data, 3-5 min)
4. **Compiles model** (torch.compile warmup, 2-3 min)
5. **Trains** 10,000 steps (~40 min)
   - You'll see logs every 10 steps with loss, learning rate, tokens/sec, MFU
   - Validation every 500 steps
   - Checkpoints saved to `checkpoints/`

Expected output:
```
Device: cuda
Downloading TinyStories...
Training BPE tokenizer...
Creating train.bin... train.bin: ~470,000,000 tokens
Creating val.bin...
Model: 109,876,224 params (12L, 768D, 12H)
torch.compile enabled

Training: 10000 steps, 131,072 tok/step, 4 accum

step     0 | loss 10.4532 | lr 2.00e-06 | 8543ms | 15,339 tok/s | MFU 0.9%
step    10 | loss 8.1023  | lr 2.20e-05 | 234ms  | 560,137 tok/s | MFU 37.4%
...
step  9990 | loss 1.0234  | lr 6.12e-05 | 221ms  | 593,041 tok/s | MFU 39.6%
  val_loss 1.0891 (best!)

Done. Best val loss: 1.0891
```

### Step 8: Generate Text
```bash
# Interactive mode
python vincigpt.py infer

# Direct mode
python vincigpt.py infer --prompt "The little dog was very happy because"
python vincigpt.py infer --prompt "Once upon a time" --temperature 0.6 --max_tokens 300
```

### Step 9: Download Your Model
```bash
# From your LOCAL machine:
scp -P PORT root@HOST_IP:/workspace/checkpoints/best.pt ./
scp -P PORT root@HOST_IP:/workspace/data/tokenizer.json ./
```

### Step 10: DESTROY THE INSTANCE
⚠️ **CRITICAL**: You are billed while the instance exists (even stopped).
1. Go to vast.ai **Instances** page
2. Click **trash icon** → **Delete** (not just Stop!)
3. Verify it's gone from the list

---

## Model Architecture & Params (optimized for 1hr H100)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| hidden_size | 768 | Sweet spot: ~110M params, trains 2.5 epochs in 1hr |
| num_hidden_layers | 12 | Chinchilla-optimal depth for this width |
| num_attention_heads | 12 | Standard 64-dim heads |
| intermediate_size | 2048 | SwiGLU 8/3 ratio (2048 ≈ 2.67 × 768) |
| max_position_embeddings | 512 | TinyStories avg story < 256 tokens |
| batch_size | 64 | Large micro-batch → high GPU utilization |
| grad_accum | 4 | Effective 131K tokens/step |
| max_lr | 6e-4 | Higher LR for smaller model (vs 3e-4 for 300M+) |
| warmup | 300 steps | ~2% of training |
| total steps | 10,000 | ~1.3B tokens = 2.5 epochs of TinyStories |

Total FLOPs: ~6 × 110M × 1.3B = 858 PetaFLOPs
H100 at 40% MFU: 396 TFLOPS → ~36 min pure training time

## Why This Beats nanoGPT on TinyStories

| Feature | vinciGPT | nanoGPT |
|---------|----------|---------|
| Activation | SwiGLU (better param efficiency) | GELU |
| Position encoding | RoPE (relative, generalizes) | Learned absolute |
| Normalization | RMSNorm (faster, no mean) | LayerNorm |
| Cross-story masking | ✅ EOT mask prevents leakage | ❌ Stories bleed across boundaries |
| Attention | Flash via SDPA | Flash via SDPA |
| Compilation | torch.compile | torch.compile |

The EOT cross-story masking is the single biggest quality advantage — it prevents the model
from learning spurious patterns across story boundaries in the packed token stream.

## Inference with Open Source Models

After training, you can also load LLaMA/Mistral/Qwen weights:
```bash
# Requires: pip install safetensors huggingface-hub transformers
python vincigpt.py infer --model meta-llama/Llama-3.2-1B --prompt "Hello world"
```

## Troubleshooting

**"CUDA out of memory"**: Reduce BATCH_SIZE from 64 → 32 in vincigpt.py
**wget fails**: vast.ai instance may not have internet; check "Direct" connection type
**torch.compile slow**: First step takes 2-3 min to compile; this is normal
**Poor generation quality**: Train longer (increase MAX_STEPS) or lower temperature to 0.5-0.6
**Resume interrupted training**: `python vincigpt.py train --resume`

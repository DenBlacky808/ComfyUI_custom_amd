# SD15 CPU-vs-MPS Tensor-Stat Probe 001

## 1. Status Summary

```
Runtime generation on MPS: PASS
Visual generation on MPS: FAIL
Visual generation on CPU: PASS
```

## 2. Gate Status

### PASS

| Gate | Notes |
|---|---|
| Startup to GUI | ComfyUI reaches GUI on AMD/macOS MPS path |
| Checkpoint visible/loadable | `v1-5-pruned-emaonly-fp16.safetensors` appears and loads |
| CLIP load | `SD1ClipModel` loads without error |
| BaseModel load | `BaseModel` loads directly to GPU/MPS |
| MPS sampler runtime completion | `25/25` steps complete without crash |
| VAE runtime completion | `AutoencoderKL` loads and runs |
| Output PNG creation | Output file is written to disk |
| CPU visual control | CPU path with same checkpoint/workflow produces a normal image |

### FAIL / NOT PASS

| Gate | Notes |
|---|---|
| MPS visual output quality | Output is colored latent/noise, not a rendered image |
| Repeated MPS visual stability | Not yet tested; single run already failed visually |
| SD 1.5 MPS quality claim | No claim of working MPS visual quality can be made at this time |

## 3. Evidence Facts

| Key | Value |
|---|---|
| torch version | `2.2.2` |
| device path | `mps` |
| runner | `run_comfy_torch22_compat.py` |
| checkpoint | `v1-5-pruned-emaonly-fp16.safetensors` |
| prompt execution time | ~13 s |

### MPS launch flags used

```
--force-fp32
--cpu-vae
--disable-smart-memory
--disable-dynamic-vram
```

### Additional attention variant tested

```
--use-split-cross-attention
```

### Observed sequence

1. `SD1ClipModel` loaded
2. `BaseModel` loaded directly to GPU/MPS
3. Sampler ran `25/25` steps to completion
4. `AutoencoderKL` loaded
5. Output PNG created
6. MPS output: colored noise (invalid)
7. CPU control: normal rendered image (same checkpoint, same workflow)

### Working conclusions

- The checkpoint is structurally OK (CPU proves it)
- The workflow graph is OK (CPU proves it)
- The CLIP conditioning path is likely OK
- The MPS runtime path is OK (no crash, no OOM, no timeout)
- The MPS visual path is NOT OK
- Likely failure zone: MPS UNet / BaseModel / sampler arithmetic, or dtype / layout / numerical interaction specific to AMD/macOS MPS

## 4. Probe Points

The following CPU-vs-MPS tensor-stat probe points must be instrumented in the future, in this order:

| # | Probe Point |
|---|---|
| 1 | Initial latent |
| 2 | Conditioning |
| 3 | First UNet noise prediction |
| 4 | Latent after first sampler step |
| 5 | Latent after final sampler step |
| 6 | Decoded image tensor |

## 5. Stats to Record at Each Probe Point

For every probe point, capture the following scalar statistics. Dump only JSON stats — never dump raw tensors to disk or to git.

| Stat | Description |
|---|---|
| `shape` | Full tensor shape, e.g. `[1, 4, 64, 64]` |
| `dtype` | e.g. `torch.float32`, `torch.float16` |
| `device` | e.g. `mps:0`, `cpu` |
| `min` | Minimum scalar value |
| `max` | Maximum scalar value |
| `mean` | Mean scalar value |
| `std` | Standard deviation |
| `nan_count` | Count of NaN elements |
| `inf_count` | Count of Inf elements |
| `digest` | Optional small checksum / hash of quantized values for change detection |

Both CPU and MPS runs must be captured with the same seed, same checkpoint, same prompt, and same sampler/scheduler settings to ensure the comparison is valid.

## 6. Interpretation Rules

| Condition | Investigation target |
|---|---|
| `initial latent` differs unexpectedly between CPU and MPS | Investigate seed handling and device-specific latent initialization |
| `conditioning` differs | Investigate CLIP conditioning path: device placement, dtype, output layout |
| CPU and MPS match before UNet but diverge at `first UNet noise prediction` | Investigate MPS UNet ops: attention, convolution, normalization, dtype/layout interaction |
| First UNet prediction is plausible but divergence appears after `latent after first sampler step` | Investigate sampler/scheduler update arithmetic on MPS |
| Final latent is plausible but `decoded image tensor` is bad | Investigate VAE decode path |
| VAE is currently forced to CPU (`--cpu-vae`) | VAE decode is **less suspicious** than MPS UNet/sampler as a first hypothesis |

## 7. Non-Goals

This document makes no claims and introduces no changes beyond the evidence record and probe plan:

- **No claim** that SD 1.5 MPS visual quality currently works
- **No dependency changes** — `torch 2.2.2` setup must not be altered
- **No runtime behavior changes** — no code is modified or added here
- **No generated images in git** — PNG output files must never be committed
- **No model files in git** — safetensors and checkpoint files must never be committed
- **No attempt to fix MPS yet** — the fix phase requires the probe data first
- **No prompt/settings chasing** before tensor divergence is localized — changing the prompt or sampler settings before knowing the first divergence point adds noise and delays diagnosis

## 8. Future Implementation Sketch

When the probe is implemented (in a separate PR, after this document is reviewed):

- Add a **read-only debug probe script** or lightweight debug hooks; do not modify production inference paths permanently
- Run the **same workflow twice**: once on CPU, once on MPS
- Use a **fixed seed** for both runs
- Use the **same checkpoint**: `v1-5-pruned-emaonly-fp16.safetensors`
- Use the **same prompt** and **same sampler/scheduler** for both runs
- **Dump only JSON stats** at each probe point — no tensors, no images
- **Compare probe outputs** to find the first divergence point
- Only after the first divergence point is identified, begin targeted fix work

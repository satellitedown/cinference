<!-- Modified by satellitedown for Cinference. Upstream attribution is retained in NOTICE. -->

# Cinference

**A custom C++/CUDA inference engine for [fafstmobel](https://huggingface.co/satellitedown/fafstmobel) on a single RTX 5090 (32 GB), with 256K context, DFlash2 decoding, and vision.**

Built from [NInfer](https://github.com/Neroued/ninfer), with source-level changes to speculative decoding, CPU/GPU round buffers, CUDA Graph management, and the DFlash2 verification kernels. Cinference modifies the native engine itself, not just its launch flags.

## What changed

- **MTP-10 decoding:** raised the draft window from 5 to 10 tokens. Longer proposals let the engine emit more tokens per verification round when the drafts are accepted.
- **Capture-based CUDA Graph reuse:** reworked MTP graph matching to use the captured node types and kernel functions. Profiles with matching signatures and batch sizes share an executable, rather than relying only on planned context ranges.
- **Expanded CPU/GPU round handling:** enlarged draft and token-position buffers and updated native validation for the longer windows. This carries MTP-10 through the decoding path, not just the command-line options.
- **Faster DFlash2 verification:** rewrote the verify-width kernels behind fafstmobel's DFlash2-15 rounds: small-token FP8/NVFP4 projection schedules, a double-buffered K-split FP8 LinearAdd, staged GDN replay records with lane-shared reductions, a warp-specialized K8V4 attention kernel that accumulates PV products in FP16 from exactly widened V and balances them across SM sub-partitions, evict-first L2 fills for once-read verify weights, wider bank-conflict-free proposal-head tiles, a GDN record convolution fused into its input projection, and a fused RMSNorm-SwiGLU FFN that hands both NVFP4 projections pre-quantized inputs. Each kernel is qualified against the existing independent oracles; greedy speculative output still matches plain target decoding.
- **Ready-to-run fafstmobel setup:** a Swift-based Qwen3.8-27B derivative with Huihui's abliteration delta and NVFP4/FP8 text weights. One ~23.7 GB NInfer v3 file bundles text, vision, MTP, and the pretrained DFlash2 draft; no local conversion or separate draft download is needed. The recommended installer uses the engine's existing DFlash2 support, not MTP-10.

## Run

For [fafstmobel](https://huggingface.co/satellitedown/fafstmobel), use the [one-menu installer](https://github.com/satellitedown/fafstmobel-cinference):

```bash
git clone https://github.com/satellitedown/fafstmobel-cinference.git
cd fafstmobel-cinference
bash setup.sh
```

Requires Linux x86_64, an RTX 5090 (32 GB), and NVIDIA drivers compatible with CUDA 13.4. Choose **1** to install, then **3** to start. The installer builds the pinned Cinference runtime and verifies the model download and accompanying license/provenance files.

### Connect

- **OpenAI-compatible API:** `http://127.0.0.1:8001/v1`
- **Model ID:** `fafstmobel-cinference`
- **Profile:** 262,144-token context, K8V4 KV cache, DFlash2 with 15 draft tokens, proposal head and vision enabled, one request at a time.

Keep the server on loopback: the installer profile has no authentication. The full 256K profile leaves little VRAM headroom. After installation, use `bash scripts/serve.sh --max-context 131072` from the installer checkout if other GPU workloads leave insufficient memory, and set the same context limit in your client.

See the installer's [configuration and client setup](https://github.com/satellitedown/fafstmobel-cinference#connect-an-application) for overrides and OMP integration. The engine still accepts other compatible v3 `.ninfer` artifacts by explicit path.

## Performance

The fafstmobel installer's [verification record](https://github.com/satellitedown/fafstmobel-cinference/blob/main/results/verification.json) covers text, tool calls, vision, and recall from a 259,843-token prompt with the 256K DFlash2-15 profile. These are integration/capacity checks, not throughput or general quality benchmarks.

The [fafstmobel model card](https://huggingface.co/satellitedown/fafstmobel#measured-results-rtx-5090-32-gb-context-16384-fp8-kv-cuda-graphs-on) reports DFlash2 at **304.1–312.9 tokens/s** on a separate 16K-context, FP8-KV, greedy 256-token workload. Those results do not measure the installer's 256K K8V4 profile.

### DFlash2 verification speedup

fafstmobel with DFlash2-15, before (`b4e8ed4`) and after the kernel changes above (`b2215c4`):

| Prompt tokens | Round time before → after | Tokens/s before → after |
|---:|---:|---:|
| 8,192 | 19.12 → 16.18 ms (−15.4%) | 393.7 → **465.5** (+18.2%) |
| 32,768 | 20.45 → 16.79 ms (−17.9%) | 447.0 → **564.6** (+26.3%)¹ |
| 131,072 | 24.22 → 19.24 ms (−20.6%) | 422.8 → **532.3** (+25.9%) |

`ninfer_bench`, greedy, 256 generated tokens, optimized proposal head, K8V4, alternating builds on one RTX 5090 with desktop GPU workloads running. At 8K and 131K both builds emitted the same tokens per round, so the tokens/s gain is engine speed alone. ¹ At 32K the optimized build also accepted more drafts (9.14 → 9.48 tokens/round); engine speed alone accounts for about +22%. [Measurements](results/rtx5090-fafstmobel-dflash2-kernels-3.json).

The second and third kernel passes are bit-exact. Against the previous release (`f6a654c`), the second pass (`6b1c3fa`) accepts the same drafts and shortens rounds by 2.8% at 8K, 3.7% at 32K and 3.0% at 131K ([measurements](results/rtx5090-fafstmobel-dflash2-kernels-2.json)). The third pass (`0ddb836`, `b2215c4`) shortens them by a further 2.3% at 8K, 2.2% at 32K and 2.1% at 131K.

A serving sweep of the first kernel pass on a synthetic Python coding prompt (512 output tokens, 1K–190K context) measured 10–21% shorter rounds. Its single-sample tokens/s varied more because greedy trajectories, and therefore acceptance, differ between builds. [Measurements](results/rtx5090-fafstmobel-dflash2-kernels.json).

### Historical Huihui measurements

The following results used the previous **Huihui Qwen3.8-27B Abliterated NVFP4** model, not fafstmobel:

| Prompt tokens | Tokens/s |
|---:|---:|
| 8,192 | 450.78 |
| 32,768 | 432.60 |
| 131,072 | 364.84 |
| 260,000 | 299.64 |

Huihui Qwen3.8-27B Abliterated NVFP4, MTP-10, K8V4. Generation speed on synthetic recall, thinking off. [Measurements](results/rtx5090-archive-recall.json).

## Model licensing

Cinference's source remains [Apache-2.0](LICENSE). The downloaded model has separate terms: fafstmobel's Swift contribution uses the **Swift Open License v1.0**, including its US$1 million commercial-use threshold. Commercial use by a legal entity exceeding that threshold requires a separate Swift Enterprise License. Read the model's [license and notices](https://huggingface.co/satellitedown/fafstmobel#licensing-and-notices), including `LICENSE.swift`, before use or redistribution. Qwen, Huihui, and DFlash2 contributions retain their Apache-2.0 terms.

[Build from source](docs/maintainer/build-system.md) · [Technical docs](docs/README.md) · [Attribution](NOTICE)

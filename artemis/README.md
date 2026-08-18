# Artemis scripts — vLLM v0.27.1 CPU (mc-010, Emerald Rapids)

Compile/test/benchmark commands for Artemis discovery runs against the
`releases/v0.27.1` baseline on the c4-standard-32-paul-a runner.

| Script | Role | Duration |
| --- | --- | --- |
| `build.sh` | Docker build of `vllm_artemis:cpu` (Dockerfile.cpu, target vllm-openai) | ~15 min cold, faster with layer cache |
| `test.sh` | Correctness gate: `fused_experts_cpu` vs pure-torch fp32 reference at Qwen3.6-35B-A3B shapes (M=1/8/64, E=256, top-k 8, hidden 2048, inter 512, bf16) | ~1 min |
| `benchmark.sh` | Serve Qwen3.6-35B-A3B (TP=1, bind 0-15, SGL on) and run `vllm bench serve` random 3000/300 conc 8 np 32; writes `artemis_results.json` | ~20 min incl. model load |

The benchmark mirrors the exp-360..364 measurement config so Artemis
results are directly comparable with the ABBA series. Key metric:
`output_throughput` (higher is better); baseline ≈ 40 tok/s.

Benchmark env overrides: `MODEL`, `NUM_PROMPTS`, `RANDOM_INPUT_LEN`,
`RANDOM_OUTPUT_LEN`, `MAX_CONCURRENCY`, `HF_CACHE_DIR` (host HF_HOME with
the model already cached; default is the mc-010 path).

# Artemis

Scripts for building and benchmarking the vLLM CPU image.

## build.sh

Builds the `vllm_artemis:cpu` Docker image from `docker/Dockerfile.cpu`.
Run from the repository root:

```bash
./artemis/build.sh
```

## benchmark.sh

Starts a vLLM OpenAI server from the `vllm_artemis:cpu` image, waits for it
to become healthy, runs `vllm bench serve` against it, and writes the numeric
metrics to `artemis_results.json` (full output in `artemis_results_raw.json`).
Containers and the bench network are torn down on exit.

```bash
./artemis/benchmark.sh
```

Override defaults via environment variables:

| Variable            | Default                 | Description              |
| ------------------- | ----------------------- | ------------------------ |
| `MODEL`             | `Qwen/Qwen3.6-35B-A3B`  | Model to serve/benchmark |
| `NUM_PROMPTS`       | `50`                    | Number of prompts        |
| `RANDOM_INPUT_LEN`  | `512`                   | Random input length      |
| `RANDOM_OUTPUT_LEN` | `512`                   | Random output length     |

Example:

```bash
MODEL=Qwen/Qwen3-0.6B NUM_PROMPTS=100 ./artemis/benchmark.sh
```

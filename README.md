# Deep Parallel

Ultra-accurate scientific reasoning with multi-path parallel thinking, verification, and ensemble strategies.

## Features

- Parallel reasoning paths (physics, math, experimental, computational, literature)
- Confidence-weighted synthesis and optional ensemble voting
- Verification layer stubs you can extend
- Ollama Modelfiles for standard and ultra configurations

## Install

```bash
pip install -e .[dev]
```

Copy `.env.example` to `.env` if your client needs OpenAI-style vars (local Ollama doesn’t require a key):

```bash
cp .env.example .env
```

## Quick start

- Run parallel reasoning demo:

```bash
deepparallel reason "How does conservation of momentum apply in elastic collisions?"
```

- Run ensemble demo:

```bash
deepparallel ensemble "State the first law of thermodynamics"
```

## Ollama: create and push

1) Find your Ollama public key and add it in settings:

- macOS: `~/.ollama/id_ed25519.pub`
- Linux: `/usr/share/ollama/.ollama/id_ed25519.pub`
- Windows: `C:\\Users\\<username>\\.ollama\\id_ed25519.pub`

2) Create and push a model (standard Modelfile):

```bash
ollama pull llama3.2
ollama create -f Modelfile Mcrowe1210/DeepParallel
ollama push Mcrowe1210/DeepParallel
```

Or create ultra configuration:

```bash
ollama create -f Modelfile-ultra Mcrowe1210/deepparallel-ultra
ollama push Mcrowe1210/deepparallel-ultra
```

Helper script:

```bash
./scripts/ollama_push.sh Mcrowe1210/DeepParallel Modelfile
```

> Note: If `ollama` is not installed, install from https://ollama.com/download

Environment notes:
- Local Ollama requires no API key; some SDKs need a dummy `OPENAI_API_KEY`.
- `.env` is in `.gitignore` so real values won’t be committed.

## Training pipeline (plan)

See `config/training_pipeline.yaml` and `scripts/train_deep_parallel.sh` for the staged approach (foundation, reasoning, DPO, ensemble). These are placeholders to be implemented with your training stack.

## Benchmarks prompts

See `prompts/benchmarks.md` for ScienceQA/ARC/MMLU tailored prompts.

## License

MIT

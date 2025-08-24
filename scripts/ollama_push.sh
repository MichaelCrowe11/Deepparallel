#!/usr/bin/env bash
set -euo pipefail

MODEL_REF="${1:-Mcrowe1210/DeepParallel}"
FILE="${2:-Modelfile}"

echo "Creating model $MODEL_REF from $FILE"
ollama create -f "$FILE" "$MODEL_REF"
echo "Pushing $MODEL_REF"
ollama push "$MODEL_REF"

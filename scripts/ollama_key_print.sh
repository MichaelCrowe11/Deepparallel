#!/usr/bin/env bash
set -euo pipefail

LINUX_KEY="/usr/share/ollama/.ollama/id_ed25519.pub"
if [[ -f "$LINUX_KEY" ]]; then
  echo "Ollama public key (Linux): $LINUX_KEY"
  cat "$LINUX_KEY"
  exit 0
fi

MAC_KEY="$HOME/.ollama/id_ed25519.pub"
if [[ -f "$MAC_KEY" ]]; then
  echo "Ollama public key (macOS): $MAC_KEY"
  cat "$MAC_KEY"
  exit 0
fi

WIN_KEY="/mnt/c/Users/$USER/.ollama/id_ed25519.pub"
if [[ -f "$WIN_KEY" ]]; then
  echo "Ollama public key (Windows via WSL): $WIN_KEY"
  cat "$WIN_KEY"
  exit 0
fi

echo "No Ollama public key found in common paths."

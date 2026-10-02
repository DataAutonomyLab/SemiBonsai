#!/usr/bin/env bash

# Resolve the repository root automatically. This file contains no credentials.
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  echo "Run 'source bash/environment.sh' from inside the SemiBonsai repository." >&2
  return 1 2>/dev/null || exit 1
}
export SemiBonsai_BASE_DIR="${REPO_ROOT}"
export PYTHONPATH="${SemiBonsai_BASE_DIR}/codes${PYTHONPATH:+:${PYTHONPATH}}"

export LLM_PROVIDER="${LLM_PROVIDER:-openai}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
export VLM_API_URL="${VLM_API_URL:-${OPENAI_BASE_URL}}"
export VLM_MODEL_TYPE="${VLM_MODEL_TYPE:-gpt-4.1}"

# Set credentials in your shell before sourcing this file:
#   export OPENAI_API_KEY="your-key"
# Optional: export VLM_API_KEY="your-key"

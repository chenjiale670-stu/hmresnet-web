#!/usr/bin/env bash
set -euo pipefail

repo="${1:-}"
if [ -z "$repo" ]; then
  echo "Usage: $0 <github-owner>/<repo-name>" >&2
  exit 1
fi

gh repo create "$repo" --private --source=. --remote=origin --push || \
gh repo create "$repo" --public --source=. --remote=origin --push


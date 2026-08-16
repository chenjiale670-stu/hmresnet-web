#!/usr/bin/env bash
set -euo pipefail

repo="${1:-}"
if [ -z "$repo" ]; then
  echo "Usage: $0 <github-owner>/<repo-name>" >&2
  exit 1
fi

if command -v gh >/dev/null 2>&1; then
  gh repo create "$repo" --private --source=. --remote=origin --push
  exit 0
fi

if git remote get-url origin >/dev/null 2>&1; then
  git push -u origin main
else
  git remote add origin "git@github.com:${repo}.git"
  git push -u origin main
fi

#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SIF="$REPO_ROOT/scripts/apptainer/biobank.sif"
apptainer exec --nv --bind "$REPO_ROOT:/repo" "$SIF" python /repo/run.py "$@"

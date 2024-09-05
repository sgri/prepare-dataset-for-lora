#!/usr/bin/env bash
set -e
baseDir=$(dirname $0)
. "$baseDir"/venv/bin/activate
export PYTHONPATH="$baseDir"/src
python src/extract_faces.py "$@"
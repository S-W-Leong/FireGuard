#!/bin/sh
set -eu
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
    echo 'Create .venv with Python 3.12 and install requirements.txt first; see README.md.' >&2
    exit 1
fi
exec .venv/bin/python server.py "$@"

#! /bin/bash

set -e

echo "Installing upmem_ai package"
# export LD_LIBRARY_PATH=$(python - <<'PY'
# import os, torch
# print(os.path.join(os.path.dirname(torch.__file__), "lib"))
# PY
# ):$LD_LIBRARY_PATH
cd ./imna && python setup.py build_ext --inplace -v --force
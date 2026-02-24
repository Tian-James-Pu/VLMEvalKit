#!/usr/bin/env bash
# Fix conda env so run.py works (libstdc++ / CXXABI_1.3.15). Run on cluster with: bash scripts/fix_uniqa_env.sh
# Requires: conda env "uniqa" activated, or pass path to env.

set -e

echo "=== 1. Current env ==="
echo "CONDA_PREFIX=${CONDA_PREFIX:-<not set>}"
echo "Python: $(which python 2>/dev/null || true)"
if [[ -n "$CONDA_PREFIX" ]]; then
  echo "Conda lib: $CONDA_PREFIX/lib"
  echo "libstdc++.so in env: $(ls -la $CONDA_PREFIX/lib/libstdc++.so* 2>/dev/null || echo 'not found')"
  echo "libicui18n in env: $(ls $CONDA_PREFIX/lib/libicui18n* 2>/dev/null || echo 'not found')"
  python -c "import ipdb; print('ipdb:', ipdb.__file__)" 2>/dev/null || echo "ipdb: not installed or import failed"
else
  echo "Activate conda env first, e.g.: conda activate uniqa"
  exit 1
fi

echo ""
echo "=== 2. Install libstdc++ from conda-forge (provides CXXABI_1.3.15) ==="
conda install -y -c conda-forge libstdcxx-ng

echo ""
echo "=== 3. Ensure conda lib is used first (this session) ==="
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

echo ""
echo "=== 4. Quick import test ==="
cd /n/fs/pvl-uniqa/VLMEvalKit 2>/dev/null || cd "$(dirname "$0")/.."
python -c "
from vlmeval.config import supported_VLM
from vlmeval.dataset import build_dataset
d = build_dataset('UniQA3D_RELPOSE')
print('OK: UniQA3D_RELPOSE built, len =', len(d.data))
" && echo "Import test passed." || { echo "Import test failed. Try running your command with:"; echo "  export LD_LIBRARY_PATH=\${CONDA_PREFIX}/lib:\$LD_LIBRARY_PATH"; exit 1; }

echo ""
echo "Done. Run your command (set LD_LIBRARY_PATH if you use a new shell):"
echo "  export LD_LIBRARY_PATH=\${CONDA_PREFIX}/lib:\$LD_LIBRARY_PATH"
echo "  python run.py --data UniQA3D_RELPOSE --model MiniCPM-V --verbose --work-dir ./outputs"

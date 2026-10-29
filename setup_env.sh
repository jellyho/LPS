#!/bin/bash
# One-shot environment setup.
#
#   sh setup_env.sh           # new env `lps`  -- training and inference
#   sh setup_env.sh droid     # into the ACTIVE env, for a host that already has DROID
#   sh setup_env.sh ogbench   # new env, with the OGBench simulator
#
# One of those, not a combination. A trailing argument overrides the environment name.
set -euo pipefail

MODE=train
NAME=""
for arg in "$@"; do
    case "$arg" in
        droid|ogbench)
            [ "$MODE" = train ] || { echo "pick one of droid, ogbench"; exit 1; }
            MODE="$arg" ;;
        -h|--help) echo "usage: sh setup_env.sh [droid|ogbench] [env_name]"; exit 0 ;;
        *) NAME="$arg" ;;
    esac
done
BASE=requirements.txt
[ "$MODE" = droid ] && BASE=requirements_droid.txt

HERE=$(cd "$(dirname "$0")" && pwd)
eval "$(conda shell.bash hook)"

if [ "$MODE" = droid ]; then
    # DROID is already here, so add to this environment instead of making one.
    NAME=${NAME:-${CONDA_DEFAULT_ENV:-}}
    if [ -z "$NAME" ]; then
        echo "activate the environment DROID is installed in first, then re-run"; exit 1
    fi
    echo "=== $NAME (existing environment) <- $BASE ==="
    conda activate "$NAME"
else
    NAME=${NAME:-lps}
    echo "=== $NAME (python 3.10) ==="
    conda create -y -n "$NAME" python=3.10
    conda activate "$NAME"
fi
if [ "$MODE" = droid ]; then
    # Say what this would move before moving it -- the robot stack is already here.
    echo "=== preflight: what would change in $NAME ==="
    _rep=$(mktemp)
    pip install -r "$HERE/$BASE" --dry-run --quiet --report "$_rep" >/dev/null
    python "$HERE/tools/preflight.py" "$_rep" || {
        rm -f "$_rep"
        echo
        echo "The packages above would change in the DROID environment. Re-run with FORCE=1 to proceed anyway."
        [ "${FORCE:-0}" = 1 ] || exit 1
    }
    rm -f "$_rep"
fi

pip install -r "$HERE/$BASE"

if [ "$MODE" = ogbench ]; then
    echo "=== OGBench extras ==="
    pip install -r "$HERE/requirements_ogbench.txt"
fi

echo "=== check ==="
python - <<'PY'
import importlib
import jax
print("jax", jax.__version__, "devices:", jax.devices())
for m in ("flax", "optax", "ml_collections", "pyarrow", "imageio_ffmpeg", "h5py"):
    try:
        importlib.import_module(m); print(f"  {m}: ok")
    except Exception as e:
        print(f"  {m}: MISSING ({type(e).__name__})")
for sim, how in (("ogbench", "ogbench"),):
    try:
        importlib.import_module(sim); print(f"  {sim}: ok")
    except Exception:
        print(f"  {sim}: not installed (run `sh setup_env.sh {how}`)")
PY
echo "=== done -- conda activate $NAME ==="

#!/usr/bin/env bash
# One-shot setup for a rented CUDA box. Nothing secret is needed or copied:
# DIAMOND's weights are a public HF repo.
set -euo pipefail

WORK="${WORK:-$HOME/wm}"
mkdir -p "$WORK" && cd "$WORK"

echo "== GPU =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "== repos =="
[ -d diamond ] || git clone --depth 1 https://github.com/eloialonso/diamond.git
[ -d world-model-divergence ] || git clone https://github.com/maskedband1t/world-model-divergence.git

echo "== python 3.12 env =="
# DIAMOND pins torch==2.4.1 / numpy==1.26 / gymnasium==0.29.1, which have no wheels
# for python >= 3.13. uv fetches a 3.12 rather than fighting the system python.
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.12 "$WORK/venv"
# shellcheck disable=SC1091
source "$WORK/venv/bin/activate"
uv pip install -r diamond/requirements.txt
uv pip install lpips

echo "== verify =="
python - <<'PY'
import torch, gymnasium, ale_py
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
print("gymnasium", gymnasium.__version__, "| ale_py", ale_py.__version__)
e = gymnasium.make("BreakoutNoFrameskip-v4", full_action_space=False, frameskip=1,
                   render_mode="rgb_array", repeat_action_probability=0.25)
assert e.unwrapped.ale.getFloat("repeat_action_probability") == 0.25
e.close(); print("sticky-action kwarg OK")
PY

cd "$WORK/world-model-divergence"
python harness/analysis.py --self-test

cat <<'MSG'

== ready ==
  source ~/wm/venv/bin/activate && cd ~/wm/world-model-divergence

  # control FIRST -- nothing downstream is interpretable without the floor
  python harness/run_day1.py control --game Breakout --diamond-root ../diamond

  # then divergence + probe features (num-samples 1: the sampler is deterministic,
  # see PREDICTIONS.md amendment 7)
  python harness/run_day1.py diverge --game Breakout --diamond-root ../diamond --num-samples 1
MSG

#!/usr/bin/env bash
# Psiduck-AI eval on a RunPod GPU pod (bf16). One command, runs detached.
#
#   cd /workspace
#   git clone https://github.com/ahaanfr/Psiduck-AI.git
#   cd Psiduck-AI && PSIDUCK_NTFY_TOPIC=<your topic> bash run_runpod.sh
#
# The run survives closing the browser tab. The pod is NOT stopped automatically
# unless you set PSIDUCK_MAX_POD_HOURS; stop it yourself after downloading the
# results. Re-running this script after a crash resumes where it stopped.
set -u
cd "$(dirname "$0")"
REPO="$(pwd)"
mkdir -p results

export PSIDUCK_QUANT="${PSIDUCK_QUANT:-bf16}"
export PSIDUCK_GSM8K_BUDGET_HOURS="${PSIDUCK_GSM8K_BUDGET_HOURS:-12}"
export HF_HOME="$REPO/.hf_cache"
MAX_POD_HOURS="${PSIDUCK_MAX_POD_HOURS:-}"  # empty = never stop the pod automatically

if [ -z "${PSIDUCK_NTFY_TOPIC:-}" ]; then
    if [ -s results/ntfy_topic.txt ]; then
        PSIDUCK_NTFY_TOPIC="$(tr -d '[:space:]' < results/ntfy_topic.txt)"
    else
        PSIDUCK_NTFY_TOPIC="psiduck-$(head -c 8 /dev/urandom | od -An -tx1 | tr -d ' \n')"
    fi
fi
export PSIDUCK_NTFY_TOPIC
echo "$PSIDUCK_NTFY_TOPIC" > results/ntfy_topic.txt

ntfy() {  # never fails the run
    curl -s -m 15 -H "Title: Psiduck eval (RunPod)" -H "Priority: ${2:-default}" \
        -d "$1" "https://ntfy.sh/$PSIDUCK_NTFY_TOPIC" > /dev/null 2>&1 || echo "(ntfy failed, continuing)"
}

stop_pod() {
    ntfy "$1 Stopping the pod now." high
    if command -v runpodctl > /dev/null && [ -n "${RUNPOD_POD_ID:-}" ]; then
        runpodctl stop pod "$RUNPOD_POD_ID" || ntfy "Could not stop the pod automatically - STOP IT IN THE RUNPOD DASHBOARD." urgent
    else
        ntfy "runpodctl unavailable - STOP THE POD IN THE RUNPOD DASHBOARD." urgent
    fi
}

# Re-launch detached so a closed browser tab doesn't kill the run.
if [ -z "${PSIDUCK_DETACHED:-}" ]; then
    PSIDUCK_DETACHED=1 nohup bash "$0" >> results/console.log 2>&1 &
    echo "=================================================================="
    echo " Started in the background (pid $!)."
    echo " ntfy topic: $PSIDUCK_NTFY_TOPIC   https://ntfy.sh/$PSIDUCK_NTFY_TOPIC"
    echo " Watch:  tail -f results/console.log      Status: cat results/STATUS.txt"
    if [ -n "$MAX_POD_HOURS" ]; then echo " Pod auto-stops ${MAX_POD_HOURS} h from now."; else echo " No auto-stop: stop the pod yourself when done."; fi
    echo "=================================================================="
    exit 0
fi

# Optional hard spend cap.
if [ -n "$MAX_POD_HOURS" ]; then
    ( sleep "$(( MAX_POD_HOURS * 3600 ))"; stop_pod "Spend cap: ${MAX_POD_HOURS} h reached." ) &
fi
ntfy "Psiduck eval starting on $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)."

if [ ! -x .venv/bin/python ]; then
    python3 -m venv .venv || { ntfy "venv creation failed" high; exit 1; }
    .venv/bin/pip install -q --upgrade pip
    .venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cu128
    .venv/bin/pip install -q -r requirements.txt || { ntfy "pip install failed" high; exit 1; }
fi
.venv/bin/python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))" \
    || { ntfy "CUDA not available in the venv - check results/console.log" urgent; exit 1; }

.venv/bin/python scripts/prefetch.py || ntfy "Prefetch failed; the driver will retry downloads itself." high

code=1
for i in $(seq 0 10); do
    .venv/bin/python scripts/run_eval.py --allow-battery
    code=$?
    [ "$code" -eq 0 ] && break
    msg="Driver exited with code $code (restart $((i + 1))/10); resuming in 30 s."
    echo "[$(date '+%F %T')] $msg" | tee -a results/run.log
    ntfy "$msg" high
    sleep 30
done
[ "$code" -ne 0 ] && ntfy "Gave up after 10 restarts. Check results/run.log." urgent

tar czf /workspace/psiduck_results.tar.gz results SUMMARY.md README.md 2> /dev/null
ntfy "DONE. Results packed at /workspace/psiduck_results.tar.gz. Download it, then STOP THE POD (it keeps billing until you do)." urgent

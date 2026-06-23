#!/usr/bin/env bash
# lgbm_watchdog.sh
# =================
# Monitors RAM usage and kills the LightGBM job if total system RAM
# exceeds THRESHOLD_GB — while leaving v25 (Chronos HPO) untouched.
#
# Usage: bash lgbm_watchdog.sh [THRESHOLD_GB]
# Default threshold: 55 GB (server has ~64 GB; v25 uses ~20 GB baseline).
#
# The watchdog polls every POLL_SECS seconds. On OOM:
#   1. SIGTERM is sent to all run_lgbm_global_*.py processes.
#   2. After GRACE_SECS, SIGKILL is sent if any survive.
# v25 (run_hpo_v25_ensemble.py) is never touched.
# LightGBM training is resumed by the HPO runner once restarted.

THRESHOLD_GB=${1:-55}
POLL_SECS=15
GRACE_SECS=10
LOG="/mnt/lab/nmwamsojo/lgbm_watchdog.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Watchdog started  threshold=${THRESHOLD_GB} GB" | tee -a "$LOG"

while true; do
    used_gb=$(awk '/MemTotal/{T=$2} /MemAvailable/{A=$2} END{printf "%.1f",(T-A)/1024/1024}' /proc/meminfo)
    v25_pid=$(pgrep -f "run_hpo_v25_ensemble.py"  | head -1)
    lgbm_pid=$(pgrep -f "run_lgbm_global"          | head -1)
    ts=$(date '+%Y-%m-%d %H:%M:%S')

    echo "[$ts] RAM ${used_gb}/${THRESHOLD_GB} GB  — v25=${v25_pid:-dead}  lgbm=${lgbm_pid:-dead}" >> "$LOG"

    # Only act if RAM is over threshold AND a LightGBM job is running
    if awk "BEGIN{exit !($used_gb > $THRESHOLD_GB)}" && [ -n "$lgbm_pid" ]; then
        echo "[$ts] OOM (${used_gb} GB > ${THRESHOLD_GB} GB) — killing LightGBM PIDs" | tee -a "$LOG"
        pkill -TERM -f "run_lgbm_global" 2>/dev/null
        sleep $GRACE_SECS
        pkill -KILL -f "run_lgbm_global" 2>/dev/null
        echo "[$ts] LightGBM killed. v25 ($v25_pid) untouched." | tee -a "$LOG"
    fi

    sleep $POLL_SECS
done

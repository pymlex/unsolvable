#!/usr/bin/env bash
# Auto-resume tip-2 synthesis if jsonl mtime/n_done stalls > STALL_SEC.
set -u
ROOT=/tmp/unsolvable
JSONL=$ROOT/results/synthesis/train-deepseek-v4-flash-0731/train.jsonl
PROG=$ROOT/results/synthesis/train-deepseek-v4-flash-0731/progress.json
STARTF=$ROOT/results/synthesis/train-deepseek-v4-flash-0731/run_start_unix.txt
STATUS=/cursor/stores/self/internal/synth-1000-status.md
LOG=$ROOT/results/synthesis/train-deepseek-v4-flash-0731/watchdog.log
TARGET=${SYNTHESIS_N_ITEMS:-500}
STALL_SEC=${SYNTH_STALL_SEC:-240}
WORKERS=${SYNTH_WORKERS:-3}
SEED=${SYNTH_WATCHDOG_SEED:-20}
TMUX_CONF=/exec-daemon/tmux.portal.conf
SESSION=synth-500

cd "$ROOT"
mkdir -p "$(dirname "$LOG")"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) watchdog start stall=${STALL_SEC}s target=$TARGET workers=$WORKERS" >>"$LOG"

last_n=-1
last_mtime=0
last_change=$(date -u +%s)

jsonl_n() {
  wc -l < "$JSONL" 2>/dev/null | tr -d '[:space:]' || echo 0
}

jsonl_mtime() {
  stat -c %Y "$JSONL" 2>/dev/null || echo 0
}

synth_pids() {
  pgrep -f 'python3 -u -c from synthesis.generate import' || true
}

write_status() {
  local phase=$1 note=$2
  local n invalid last rate eta now_str run_start now stalled
  n=$(jsonl_n)
  invalid=$(python3 -c "import json;print(json.load(open('$PROG')).get('n_invalid',0))" 2>/dev/null || echo 0)
  last=$(python3 -c "import json;print(json.load(open('$PROG')).get('last_error') or 'none')" 2>/dev/null || echo none)
  run_start=$(cat "$STARTF" 2>/dev/null || date -u +%s)
  now=$(date -u +%s)
  now_str=$(date -u '+%Y-%m-%d %H:%M:%S UTC')
  stalled=$((now - last_change))
  rate=$(python3 -c "print(round(max(int('$n'),0)/max(($now-float('$run_start'))/60.0,1e-6),2))" 2>/dev/null || echo n/a)
  eta=$(python3 -c "n=int('$n'); r=n/max(($now-float('$run_start'))/60.0,1e-6); rem=max(int('$TARGET')-n,0); print(f'{rem/r:.1f} min' if r>0 else 'n/a')" 2>/dev/null || echo n/a)
  printf '%s\n' \
    "# Synth 500 status" \
    "" \
    "- updated: $now_str" \
    "- phase: $phase" \
    "- n_done: **$n / $TARGET**" \
    "- invalid_or_skip: $invalid" \
    "- rate: $rate items/min" \
    "- ETA: $eta" \
    "- last_error: $last" \
    "- workers: $WORKERS" \
    "- watchdog: stall>${STALL_SEC}s auto-resume ON" \
    "- stall_timer: ${stalled}s / ${STALL_SEC}s" \
    "- note: $note" \
    "- output: results/synthesis/train-deepseek-v4-flash-0731/train.jsonl" \
    >"$STATUS"
}

hard_kill_synth() {
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) hard_kill begin" >>"$LOG"
  tmux -f "$TMUX_CONF" has-session -t "=$SESSION" 2>/dev/null && \
    tmux -f "$TMUX_CONF" kill-session -t "$SESSION" 2>/dev/null || true
  local pids
  pids=$(synth_pids)
  if [ -n "$pids" ]; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) kill -TERM $pids" >>"$LOG"
    kill -TERM $pids 2>/dev/null || true
    sleep 2
    pids=$(synth_pids)
    if [ -n "$pids" ]; then
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) kill -KILL $pids" >>"$LOG"
      kill -KILL $pids 2>/dev/null || true
      sleep 1
    fi
  fi
  pkill -9 -f 'python3 -u -c from synthesis.generate import' 2>/dev/null || true
  sleep 1
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) hard_kill done remaining=$(synth_pids)" >>"$LOG"
}

start_synth() {
  local seed=$1
  hard_kill_synth
  date -u +%s >"$STARTF"
  python3 - <<PY
import json
from pathlib import Path
from datetime import datetime, timezone
p=Path("$PROG")
d=json.loads(p.read_text()) if p.exists() else {}
d["n_done"]=int("$(jsonl_n)")
d["n_target"]=int("$TARGET")
d["workers"]=int("$WORKERS")
d["note"]="watchdog auto-resume seed=$seed"
d["updated_at_utc"]=datetime.now(timezone.utc).isoformat()
p.write_text(json.dumps(d, ensure_ascii=False, indent=2)+"\n")
PY
  tmux -f "$TMUX_CONF" new-session -d -s "$SESSION" -c "$ROOT" -- bash -lc "
export PATH=\"\$HOME/.local/bin:\$PATH\"
export ZVENOAI_API_KEY=\"${ZVENOAI_API_KEY}\"
export SYNTHESIS_N_ITEMS=$TARGET
export DATASET_ID=data/reshuege-ru-2026/train.jsonl
export SYNTH_STATUS_PATH=/cursor/stores/bc-190c58b9-1167-4499-a9a7-ccf4cda65298/internal/synth-1000-status.md
export SYNTH_PROGRESS_EVERY=10
export SYNTH_WORKERS=$WORKERS
export ZVENO_HTTP_RETRIES=6
cd $ROOT
python3 -u -c \"from synthesis.generate import generate_synthetic_split; generate_synthetic_split(n_items=$TARGET, dataset='data/reshuege-ru-2026/train.jsonl', seed=$seed)\" 2>&1 | tee -a results/synthesis/train-deepseek-v4-flash-0731/run.log
echo EXIT:\$? >> results/synthesis/train-deepseek-v4-flash-0731/run.log
"
  sleep 3
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) started synth seed=$seed pids=$(synth_pids)" >>"$LOG"
}

while true; do
  n=$(jsonl_n)
  mtime=$(jsonl_mtime)
  now=$(date -u +%s)
  py=$(synth_pids | head -1)

  if [ "$n" -ge "$TARGET" ]; then
    write_status done "target reached; watchdog exiting"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) done n=$n" >>"$LOG"
    break
  fi

  if [ "$n" != "$last_n" ] || [ "$mtime" != "$last_mtime" ]; then
    last_n=$n
    last_mtime=$mtime
    last_change=$now
  fi

  stalled=$((now - last_change))

  if [ -z "$py" ]; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) no python; auto-resume n=$n" >>"$LOG"
    write_status "watchdog-restart" "generator missing; auto-resume n=$n"
    SEED=$((SEED + 1))
    start_synth "$SEED"
    last_change=$(date -u +%s)
    last_n=$(jsonl_n)
    last_mtime=$(jsonl_mtime)
    sleep 20
    continue
  fi

  if [ "$stalled" -ge "$STALL_SEC" ]; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) STALL ${stalled}s at n=$n mtime=$mtime py=$py; hard restart" >>"$LOG"
    write_status "watchdog-restart" "jsonl stalled ${stalled}s at n=$n; hard-kill + auto-resume"
    SEED=$((SEED + 1))
    start_synth "$SEED"
    last_change=$(date -u +%s)
    last_n=$(jsonl_n)
    last_mtime=$(jsonl_mtime)
    sleep 30
    continue
  fi

  write_status running "pid $py; stall_timer=${stalled}s/${STALL_SEC}s"
  sleep 20
done

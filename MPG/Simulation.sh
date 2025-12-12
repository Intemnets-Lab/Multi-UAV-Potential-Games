#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# LOG FUNCTION
# ============================================================================
log(){ printf '[%(%F %T)T] %s\n' -1 "$*"; }

# ============================================================================
# DISCORD WEBHOOK + MESSAGE FORMATTERS
# ============================================================================

WEBHOOK_URL="_MF8WAneDb9EDHzn5oJO"

# --- Basic text message ---
send_discord() {
    local MESSAGE="$1"
    curl -s -H "Content-Type: application/json" \
         -X POST \
         -d "{\"content\": \"$MESSAGE\"}" \
         "$WEBHOOK_URL" >/dev/null
}

# --- Fancy EMBED message ---
send_embed() {
    local TITLE="$1"
    local DESC="$2"
    local COLOR="$3"  # decimal color (e.g. green 3066993)

    curl -s -H "Content-Type: application/json" \
         -X POST \
         -d "$(cat <<EOF
{
  "embeds": [{
    "title": "$TITLE",
    "description": "$DESC",
    "color": $COLOR
  }]
}
EOF
)" "$WEBHOOK_URL" >/dev/null
}

# --- Error notification ---
send_error() {
    local MSG="$1"
    send_embed "❌ ERROR" "$MSG" 15158332
}

# --- Success notification ---
send_success() {
    local MSG="$1"
    send_embed "✅ SUCCESS" "$MSG" 3066993
}

# --- Info notification ---
send_info() {
    local MSG="$1"
    send_embed "ℹ️ Info" "$MSG" 3447003
}

# --- Section header ---
send_section() {
    local MSG="$1"
    send_embed "📘 Simulation Update" "$MSG" 15844367
}

# ============================================================================
# PROJECT SETUP
# ============================================================================
ROOT="$(cd -- "$(dirname "$0")" >/dev/null 2>&1 && pwd)"
log "📂 Project root = $ROOT"

send_section "🚀 Simulation started  
**Machine:** $HOSTNAME  
**Root:** \`$ROOT\`"

VENV_DIR="/home/intemnets-lab/venv/uav"
PY="$VENV_DIR/bin/python3.10"
PIP="$VENV_DIR/bin/pip"

if [[ ! -x "$PY" ]]; then
    log "❌ Python 3.10 venv not found at $PY"
    send_error "Python venv not found at: \`$PY\`  
Simulation aborted."
    exit 1
fi

log "🐍 Using Python: $PY"
log "🐍 Python version: $($PY -V)"

# ============================================================================
# CHECK REQUIRED PACKAGES
# ============================================================================
REQS=(numpy pandas matplotlib pyyaml openpyxl imageio)

log "📦 Checking required packages"
send_info "Checking Python dependencies…"

for pkg in "${REQS[@]}"; do
    if ! "$PY" -c "import $pkg" 2>/dev/null; then
        log "📥 Installing missing package: $pkg"
        send_info "Installing missing package: **$pkg**"
        "$PIP" install "$pkg" -q
    fi
done

log "✅ All dependencies ready"
send_success "All Python packages verified."

# ============================================================================
# UAV LIST
# ============================================================================
UAVS=(3 4 5 6 7 8 9 10)

log "==================================================="
log "                 UAV list = ${UAVS[*]}"
log "==================================================="

send_section "UAV Configuration  
UAVs to simulate: **${UAVS[*]}**  
Total runs: **${#UAVS[@]}**"

# ============================================================================
# SCRIPT PATHS
# ============================================================================
MAIN="$ROOT/Overlap.py"
IRADA="$ROOT/Comparison6.py"
ANALYSIS="$ROOT/Analysis.py"

LOG_DIR="$ROOT/SimRuns"
mkdir -p "$LOG_DIR"

# ============================================================================
# MAIN SIMULATION LOOP
# ============================================================================
TOTAL=${#UAVS[@]}
COUNT=0
START_TIME=$(date +%s)

for u in "${UAVS[@]}"; do
    COUNT=$((COUNT + 1))
    LOG_FILE="$LOG_DIR/run_${u}.log"
    UAV_START_TIME=$(date +%s)  # Track start time for current UAV batch

    send_section "▶️ **Starting UAV batch: $u**"

    # store high-level stage summaries for this UAV
    SUMMARY="**UAV Count: $u**\n"

    # --- Overlap.py ---
    log "▶️ Running MPG.py"
    send_info "Running **Games.py** for $u UAVs…"

    if "$PY" "$MAIN" --num_uavs "$u" 2>&1 | tee "$LOG_FILE"; then
        SUMMARY+="✔️ Games.py completed\n"
    else
        send_error "Games.py crashed for **$u UAVs**"
        exit 1
    fi

    # --- IRADA ---
    log "▶️ Running IRADA.py"
    send_info "Running **IRADA.py** for $u UAVs…"

    if "$PY" "$IRADA" --num_uavs "$u" 2>&1 | tee -a "$LOG_FILE"; then
        SUMMARY+="✔️ IRADA.py completed\n"
    else
        send_error "IRADA.py crashed for **$u UAVs**"
        exit 1
    fi

    # --- Analysis ---
    log "▶️ Running Analysis.py"
    send_info "Running **Analysis.py** for $u UAVs…"

    if "$PY" "$ANALYSIS" --num_uavs "$u" 2>&1 | tee -a "$LOG_FILE"; then
        SUMMARY+="✔️ Analysis.py completed\n"
    else
        send_error "Analysis.py crashed for **$u UAVs**"
        exit 1
    fi

    # Time for this UAV batch
    UAV_END_TIME=$(date +%s)
    UAV_TIME=$((UAV_END_TIME - UAV_START_TIME))
    AVG_TIME_PER_UAV=$((UAV_TIME / 1))  # Only for 1 UAV batch
    REMAINING_UAVS=$((TOTAL - COUNT))
    ETA_UAV=$((REMAINING_UAVS * AVG_TIME_PER_UAV))  # Estimated time for the rest of UAVs

    # Calculate full ETA
    ELAPSED_TIME=$(($UAV_END_TIME - $START_TIME))
    FULL_ETA=$((ELAPSED_TIME * TOTAL / COUNT - ELAPSED_TIME))

    # Convert times to HH:MM:SS
    ETA_UAV_FORMATTED=$(printf "%02d:%02d:%02d" $((ETA_UAV / 3600)) $(((ETA_UAV / 60) % 60)) $((ETA_UAV % 60)))
    FULL_ETA_FORMATTED=$(printf "%02d:%02d:%02d" $((FULL_ETA / 3600)) $(((FULL_ETA / 60) % 60)) $((FULL_ETA % 60)))

    send_info "✅ Finished UAV batch $u.  
Estimated time remaining for this batch: **$ETA_UAV_FORMATTED**  
Full simulation ETA: **$FULL_ETA_FORMATTED**"

    log "✅ FINISHED: num_uavs=$u"
    send_success "Finished batch for **$u UAVs**  
$SUMMARY"
done

# ============================================================================
# FINISH
# ============================================================================
log "🎉 All simulations completed successfully."
send_section "🎉 **All simulations completed successfully.**  
Total UAV batches: **$TOTAL**  
Logs in: \`$LOG_DIR\`"

#!/usr/bin/env bash
# One command for a phone session: Ollama (if the model needs it), livekit-server,
# and PAL in remote mode. Extra arguments go to main.py, e.g. ./start-remote.sh --verbose
# Ctrl+C stops everything this script started.
set -euo pipefail
cd "$(dirname "$0")"

tmp="${TMPDIR:-/tmp}"
LIVEKIT_LOG="${tmp%/}/pal-livekit.log"
started=()

cleanup() {
    # ${a[@]+...}: macOS's bash 3.2 treats an empty array as unset under set -u
    pids=(${started[@]+"${started[@]}"})
    [ ${#pids[@]} -eq 0 ] && return
    kill "${pids[@]}" 2>/dev/null || true
    sleep 1
    # livekit-server drains rooms on the first signal and only quits on a second one
    kill "${pids[@]}" 2>/dev/null || true
}
trap cleanup EXIT

wait_for_port() {  # host, port, name
    for _ in $(seq 1 50); do
        nc -z "$1" "$2" 2>/dev/null && return 0
        sleep 0.2
    done
    echo "$3 didn't come up on $1:$2" >&2
    exit 1
}

# The address the phone connects to: Tailscale if it's there, otherwise the Wi-Fi network
tailscale_cli=$(command -v tailscale || echo /Applications/Tailscale.app/Contents/MacOS/Tailscale)
if [ -x "$tailscale_cli" ] && ip=$("$tailscale_cli" ip -4 2>/dev/null) && [ -n "$ip" ]; then
    network="Tailscale"
    # Listen only on the Tailscale address: others on the same Wi-Fi can't join the room
    listen="$ip"
else
    ip=$(ipconfig getifaddr en0 || true)
    network="Wi-Fi"
    listen="0.0.0.0"
fi
if [ -z "$ip" ]; then
    echo "No network address found (no Tailscale, no Wi-Fi)" >&2
    exit 1
fi
local_check=$([ "$listen" = "0.0.0.0" ] && echo 127.0.0.1 || echo "$listen")

if [[ "${PAL_MODEL:-ollama:}" == ollama:* ]] && ! nc -z 127.0.0.1 11434 2>/dev/null; then
    echo "Starting Ollama"
    ollama serve >/dev/null 2>&1 &
    started+=($!)
    wait_for_port 127.0.0.1 11434 Ollama
fi

if nc -z 127.0.0.1 7880 2>/dev/null || nc -z "$local_check" 7880 2>/dev/null; then
    echo "livekit-server already running, reusing it (it keeps the addresses it was started with)"
else
    echo "Starting livekit-server on $listen (log: $LIVEKIT_LOG)"
    livekit-server --dev --bind "$listen" --node-ip "$ip" >"$LIVEKIT_LOG" 2>&1 &
    started+=($!)
    wait_for_port "$local_check" 7880 livekit-server
fi

echo
echo "  In the phone app, connect to: $ip  ($network)"
echo

LIVEKIT_URL="${LIVEKIT_URL:-ws://$local_check:7880}" PAL_TOKEN_HOST="$listen" \
    poetry run python main.py --remote --preview "$@"

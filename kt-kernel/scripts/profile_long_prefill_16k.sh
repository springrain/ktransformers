#!/usr/bin/env bash
set -Eeuo pipefail

# Long-prefill benchmark + low-overhead GPU profile for an SGLang server.
#
# Required:
#   PROMPT_FILE=/path/to/representative_long_prompt.txt
#
# Typical Docker mapping:
#   -v /data/sglang-profiles:/profiles
#
# Example:
#   BASE_URL=http://127.0.0.1:40010 \
#   PROMPT_FILE=/data/prompts/real_16k_plus.txt \
#   SERVER_PROFILE_ROOT=/profiles \
#   HOST_PROFILE_ROOT=/data/sglang-profiles \
#   bash profile_long_prefill_16k.sh

BASE_URL="${BASE_URL:-http://127.0.0.1:40010}"
BASE_URL="${BASE_URL%/}"

PROMPT_FILE="${PROMPT_FILE:-/data/models/long_prompt.txt}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-16384}"
MIN_PROMPT_TOKENS="${MIN_PROMPT_TOKENS:-16384}"
BASELINE_RUNS="${BASELINE_RUNS:-3}"
PROFILE_STEPS="${PROFILE_STEPS:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1}"

# SERVER_PROFILE_ROOT is a path inside the SGLang container.
# HOST_PROFILE_ROOT is the corresponding bind-mounted path on the Docker host.
SERVER_PROFILE_ROOT="${SERVER_PROFILE_ROOT:-/profiles}"
HOST_PROFILE_ROOT="${HOST_PROFILE_ROOT:-/data/sglang-profiles}"

# Optional: set this when /profiles is not bind-mounted and traces must be copied
# out with docker cp.
CONTAINER="${CONTAINER:-}"

TAG="$(date +%Y%m%d-%H%M%S)"
PROFILE_NAME="kt-prefill-16k-${TAG}"
SERVER_PROFILE_DIR="${SERVER_PROFILE_ROOT%/}/${PROFILE_NAME}"
HOST_PROFILE_DIR="${HOST_PROFILE_ROOT%/}/${PROFILE_NAME}"
LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-./${PROFILE_NAME}-client}"

REQUEST_JSON="${LOCAL_LOG_DIR}/request.json"
WARMUP_RESPONSE="${LOCAL_LOG_DIR}/warmup-response.json"
PROFILE_RESPONSE="${LOCAL_LOG_DIR}/profile-response.json"
PROFILE_CONTROL_LOG="${LOCAL_LOG_DIR}/start-profile.log"
PROFILE_CONTROL_ERR="${LOCAL_LOG_DIR}/start-profile.err"

PROFILE_PID=""
DMON_PID=""

cleanup() {
    if [[ -n "${DMON_PID}" ]] && kill -0 "${DMON_PID}" 2>/dev/null; then
        kill "${DMON_PID}" 2>/dev/null || true
        wait "${DMON_PID}" 2>/dev/null || true
    fi

    if [[ -n "${PROFILE_PID}" ]] && kill -0 "${PROFILE_PID}" 2>/dev/null; then
        curl --silent --show-error --max-time 60 \
            -X POST "${BASE_URL}/stop_profile" \
            > "${LOCAL_LOG_DIR}/emergency-stop-profile.log" 2>&1 || true
        kill "${PROFILE_PID}" 2>/dev/null || true
        wait "${PROFILE_PID}" 2>/dev/null || true
    fi
}

trap cleanup EXIT INT TERM

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "ERROR: required command not found: $1" >&2
        exit 1
    fi
}

flush_cache() {
    curl --fail --silent --show-error \
        --connect-timeout 10 --max-time 60 \
        -X POST "${BASE_URL}/flush_cache?timeout=30" \
        > /dev/null
}

send_request() {
    local response_file="$1"
    local time_file="$2"

    curl --fail --silent --show-error \
        --connect-timeout 10 --max-time 1800 \
        -H "Content-Type: application/json" \
        -o "${response_file}" \
        -w '%{time_total}\n' \
        --data-binary "@${REQUEST_JSON}" \
        "${BASE_URL}/generate" \
        > "${time_file}"
}

report_response() {
    local label="$1"
    local response_file="$2"
    local time_file="$3"

    python3 - "${label}" "${response_file}" "${time_file}" <<'PY'
import json
import pathlib
import sys

label, response_path, time_path = sys.argv[1:]
response = json.loads(pathlib.Path(response_path).read_text(encoding="utf-8"))
meta = response.get("meta_info") or {}
prompt_tokens = int(meta.get("prompt_tokens") or 0)
completion_tokens = int(meta.get("completion_tokens") or 0)
e2e_latency = meta.get("e2e_latency")
http_seconds = float(pathlib.Path(time_path).read_text().strip())
approx_rate = prompt_tokens / http_seconds if prompt_tokens and http_seconds > 0 else 0.0

print(f"[{label}]")
print(f"  prompt_tokens       : {prompt_tokens}")
print(f"  completion_tokens   : {completion_tokens}")
print(f"  HTTP wall time      : {http_seconds:.6f} s")
if e2e_latency is not None:
    print(f"  server e2e_latency  : {float(e2e_latency):.6f} s")
print(f"  approximate rate    : {approx_rate:.2f} input tok/s")
print("  NOTE: approximate rate includes tokenization, scheduling, one output token, and HTTP overhead.")
PY
}

require_command curl
require_command python3

if [[ ! -f "${PROMPT_FILE}" ]]; then
    echo "ERROR: PROMPT_FILE does not exist: ${PROMPT_FILE}" >&2
    echo "Use a representative real or sanitized long document, not one repeated sentence." >&2
    exit 1
fi

if [[ ! "${BASELINE_RUNS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: BASELINE_RUNS must be a positive integer." >&2
    exit 1
fi

if [[ ! "${PROFILE_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: PROFILE_STEPS must be a positive integer." >&2
    exit 1
fi

mkdir -p "${LOCAL_LOG_DIR}"

python3 - "${PROMPT_FILE}" "${REQUEST_JSON}" "${MAX_NEW_TOKENS}" <<'PY'
import json
import pathlib
import sys

prompt_path, output_path, max_new_tokens = sys.argv[1:]
text = pathlib.Path(prompt_path).read_text(encoding="utf-8")
if not text.strip():
    raise SystemExit("PROMPT_FILE is empty")

payload = {
    "text": text,
    "sampling_params": {
        "temperature": 0,
        "max_new_tokens": int(max_new_tokens),
        "ignore_eos": True,
    },
}
pathlib.Path(output_path).write_text(
    json.dumps(payload, ensure_ascii=False), encoding="utf-8"
)
PY

echo "============================================================"
echo "SGLang URL              : ${BASE_URL}"
echo "Prompt file             : ${PROMPT_FILE}"
echo "Expected chunk size     : ${CHUNKED_PREFILL_SIZE}"
echo "Minimum prompt tokens   : ${MIN_PROMPT_TOKENS}"
echo "Baseline runs           : ${BASELINE_RUNS}"
echo "Profile steps           : ${PROFILE_STEPS}"
echo "Container trace dir     : ${SERVER_PROFILE_DIR}"
echo "Host trace dir          : ${HOST_PROFILE_DIR}"
echo "Client log dir          : ${LOCAL_LOG_DIR}"
echo "============================================================"

echo "[1/9] Checking server health"
curl --fail --silent --show-error \
    --connect-timeout 5 --max-time 15 \
    "${BASE_URL}/health" \
    > "${LOCAL_LOG_DIR}/health.log"

echo "[2/9] Reading resolved server configuration"
curl --fail --silent --show-error \
    --connect-timeout 5 --max-time 30 \
    "${BASE_URL}/server_info" \
    > "${LOCAL_LOG_DIR}/server-info.json"

python3 - "${LOCAL_LOG_DIR}/server-info.json" "${CHUNKED_PREFILL_SIZE}" <<'PY'
import json
import pathlib
import sys

path, expected_chunk = sys.argv[1:]
obj = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))

keys = {
    "chunked_prefill_size",
    "max_prefill_tokens",
    "kt_gpu_prefill_token_threshold",
    "kt_num_gpu_layers",
    "kt_num_gpu_experts",
    "kt_gpu_experts_ratio",
    "kt_expert_placement_strategy",
    "init_expert_location",
    "kt_enable_dynamic_expert_update",
    "enable_dynamic_chunking",
    "enable_mixed_chunk",
    "context_length",
    "mem_fraction_static",
    "launch_command",
}
found = {}

def walk(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in keys and key not in found:
                found[key] = child
            walk(child)
    elif isinstance(value, list):
        for child in value:
            walk(child)

walk(obj)
print("Resolved server configuration:")
for key in sorted(keys - {"launch_command"}):
    print(f"  {key:34s}: {found.get(key, '<not found>')}")
print(f"  {'launch_command':34s}: {found.get('launch_command', '<not found>')}")
print(
    "  NOTE: also inspect startup log lines beginning with "
    "'KT GPU experts: layer' to confirm the effective per-layer placement mask."
)

chunk = found.get("chunked_prefill_size")
max_prefill = found.get("max_prefill_tokens")
threshold = found.get("kt_gpu_prefill_token_threshold")
try:
    expected_chunk = int(expected_chunk)
    if chunk is not None and int(chunk) != expected_chunk:
        print(f"WARNING: resolved chunked_prefill_size={chunk}, expected {expected_chunk}")
    if max_prefill is not None and int(max_prefill) < expected_chunk:
        print(
            f"WARNING: max_prefill_tokens={max_prefill} is below the expected "
            f"chunk size {expected_chunk}; the effective prefill batch can be smaller."
        )
    if threshold is not None and chunk is not None and int(threshold) > int(chunk):
        print(
            f"WARNING: kt_gpu_prefill_token_threshold={threshold} is above the "
            f"resolved chunked_prefill_size={chunk}; a single-request chunk may "
            "not enter the layerwise/full-GPU path."
        )
    if threshold is not None and max_prefill is not None and int(threshold) > int(max_prefill):
        print(
            "WARNING: kt_gpu_prefill_token_threshold is above max_prefill_tokens; "
            "a single-request prefill batch may never enter the layerwise/full-GPU path."
        )
except (TypeError, ValueError):
    print("WARNING: could not validate numeric prefill configuration values.")
PY

echo "[3/9] Clearing a stale profiler session if one exists"
curl --silent --show-error --max-time 30 \
    -X POST "${BASE_URL}/stop_profile" \
    > "${LOCAL_LOG_DIR}/previous-stop-profile.log" 2>&1 || true

echo "[4/9] Warming the exact long-prefill shape (layerwise slots / kernels)"
flush_cache
send_request "${WARMUP_RESPONSE}" "${LOCAL_LOG_DIR}/warmup-time.txt"
report_response "warmup" "${WARMUP_RESPONSE}" "${LOCAL_LOG_DIR}/warmup-time.txt"

PROMPT_TOKENS="$(python3 - "${WARMUP_RESPONSE}" <<'PY'
import json
import pathlib
import sys
obj = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print(int((obj.get("meta_info") or {}).get("prompt_tokens") or 0))
PY
)"

if (( PROMPT_TOKENS < MIN_PROMPT_TOKENS )); then
    echo "ERROR: prompt contains only ${PROMPT_TOKENS} tokens; expected at least ${MIN_PROMPT_TOKENS}." >&2
    echo "Provide a longer representative PROMPT_FILE." >&2
    exit 1
fi

if (( PROMPT_TOKENS < CHUNKED_PREFILL_SIZE )); then
    echo "ERROR: prompt token count is below chunked-prefill-size=${CHUNKED_PREFILL_SIZE}." >&2
    exit 1
fi

echo "[5/9] Running profiler-off cold-prefill baselines"
for ((run = 1; run <= BASELINE_RUNS; run++)); do
    flush_cache
    response_file="${LOCAL_LOG_DIR}/baseline-${run}-response.json"
    time_file="${LOCAL_LOG_DIR}/baseline-${run}-time.txt"
    send_request "${response_file}" "${time_file}"
    report_response "baseline-${run}" "${response_file}" "${time_file}"
done

echo "[6/9] Flushing Radix cache before the profiled cold prefill"
flush_cache

echo "[7/9] Starting optional nvidia-smi monitoring"
if command -v nvidia-smi >/dev/null 2>&1; then
    if command -v timeout >/dev/null 2>&1; then
        timeout 1800 nvidia-smi dmon -s put -d 1 -o DT \
            > "${LOCAL_LOG_DIR}/nvidia-smi-dmon.log" 2>&1 &
    else
        nvidia-smi dmon -s put -d 1 -o DT \
            > "${LOCAL_LOG_DIR}/nvidia-smi-dmon.log" 2>&1 &
    fi
    DMON_PID=$!
else
    echo "nvidia-smi not found on this host; skipping dmon."
fi

echo "[8/9] Starting a GPU-only profiler for ${PROFILE_STEPS} forward step(s)"
curl --fail --silent --show-error \
    --connect-timeout 10 --max-time 1800 \
    -X POST "${BASE_URL}/start_profile" \
    -H "Content-Type: application/json" \
    -d @- \
    > "${PROFILE_CONTROL_LOG}" \
    2> "${PROFILE_CONTROL_ERR}" <<JSON &
{
  "output_dir": "${SERVER_PROFILE_DIR}",
  "num_steps": ${PROFILE_STEPS},
  "activities": ["GPU"],
  "with_stack": false,
  "record_shapes": false,
  "merge_profiles": false,
  "profile_prefix": "kt-prefill-16k",
  "detailed_annotations": false
}
JSON

PROFILE_PID=$!
sleep 3

send_request "${PROFILE_RESPONSE}" "${LOCAL_LOG_DIR}/profile-time.txt"
report_response "profiled-run" "${PROFILE_RESPONSE}" "${LOCAL_LOG_DIR}/profile-time.txt"

if ! wait "${PROFILE_PID}"; then
    PROFILE_PID=""
    echo "ERROR: /start_profile failed." >&2
    cat "${PROFILE_CONTROL_ERR}" >&2 || true
    exit 1
fi
PROFILE_PID=""

sleep 5

if [[ -n "${DMON_PID}" ]] && kill -0 "${DMON_PID}" 2>/dev/null; then
    kill "${DMON_PID}" 2>/dev/null || true
    wait "${DMON_PID}" 2>/dev/null || true
    DMON_PID=""
fi

echo "[9/9] Collecting trace locations"
if [[ -d "${HOST_PROFILE_DIR}" ]]; then
    echo "Trace files in bind-mounted host directory:"
    find "${HOST_PROFILE_DIR}" -maxdepth 2 -type f -ls
elif [[ -n "${CONTAINER}" ]] && command -v docker >/dev/null 2>&1; then
    copy_dir="${LOCAL_LOG_DIR}/traces"
    mkdir -p "${copy_dir}"
    echo "Bind-mounted trace directory was not found; copying from container ${CONTAINER}."
    docker cp "${CONTAINER}:${SERVER_PROFILE_DIR}/." "${copy_dir}/"
    find "${copy_dir}" -maxdepth 2 -type f -ls
else
    echo "Trace was written inside the SGLang container: ${SERVER_PROFILE_DIR}"
    echo "Either inspect the bind mount at ${HOST_PROFILE_DIR}, or run:"
    echo "  docker cp <container>:${SERVER_PROFILE_DIR}/. ${LOCAL_LOG_DIR}/traces/"
fi

echo
echo "Completed. Use profiler-off baseline results for throughput, and the GPU-only trace for timeline attribution."
echo "Client logs: ${LOCAL_LOG_DIR}"

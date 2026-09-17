#!/usr/bin/env bash
set -euo pipefail

: "${SPUR_JOB_ID:?run inside a SPUR allocation}"

IMAGE="${IMAGE:-lmsysorg/sglang:dev-dsv41-mi35x@sha256:c609f74f01b206af9664fb66da8ebef47642a7328a2478eabeebb352e014661a}"
MODEL_REPO="${MODEL_REPO:-deepseek-ai/DeepSeek-V4.1-Flash}"
MODEL_REV="${MODEL_REV:-dba1be0a40aa45a94ad051997016db3960a90277}"
AITER_SHA="${AITER_SHA:-a5d184b9f}"
SGLANG_SHA="${SGLANG_SHA:-b2ab7daa}"
INFERENCEX_SHA="${INFERENCEX_SHA:-57bec457c08e5223d7143f1e533ae7abf046ebd5}"
AB_DURATION="${AB_DURATION:-1200}"
CANONICAL_DURATION="${CANONICAL_DURATION:-3600}"
PORT="${PORT:-18893}"
NCCL_PORT="${NCCL_PORT:-28893}"

run_id="${GS_HOLD_ID:-manual}-${SPUR_JOB_ID}"
root="/var/tmp/dsv41-fmoe-canonical-${run_id}"
model="${root}/model"
src="${root}/src"
results="${root}/results"
server_name="dsv41-fmoe-${run_id//[^a-zA-Z0-9_.-]/-}"
sampler_pid=""
server_running=0

docker_cleanup_root() {
  [[ -d "${root}" ]] || return 0
  docker run --rm \
    --label "spur_job_id=${SPUR_JOB_ID}" \
    -v "${root}:/cleanup" \
    --entrypoint bash \
    "${IMAGE}" \
    -lc 'rm -rf /cleanup/* /cleanup/.[!.]* /cleanup/..?*'
  rm -rf "${root}"
}

cleanup() {
  rc=$?
  trap - EXIT
  set +e
  if [[ -n "${sampler_pid}" ]]; then
    kill "${sampler_pid}" 2>/dev/null
    wait "${sampler_pid}" 2>/dev/null
  fi
  if (( server_running )); then
    docker logs --tail 160 "${server_name}" 2>&1 || true
  fi
  docker rm -f "${server_name}" >/dev/null 2>&1 || true
  if (( rc != 0 )); then
    echo "DSV41_CANONICAL_FAILURE rc=${rc} host=$(hostname) run=${run_id}"
  fi
  docker_cleanup_root
  exit "${rc}"
}
trap cleanup EXIT

docker pull "${IMAGE}"
docker_cleanup_root
mkdir -p \
  "${model}" \
  "${src}" \
  "${results}" \
  "${root}/client-state" \
  "${root}/aiperf-cache" \
  "${root}/trace"

rocm-smi --showmeminfo vram --json >"${root}/preflight.json"
visible_devices="${HIP_VISIBLE_DEVICES:-${ROCR_VISIBLE_DEVICES:-}}"
python3 - "${root}/preflight.json" "${visible_devices}" >"${root}/gpu_ids" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1]))
observed = {}
for card, values in data.items():
    observed[int(card.removeprefix("card"))] = int(
        next(
            value
            for key, value in values.items()
            if "Total Used Memory" in key
        )
    )
selected = (
    [int(value) for value in sys.argv[2].split(",") if value]
    if sys.argv[2]
    else sorted(observed)
)
if len(selected) != 4 or any(card not in observed for card in selected):
    raise SystemExit(
        f"expected 4 allocated GPUs, selected={selected}, observed={observed}"
    )
used = {card: observed[card] for card in selected}
dirty = {card: value for card, value in used.items() if value > 5 * 2**30}
if dirty:
    raise SystemExit(f"allocated GPUs are not clean: {dirty}")
for card in selected:
    print(card)
PY
mapfile -t gpu_ids <"${root}/gpu_ids"
echo "DSV41_PREFLIGHT host=$(hostname) gpus=${gpu_ids[*]} at=$(date -Is)"

mapfile -t render_devices < <(
  printf '%s\n' /dev/dri/renderD* | sort -V
)
device_args=(--device=/dev/kfd)
for gpu_id in "${gpu_ids[@]}"; do
  render_device="${render_devices[gpu_id]:-}"
  if [[ ! -c "${render_device}" ]]; then
    echo "no render device for GPU ${gpu_id}: ${render_device:-missing}"
    exit 1
  fi
  device_args+=("--device=${render_device}")
done

python3 - "${PORT}" "${NCCL_PORT}" <<'PY'
import socket
import sys

for value in sys.argv[1:]:
    sock = socket.socket()
    sock.bind(("0.0.0.0", int(value)))
    sock.close()
PY

echo "DSV41_MODEL_DOWNLOAD_START at=$(date -Is)"
docker run --rm \
  --label "spur_job_id=${SPUR_JOB_ID}" \
  --network=host \
  -e HF_HUB_DISABLE_PROGRESS_BARS=1 \
  -v "${model}:/model" \
  --entrypoint hf \
  "${IMAGE}" \
  download "${MODEL_REPO}" \
    --revision "${MODEL_REV}" \
    --local-dir /model \
    --max-workers 16
test -f "${model}/model-00048-of-00048.safetensors"
shards=("${model}"/model-*.safetensors)
echo "DSV41_MODEL_DOWNLOAD_DONE shards=${#shards[@]} at=$(date -Is)"

git clone --filter=blob:none https://github.com/ROCm/aiter.git "${src}/aiter"
git -C "${src}/aiter" checkout "${AITER_SHA}"
git clone --filter=blob:none https://github.com/jhinpan/sglang-fork.git "${src}/sglang"
git -C "${src}/sglang" checkout "${SGLANG_SHA}"
git clone --filter=blob:none https://github.com/SemiAnalysisAI/InferenceX.git "${src}/InferenceX"
git -C "${src}/InferenceX" checkout "${INFERENCEX_SHA}"
git -C "${src}/InferenceX" submodule update --init --depth 1 utils/aiperf

python3 - \
  "${src}/aiter/aiter/configs/model_configs/dsv41_flash_fp8fp4_tuned_fmoe.csv" \
  "${root}/candidate-runtime.csv" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="") as stream:
    rows = list(csv.DictReader(stream))
with open(sys.argv[2], "w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
    writer.writeheader()
    for row in rows:
        row["topk"] = "5"
        writer.writerow(row)
PY
echo "DSV41_SERVER_RUNTIME image-pinned topk5-compat"

cat >"${root}/hbm_sampler.py" <<'PY'
import json
import subprocess
import sys
import time

selected = {f"card{value}" for value in sys.argv[1:]}
while True:
    now = time.time()
    run = subprocess.run(
        ["rocm-smi", "--showmeminfo", "vram", "--json"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if run.returncode == 0:
        try:
            for card, values in json.loads(run.stdout).items():
                if card not in selected:
                    continue
                used = next(
                    int(value)
                    for key, value in values.items()
                    if "Total Used Memory" in key
                )
                total = next(
                    int(value)
                    for key, value in values.items()
                    if "Total Memory" in key and "Used" not in key
                )
                print(
                    json.dumps(
                        {
                            "ts": now,
                            "card": card,
                            "used_bytes": used,
                            "total_bytes": total,
                        }
                    ),
                    flush=True,
                )
        except Exception as exc:
            print(json.dumps({"ts": now, "error": repr(exc)}), flush=True)
    else:
        print(
            json.dumps({"ts": now, "error": run.stderr.strip()}),
            flush=True,
        )
    time.sleep(2)
PY

cat >"${root}/run_client.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
name="$1"
port="$2"
duration="$3"
warmup="$4"

export INFMAX_CONTAINER_WORKSPACE=/ix
source /ix/benchmarks/benchmark_lib.sh
export MODEL=deepseek-ai/DeepSeek-V4.1-Flash
export MODEL_PREFIX=dsv41flash
export TP=4
export CONC=4
export KV_OFFLOADING=none
export TOTAL_CPU_DRAM_GB=0
export RESULT_DIR="/results/${name}"
export RESULT_FILENAME="${name}"
export DURATION="${duration}"
export PORT="${port}"
export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126
export AIPERF_SERVER_URL="http://localhost:${port}"
export AIPERF_SERVER_METRICS_URLS="${AIPERF_SERVER_URL}/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="sglang:"
export AIPERF_DATASET_MMAP_CACHE_DIR=/aiperf_mmap_cache
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
export AGENTIC_OUTPUT_DIR="${RESULT_DIR}"
export AIPERF_WARMUP_REQUESTS_PER_LANE="${warmup}"

install -d -m 700 "${RESULT_DIR}"
resolve_trace_source
install_agentic_deps
build_replay_cmd "${RESULT_DIR}"
printf '%s\n' "${REPLAY_CMD}" >"${RESULT_DIR}/benchmark_command.txt"
run_agentic_replay_and_write_outputs "${RESULT_DIR}"
SH
chmod 700 "${root}/run_client.sh"

baseline_config="/sgl-workspace/aiter/aiter/configs/tuned_fmoe.csv"
candidate_config="/sgl-workspace/aiter/aiter/configs/tuned_fmoe.csv:/config/dsv41.csv"

start_server() {
  arm="$1"
  config="$2"
  docker rm -f "${server_name}" >/dev/null 2>&1 || true
  docker run -d \
    --name "${server_name}" \
    --label "spur_job_id=${SPUR_JOB_ID}" \
    --network=host \
    --ipc=host \
    --shm-size=64g \
    "${device_args[@]}" \
    --group-add video \
    --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --ulimit memlock=-1 \
    -e NCCL_SOCKET_IFNAME='^tailscale0' \
    -e SGLANG_USE_AITER=1 \
    -e SGLANG_MOE_PADDING=1 \
    -e AITER_FLYDSL_FORCE_REDUCE=1 \
    -e AITER_FLYDSL_STAGE2_FP8=0 \
    -e ROCM_QUICK_REDUCE_QUANTIZATION=NONE \
    -e AITER_ONLINE_TUNE=0 \
    -e AITER_BF16_FP8_MOE_BOUND=0 \
    -e TRITON_HIP_USE_ASYNC_COPY=0 \
    -e SGLANG_DSV41_REASONING_EFFORT=high \
    -e SGLANG_SIMULATE_ACC_LEN=3.51 \
    -e SGLANG_SIMULATE_ACC_METHOD=match-expected \
    -e SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token \
    -e SGLANG_TIMEOUT_KEEP_ALIVE=900 \
    -e HF_HUB_OFFLINE=1 \
    -e HF_DATASETS_OFFLINE=1 \
    -e PYTHONUNBUFFERED=1 \
    -e PYTHONPATH=/src/sglang/python \
    -e "AITER_CONFIG_FMOE=${config}" \
    -v "${model}:/models/DeepSeek-V4.1-Flash:ro" \
    -v "${src}/sglang:/src/sglang" \
    -v "${root}/candidate-runtime.csv:/config/dsv41.csv:ro" \
    --entrypoint '' \
    "${IMAGE}" \
    python3 -m sglang.launch_server \
      --model-path /models/DeepSeek-V4.1-Flash \
      --served-model-name deepseek-ai/DeepSeek-V4.1-Flash \
      --host 0.0.0.0 \
      --port "${PORT}" \
      --nccl-port "${NCCL_PORT}" \
      --trust-remote-code \
      --tp 4 \
      --ep-size 4 \
      --disable-radix-cache \
      --mem-fraction-static 0.65 \
      --speculative-algorithm DSPARK \
      --speculative-dspark-block-size 5 \
      --cuda-graph-max-bs-decode 64 \
      --cuda-graph-backend-prefill breakable \
      --cuda-graph-max-bs-prefill 4096 \
      --max-running-requests 128 \
      --reasoning-parser auto \
      --tool-call-parser auto \
      --watchdog-timeout 3600 \
      --enable-metrics >/dev/null
  server_running=1

  for _ in $(seq 1 480); do
    if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      echo "DSV41_SERVER_READY arm=${arm} at=$(date -Is)"
      return 0
    fi
    if [[ "$(docker inspect -f '{{.State.Running}}' "${server_name}" 2>/dev/null)" != "true" ]]; then
      docker logs --tail 200 "${server_name}" 2>&1 || true
      return 1
    fi
    sleep 5
  done
  echo "server readiness timeout: ${arm}"
  return 1
}

stop_server() {
  tag="$1"
  docker logs "${server_name}" >"${root}/server-${tag}.log" 2>&1 || true
  docker rm -f "${server_name}" >/dev/null
  server_running=0
  for _ in $(seq 1 60); do
    rocm-smi --showmeminfo vram --json >"${root}/clear.json"
    if python3 - "${root}/clear.json" "${gpu_ids[@]}" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1]))
used = []
for raw in sys.argv[2:]:
    values = data[f"card{raw}"]
    used.append(
        int(
            next(
                value
                for key, value in values.items()
                if "Total Used Memory" in key
            )
        )
    )
raise SystemExit(0 if max(used) <= 5 * 2**30 else 1)
PY
    then
      return 0
    fi
    sleep 2
  done
  echo "GPU memory did not clear after ${tag}"
  return 1
}

run_client() {
  name="$1"
  duration="$2"
  warmup="$3"
  echo "DSV41_CLIENT_START name=${name} duration=${duration} at=$(date -Is)"
  docker run --rm \
    --name "dsv41-client-${name}-${run_id//[^a-zA-Z0-9_.-]/-}" \
    --label "spur_job_id=${SPUR_JOB_ID}" \
    --network=host \
    -v "${src}/InferenceX:/ix" \
    -v "${root}/run_client.sh:/run_client.sh:ro" \
    -v "${results}:/results" \
    -v "${root}/client-state:/tmp/inferencex-agentic-1" \
    -v "${root}/aiperf-cache:/aiperf_mmap_cache" \
    "${IMAGE}" \
    bash /run_client.sh "${name}" "${PORT}" "${duration}" "${warmup}"
  echo "DSV41_CLIENT_DONE name=${name} at=$(date -Is)"
}

python3 "${root}/hbm_sampler.py" "${gpu_ids[@]}" >"${root}/hbm.jsonl" 2>&1 &
sampler_pid="$!"

start_server baseline-a1 "${baseline_config}"
run_client ab-a1-baseline "${AB_DURATION}" 1
stop_server baseline-a1

start_server candidate-b1 "${candidate_config}"
run_client ab-b1-candidate "${AB_DURATION}" 1
stop_server candidate-b1

start_server baseline-a2 "${baseline_config}"
run_client ab-a2-baseline "${AB_DURATION}" 1
stop_server baseline-a2

start_server candidate-b2 "${candidate_config}"
run_client ab-b2-candidate "${AB_DURATION}" 1
run_client canonical-candidate-3600 "${CANONICAL_DURATION}" 10
curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null
test "$(docker inspect -f '{{.State.Running}}' "${server_name}")" = "true"
stop_server candidate-b2

trace_device_args=(--device=/dev/kfd "--device=${render_devices[gpu_ids[0]]}")
if docker run --rm \
  --label "spur_job_id=${SPUR_JOB_ID}" \
  "${trace_device_args[@]}" \
  --group-add video \
  --ipc=host \
  --network=host \
  -e PYTHONPATH=/src/aiter \
  -e AITER_LOG_LEVEL=ERROR \
  -e AITER_AOT_IMPORT=0 \
  -e AITER_BF16_FP8_MOE_BOUND=0 \
  -e TRITON_HIP_USE_ASYNC_COPY=0 \
  -v "${src}:/src" \
  -v "${root}/trace:/trace" \
  "${IMAGE}" \
  bash -lc '
    command -v rocprofv3
    cd /src/aiter
    rocprofv3 \
      --kernel-trace \
      --stats \
      --output-format csv \
      --output-directory /trace \
      -- \
      python3 op_tests/op_benchmarks/bench_dsv41_fmoe.py \
        --csv aiter/configs/model_configs/dsv41_flash_fp8fp4_tuned_fmoe.csv \
        --tokens 4096 \
        --ep-ranks 0 \
        --rounds 1 \
        --iterations 10 \
        --execution graph
  '
then
  trace_status=passed
else
  trace_status=failed
fi

kill "${sampler_pid}"
wait "${sampler_pid}" 2>/dev/null || true
sampler_pid=""

python3 - "${results}" "${root}/hbm.jsonl" "${root}" "${trace_status}" <<'PY'
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

results = Path(sys.argv[1])
hbm_file = Path(sys.argv[2])
root = Path(sys.argv[3])
trace_status = sys.argv[4]

names = (
    "ab-a1-baseline",
    "ab-b1-candidate",
    "ab-a2-baseline",
    "ab-b2-candidate",
    "canonical-candidate-3600",
)
for name in names:
    path = results / name / f"{name}.json"
    try:
        data = json.loads(path.read_text())
    except Exception:
        continue
    metrics = data.get("request_metrics", {})
    throughput = metrics.get("throughput", {})
    latency = metrics.get("latency", {})
    print(
        "DSV41_RESULT_SUMMARY "
        + json.dumps(
            {
                "name": path.stem,
                "requests_total": data.get("num_requests_total"),
                "requests_successful": data.get("num_requests_successful"),
                "output_tps": throughput.get("output", {}).get(
                    "tokens_per_second"
                ),
                "ttft_p50": latency.get("ttft", {}).get("p50"),
                "itl_p50": latency.get("itl", {}).get("p50"),
            },
            sort_keys=True,
        )
    )

rows = defaultdict(list)
for line in hbm_file.read_text().splitlines():
    item = json.loads(line)
    if "card" in item:
        rows[item["card"]].append(item)
hbm = {}
for card, values in sorted(rows.items()):
    end = values[-1]["ts"]
    tail = [row for row in values if row["ts"] >= end - 600]
    xs = [(row["ts"] - tail[0]["ts"]) / 60 for row in tail]
    ys = [row["used_bytes"] / 2**30 for row in tail]
    xbar = sum(xs) / len(xs)
    ybar = sum(ys) / len(ys)
    denominator = sum((x - xbar) ** 2 for x in xs)
    slope = (
        sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
        / denominator
        if denominator
        else math.nan
    )
    hbm[card] = {
        "samples": len(values),
        "peak_used_gib": max(row["used_bytes"] for row in values) / 2**30,
        "min_free_gib": min(
            (row["total_bytes"] - row["used_bytes"]) / 2**30
            for row in values
        ),
        "last_600s_slope_gib_per_min": slope,
        "last_600s_used_range_gib": max(ys) - min(ys),
    }
print("DSV41_HBM_SUMMARY " + json.dumps(hbm, sort_keys=True))

for path in sorted(root.glob("server-*.log")):
    text = path.read_text(errors="replace")
    print(
        "DSV41_SERVER_SUMMARY "
        + json.dumps(
            {
                "name": path.stem,
                "stage1": sorted(
                    set(re.findall(r"kernelName1='([^']+)'", text))
                ),
                "stage2": sorted(
                    set(re.findall(r"kernelName2='([^']+)'", text))
                ),
                "fallback_count": text.count("no tuned FlyDSL config"),
                "traceback_count": text.count(
                    "Traceback (most recent call last)"
                ),
                "error_count": len(
                    re.findall(
                        r"\b(?:ERROR|memory fault|out of memory)\b",
                        text,
                        re.I,
                    )
                ),
            },
            sort_keys=True,
        )
    )
print("DSV41_TRACE_SUMMARY " + json.dumps({"status": trace_status}))
PY

echo "DSV41_CANONICAL_COMPLETE host=$(hostname) run=${run_id} at=$(date -Is)"

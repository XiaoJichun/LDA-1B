#!/bin/bash
# LDA-1B 推理评估脚本
# 用法: bash scripts/eval.sh
# 支持自适应环境检测，所有参数有默认值，运行时可手动覆盖

set -euo pipefail

# ============================================================
# 自动检测：项目根目录（在 cd 前保存脚本绝对路径）
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_ABS_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ============================================================
# 激活 uv 虚拟环境（如存在），并确保项目根目录可 import
# ============================================================
if [ -f "${PROJECT_ROOT}/.venv/bin/activate" ]; then
    source "${PROJECT_ROOT}/.venv/bin/activate"
fi
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ============================================================
# 自动检测：网卡名
# ============================================================
AUTO_NET_IF=$(awk 'NR>2{gsub(/:/, "", $1); print $1}' /proc/net/dev 2>/dev/null \
    | grep -vE '^(lo|docker|br-|virbr|dummy|sit|tun|tap)' \
    | head -1)
if [ -z "${AUTO_NET_IF}" ]; then
    AUTO_NET_IF=$(ip -o link show 2>/dev/null \
        | awk -F': ' '{print $2}' | sed 's/@.*//' \
        | grep -vE '^(lo|docker|br-|virbr|dummy|sit|tun|tap)' \
        | head -1)
fi
AUTO_NET_IF="${AUTO_NET_IF:-lo}"

# ============================================================
# 默认参数
# ============================================================
DEFAULT_RUN_ROOT="/root/workspace/vla_repo/xiaojichun/checkpoints/lda-1b"
DEFAULT_RUN_ID="test_run"
DEFAULT_DATA_ROOT="${PROJECT_ROOT}/playground/demo_data"
DEFAULT_DATA_MIX="demo_data"
DEFAULT_TRAJS=2
DEFAULT_ACTION_HORIZON=16
DEFAULT_IS_DELTA_ACTION=false
DEFAULT_SEED=42
DEFAULT_PLOT=true
DEFAULT_CREATE_VIDEO=true

# ============================================================
# 交互式输入
# ============================================================
echo ""
echo "══════════════════════════════════════════════════"
echo "  LDA-1B Eval — 按 Enter 使用 [默认值]"
echo "══════════════════════════════════════════════════"
echo ""

_ask() {
    local label="$1" default="$2" var="$3"
    read -r -p "  ${label} [${default}]: " _input || true
    printf -v "${var}" '%s' "${_input:-${default}}"
}

_ask "Checkpoint 根目录 (run_root_dir)" "${DEFAULT_RUN_ROOT}"        RUN_ROOT
_ask "Run ID                          " "${DEFAULT_RUN_ID}"           RUN_ID

# 自动检测当前 run 下最新的 checkpoint steps
BASE_DIR="${RUN_ROOT}/${RUN_ID}"
AUTO_STEPS=""
if [ -d "${BASE_DIR}/checkpoints" ]; then
    AUTO_STEPS=$(ls "${BASE_DIR}/checkpoints"/steps_*_pytorch_model.pt 2>/dev/null \
        | grep -oE 'steps_[0-9]+' | grep -oE '[0-9]+' | sort -n | tail -1 || true)
fi
DEFAULT_STEPS="${AUTO_STEPS:-120000}"

_ask "评估的训练步数 (steps)           " "${DEFAULT_STEPS}"           STEPS
_ask "数据集根目录 (data_root_dir)     " "${DEFAULT_DATA_ROOT}"       DATA_ROOT
_ask "数据集名 (data_mix)              " "${DEFAULT_DATA_MIX}"        DATA_MIX
_ask "评估轨迹数 (trajs)               " "${DEFAULT_TRAJS}"           TRAJS
_ask "Action Horizon                   " "${DEFAULT_ACTION_HORIZON}"  ACTION_HORIZON
_ask "Delta Action 模式 (true/false)   " "${DEFAULT_IS_DELTA_ACTION}" IS_DELTA_ACTION
_ask "保存轨迹图 (true/false)          " "${DEFAULT_PLOT}"            PLOT
_ask "生成轨迹视频 (true/false)        " "${DEFAULT_CREATE_VIDEO}"    CREATE_VIDEO
_ask "随机种子                         " "${DEFAULT_SEED}"            SEED

# ============================================================
# 派生路径
# ============================================================
MODEL_PATH="${BASE_DIR}/checkpoints/steps_${STEPS}_pytorch_model.pt"
CONFIG_YAML="${BASE_DIR}/config.yaml"
PLOT_PATH="${BASE_DIR}/results/steps_${STEPS}"
VIDEO_PATH="${BASE_DIR}/trajectory_video/steps_${STEPS}"

# ============================================================
# 校验
# ============================================================
ERRORS=0

if [ ! -f "${MODEL_PATH}" ]; then
    echo ""
    echo "[ERROR] checkpoint 不存在: ${MODEL_PATH}" >&2
    # 列出可用的
    if [ -d "${BASE_DIR}/checkpoints" ]; then
        echo "        可用 checkpoint:"
        ls "${BASE_DIR}/checkpoints"/steps_*_pytorch_model.pt 2>/dev/null \
            | sed 's/^/          /' || echo "          （无）"
    fi
    ERRORS=1
fi

if [ ! -f "${CONFIG_YAML}" ]; then
    echo ""
    echo "[WARN] 未找到训练时保存的 config.yaml: ${CONFIG_YAML}"
    echo "       将使用默认训练配置 lda/config/training/LDA_pretrain.yaml"
    CONFIG_YAML="lda/config/training/LDA_pretrain.yaml"
fi

if [ ! -d "${DATA_ROOT}" ]; then
    echo ""
    echo "[ERROR] 数据集根目录不存在: ${DATA_ROOT}" >&2
    ERRORS=1
fi

if [ "${ERRORS}" -ne 0 ]; then
    exit 1
fi

mkdir -p "${PLOT_PATH}" "${VIDEO_PATH}"

# ============================================================
# 确认
# ============================================================
echo ""
echo "══════════════════════════════════════════════════"
echo "  配置确认"
echo "══════════════════════════════════════════════════"
echo "  Checkpoint:   ${MODEL_PATH}"
echo "  Config:       ${CONFIG_YAML}"
echo "  数据目录:     ${DATA_ROOT}  (mix: ${DATA_MIX})"
echo "  评估轨迹数:   ${TRAJS}"
echo "  Action Horizon: ${ACTION_HORIZON}  Delta: ${IS_DELTA_ACTION}"
echo "  轨迹图:       ${PLOT_PATH}"
echo "  视频输出:     ${VIDEO_PATH}"
echo "  网卡:         ${AUTO_NET_IF}"
echo "──────────────────────────────────────────────────"
read -r -p "  确认启动评估？[Y/n] " _confirm
_confirm="${_confirm:-Y}"
[[ "${_confirm}" =~ ^[Yy]$ ]] || { echo "已取消"; exit 0; }
echo ""

# ============================================================
# 环境变量
# ============================================================
export NCCL_SOCKET_IFNAME="${AUTO_NET_IF}"
export NCCL_IB_DISABLE=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000
export WANDB_MODE=disabled

# ============================================================
# 启动评估
# ============================================================
python lda/eval/eval_policy.py \
    --config_yaml "${CONFIG_YAML}" \
    --seed "${SEED}" \
    --evaluation.model_path "${MODEL_PATH}" \
    --datasets.vla_data.data_root_dir "${DATA_ROOT}" \
    --datasets.vla_data.data_mix "${DATA_MIX}" \
    --run_root_dir "${RUN_ROOT}" \
    --run_id "${RUN_ID}" \
    --evaluation.action_horizon "${ACTION_HORIZON}" \
    --evaluation.plot "${PLOT}" \
    --evaluation.plot_state false \
    --evaluation.save_plot_path "${PLOT_PATH}" \
    --evaluation.create_trajectory_video "${CREATE_VIDEO}" \
    --evaluation.video_output_path "${VIDEO_PATH}" \
    --evaluation.trajs "${TRAJS}" \
    --evaluation.start_traj 0 \
    --evaluation.end_traj 10000000 \
    --is_delta_action "${IS_DELTA_ACTION}"

#!/bin/bash
# LDA-1B 训练启动脚本
# 用法: bash scripts/train.sh
# 支持自适应环境检测，所有参数有默认值，运行时可手动覆盖

set -euo pipefail

# ============================================================
# 固定超参（不常改，直接在此处修改）
# ============================================================
Framework_name=QwenMMDiT
DIT_TYPE="DiT-L"
freeze_module_list='qwen_vl_interface,action_model.vision_encoder'
obs_horizon=1
state_dim=null
action_dim=138
max_num_embodiments=1
use_delta_action=false
positional_embeddings=null
TRAINING_TASK_WEIGHTS="[1,1,1,1]"
NUM_TASKS=4          # 必须与 TRAINING_TASK_WEIGHTS 长度一致
repeated_diffusion_steps=1
return_vlm_inputs=false
future_obs_index=5
data_mix=demo_data
only_policy=false
policy_and_video_gen=false
only_wo_video_gen=false
pretrained_checkpoint=null
wandb_entity=your_wandb_entity

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
# 自动检测：GPU 数量
# ============================================================
if command -v nvidia-smi &>/dev/null; then
    AUTO_NUM_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
else
    AUTO_NUM_GPU=1
fi
AUTO_NUM_GPU="${AUTO_NUM_GPU:-1}"

# ============================================================
# 自动检测：网卡名（跳过 lo / docker / br- / virbr / sit）
# 优先读 /proc/net/dev（容器内兼容），次选 ip link
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
# 默认参数（机器相关，运行时可手动覆盖）
# ============================================================
DEFAULT_BASE_VLM="/root/workspace/vla_repo/vla_data_repo/model_data/qwen"
DEFAULT_VISION_PATH="/root/workspace/vla_repo/vla_data_repo/model_data/dinov3-vits16/facebook"
DEFAULT_DATA_ROOT="${PROJECT_ROOT}/playground/demo_data"
DEFAULT_RUN_ROOT="/root/workspace/vla_repo/xiaojichun/checkpoints/lda-1b"
DEFAULT_RUN_ID="run_$(date +%Y%m%d_%H%M%S)"
DEFAULT_BATCH_SIZE=4
DEFAULT_MAX_STEPS=200000
DEFAULT_NUM_GPU="${AUTO_NUM_GPU}"

# ============================================================
# 交互式输入
# ============================================================
echo ""
echo "══════════════════════════════════════════════════"
echo "  LDA-1B Training — 按 Enter 使用 [默认值]"
echo "══════════════════════════════════════════════════"
echo "  [自动检测] GPU 数量: ${AUTO_NUM_GPU}  网卡: ${AUTO_NET_IF}"
echo "──────────────────────────────────────────────────"
echo ""

_ask() {
    local label="$1" default="$2" var="$3"
    read -r -p "  ${label} [${default}]: " _input || true
    printf -v "${var}" '%s' "${_input:-${default}}"
}

_ask "VLM 模型路径        (base_vlm)" "${DEFAULT_BASE_VLM}"   BASE_VLM
_ask "视觉编码器父目录                " "${DEFAULT_VISION_PATH}"  VISION_PATH
_ask "数据集根目录        (data_root)" "${DEFAULT_DATA_ROOT}"  DATA_ROOT
_ask "Checkpoint 输出目录            " "${DEFAULT_RUN_ROOT}"   RUN_ROOT
_ask "Run ID（本次运行名称，决定 checkpoint 子目录名）" "${DEFAULT_RUN_ID}" RUN_ID
_ask "Per-device batch size（≥${NUM_TASKS}）" "${DEFAULT_BATCH_SIZE}" BATCH_SIZE
_ask "GPU 数量 (num_processes)        " "${DEFAULT_NUM_GPU}"   NUM_PROCESSES
_ask "最大训练步数                   " "${DEFAULT_MAX_STEPS}"  MAX_STEPS

# ============================================================
# 校验
# ============================================================
if [ ! -d "${BASE_VLM}" ]; then
    echo ""
    echo "[ERROR] VLM 路径不存在: ${BASE_VLM}" >&2
    exit 1
fi
if [ ! -d "${VISION_PATH}" ]; then
    echo ""
    echo "[ERROR] 视觉编码器路径不存在: ${VISION_PATH}" >&2
    exit 1
fi
if [ ! -d "${DATA_ROOT}" ]; then
    echo ""
    echo "[ERROR] 数据集根目录不存在: ${DATA_ROOT}" >&2
    exit 1
fi
if ! [[ "${BATCH_SIZE}" =~ ^[0-9]+$ ]]; then
    echo ""
    echo "[ERROR] batch_size 必须是正整数，得到: '${BATCH_SIZE}'" >&2
    exit 1
fi
if ! [[ "${NUM_PROCESSES}" =~ ^[0-9]+$ ]]; then
    echo ""
    echo "[ERROR] num_processes 必须是正整数，得到: '${NUM_PROCESSES}'" >&2
    exit 1
fi
if [ "${BATCH_SIZE}" -lt "${NUM_TASKS}" ]; then
    echo ""
    echo "[WARN] batch_size=${BATCH_SIZE} < NUM_TASKS=${NUM_TASKS}，自动调整为 ${NUM_TASKS}"
    BATCH_SIZE=${NUM_TASKS}
fi

# ============================================================
# 生成临时 DeepSpeed 配置（自动同步 batch size 和路径）
# ============================================================
TMP_DIR=$(mktemp -d)
trap 'rm -rf "${TMP_DIR}"' EXIT

TMP_DS_JSON="${TMP_DIR}/ds_config.json"
TMP_ACCEL_YAML="${TMP_DIR}/deepspeed_zero2.yaml"

cat > "${TMP_DS_JSON}" << EOF
{
    "fp16": {"enabled": false},
    "bf16": {"enabled": true},
    "train_micro_batch_size_per_gpu": ${BATCH_SIZE},
    "train_batch_size": "auto",
    "gradient_accumulation_steps": 1,
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": true,
        "allgather_bucket_size": 5e8,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "cpu_offload": false
    },
    "gradient_clipping": 1.0,
    "steps_per_print": 10
}
EOF

cat > "${TMP_ACCEL_YAML}" << EOF
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  deepspeed_config_file: "${TMP_DS_JSON}"
  deepspeed_multinode_launcher: standard
  zero3_init_flag: false
distributed_type: DEEPSPEED
num_machines: 1
num_processes: ${NUM_PROCESSES}
EOF

# ============================================================
# 确认并启动
# ============================================================
OUTPUT_DIR="${RUN_ROOT}/${RUN_ID}"
mkdir -p "${OUTPUT_DIR}"
cp "${SCRIPT_ABS_PATH}" "${OUTPUT_DIR}/train.sh"

echo ""
echo "══════════════════════════════════════════════════"
echo "  配置确认"
echo "══════════════════════════════════════════════════"
echo "  VLM 路径:     ${BASE_VLM}"
echo "  视觉编码器:   ${VISION_PATH}"
echo "  数据目录:     ${DATA_ROOT}"
echo "  输出目录:     ${OUTPUT_DIR}"
echo "  Batch Size:   ${BATCH_SIZE}"
echo "  GPU 数量:     ${NUM_PROCESSES}"
echo "  网卡:         ${AUTO_NET_IF}"
echo "  最大步数:     ${MAX_STEPS}"
echo "──────────────────────────────────────────────────"
read -r -p "  确认启动训练？[Y/n] " _confirm
_confirm="${_confirm:-Y}"
[[ "${_confirm}" =~ ^[Yy]$ ]] || { echo "已取消"; exit 0; }
echo ""

# ============================================================
# 环境变量
# ============================================================
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_SOCKET_IFNAME="${AUTO_NET_IF}"
export NCCL_IB_DISABLE=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000
export WANDB_MODE=disabled

# ============================================================
# 启动训练
# ============================================================
accelerate launch \
  --config_file "${TMP_ACCEL_YAML}" \
  --num_processes "${NUM_PROCESSES}" \
  lda/training/train_LDA.py \
  --config_yaml lda/config/training/LDA_pretrain.yaml \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${BASE_VLM} \
  --framework.action_model.vision_encoder_path ${VISION_PATH} \
  --framework.action_model.action_model_type ${DIT_TYPE} \
  --framework.action_model.max_num_embodiments ${max_num_embodiments} \
  --framework.action_model.state_dim ${state_dim} \
  --framework.action_model.action_dim ${action_dim} \
  --framework.action_model.obs_horizon ${obs_horizon} \
  --framework.action_model.future_obs_index ${future_obs_index} \
  --framework.action_model.only_policy ${only_policy} \
  --framework.action_model.policy_and_video_gen ${policy_and_video_gen} \
  --framework.action_model.only_wo_video_gen ${only_wo_video_gen} \
  --framework.action_model.diffusion_model_cfg.positional_embeddings ${positional_embeddings} \
  --datasets.vla_data.use_delta_action ${use_delta_action} \
  --datasets.vla_data.data_root_dir ${DATA_ROOT} \
  --datasets.vla_data.training_task_weights ${TRAINING_TASK_WEIGHTS} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size ${BATCH_SIZE} \
  --datasets.vla_data.return_vlm_inputs ${return_vlm_inputs} \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps ${MAX_STEPS} \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 10 \
  --trainer.eval_interval 100 \
  --trainer.repeated_diffusion_steps ${repeated_diffusion_steps} \
  --trainer.learning_rate.base 4e-5 \
  --trainer.pretrained_checkpoint ${pretrained_checkpoint} \
  --run_root_dir ${RUN_ROOT} \
  --run_id ${RUN_ID} \
  --wandb_project lda \
  --wandb_entity ${wandb_entity} \
  --is_debug False

# LDA-1B 单机单卡训练调试全记录

> 环境：NVIDIA RTX A6000 (47GB)，PyTorch 2.6.0+cu124，Python 3.10，transformers 4.57.0，huggingface_hub 0.36.0

---

## 问题总览

从 `scripts/run_scripts/run_lerobot_datasets_LDA.sh` 启动训练，依次遭遇以下错误并逐一修复：

| # | 错误类型 | 根因 | 修复文件 |
|---|---------|------|---------|
| 1 | `FileNotFoundError: deepspeed_zero2.yaml` | 脚本工作目录不对 + yaml 内硬编码路径 | 脚本 + deepspeed_zero2.yaml |
| 2 | `CUDA error: invalid device ordinal` | 配置要求 8 GPU，实际只有 1 张 | 脚本 + deepspeed_zero2.yaml |
| 3 | `HFValidationError: Repo id must be in the form...` | 本地模型路径不存在，transformers 误当 HF Hub ID | 脚本（base_vlm 路径错误） |
| 4 | `NotImplementedError: VLM model ... not implemented` | 目录名 `qwen` 不含关键字，路由失败 | `vlm/__init__.py` |
| 5 | `ImportError: flash_attn not installed` | flash_attn 未安装且代码写死 | `QWen3.py`，`QWen2_5.py` + 安装包 |
| 6 | `HFValidationError`（DINOv3） | `vision_encoder_path` 包含了模型名，导致路径重复拼接 | 脚本 |
| 7 | `FileNotFoundError: sim_pick_place/sim_pick_place` | `data_root_dir` 包含了数据集名，导致路径重复拼接 | 脚本 |
| 8 | `NCCL Bootstrap: no socket interface found` | 网卡名写死 `eth0`，本机实际是 `eno1` | 脚本 |
| 9 | `CUDA OOM` | batch_size=64 在单卡 47GB 上远超显存 | 脚本 + ds_config.yaml |
| 10 | `UnboundLocalError: output_dict` | `finally` 块在 OOM 前未赋值就 `del` | `train_LDA.py` |
| 11 | `AssertionError: batch_size must be >= number of tasks` | batch_size=2 < 任务数 4 | 脚本 |

---

## 一、环境准备：安装 flash_attn

原始代码写死 `attn_implementation="flash_attention_2"`，但 flash_attn 未安装。

```bash
# flash_attn 2.8.3 与 PyTorch 2.6.0 有 ABI 不兼容，需指定 2.7.x
pip install "flash-attn==2.7.4.post1" --no-build-isolation
```

验证：
```bash
python3 -c "import flash_attn; print(flash_attn.__version__)"
# 输出: 2.7.4.post1
```

---

## 二、文件修改详情

### 2.1 `scripts/run_scripts/run_lerobot_datasets_LDA.sh`

**改动 1：自动切换工作目录（修复问题 1）**

脚本内所有路径均为相对于项目根目录的相对路径，从任意目录调用都需自动 cd 到项目根。

```bash
# 在脚本第一行之后添加
SCRIPT_ABS_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$0")/../.."
```

原来的 `cp $0 ${output_dir}/` 在 cd 后 `$0` 相对路径失效，改为用绝对路径：
```bash
# 原：cp $0 ${output_dir}/
# 改：
cp "${SCRIPT_ABS_PATH}" ${output_dir}/
```

---

**改动 2：GPU 进程数（修复问题 2）**

原始脚本为 8 GPU 集群设计，本机只有 1 张 GPU：

```bash
# 原：--num_processes 8
# 改：
--num_processes 1
```

---

**改动 3：NCCL 环境变量（修复问题 8）**

```bash
# 原：
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=mlx5_2,mlx5_3
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1

# 改：（本机网卡为 eno1；同时修正废弃变量名）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # 减少显存碎片
export NCCL_SOCKET_IFNAME=eno1
export NCCL_IB_DISABLE=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
```

查看本机实际网卡名：
```bash
cat /proc/net/dev | awk 'NR>2{print $1}' | tr -d ':'
# 输出：lo  eno1  wlx58d9d5cc4044  docker0
```

---

**改动 4：模型路径（修复问题 3）**

```bash
# 原：base_vlm=/root/workspace/vla_repo/vla_data_repo/model_data/Qwen3-VL-4B
# 实际目录名为 qwen，改为：
base_vlm=/root/workspace/vla_repo/vla_data_repo/model_data/qwen
```

---

**改动 5：视觉编码器路径（修复问题 6）**

代码会自动拼接模型名 `dinov3-vits16-pretrain-lvd1689m`，所以脚本只需填父目录：

```bash
# 原（含模型名，导致重复拼接）：
# vision_encoder_path=.../model_data//dinov3-vits16/facebook/dinov3-vits16-pretrain-lvd1689m

# 改（只填父目录）：
vision_encoder_path=/root/workspace/vla_repo/vla_data_repo/model_data/dinov3-vits16/facebook
```

代码拼接逻辑（`MMDiT_ActionHeader.py`）：
```python
pretrained_model_name = os.path.join(config.vision_encoder_path,
                                     f'dinov3-vit{self.vision_encoder_size}16-pretrain-lvd1689m')
```

---

**改动 6：数据集路径（修复问题 7）**

`mixtures.py` 的 `demo_data` 会自动拼接 `sim_pick_place`，所以 `data_root_dir` 只填父目录：

```bash
# 原（含数据集名，导致重复拼接）：
# data_root_dir=/root/workspace/LDA-1B/playground/demo_data/sim_pick_place

# 改：
data_root_dir=/root/workspace/LDA-1B/playground/demo_data
```

---

**改动 7：添加缺失变量（修复 undefined 变量）**

```bash
# 在脚本中 repeated_diffusion_steps 后添加：
return_vlm_inputs=false
```

---

**改动 8：Batch size（修复问题 9 + 11）**

- 任务数为 4（policy / forward_dynamics / inverse_dynamics / video_gen），batch_size 必须 ≥ 4
- 单卡 47GB 显存，模型本身约 43GB，batch_size=64 必定 OOM

```bash
# 原：--datasets.vla_data.per_device_batch_size 64
# 改：
--datasets.vla_data.per_device_batch_size 4
```

---

### 2.2 `lda/config/deepseeds/deepspeed_zero2.yaml`

**改动 1：修复硬编码路径（修复问题 1）**

```yaml
# 原：
deepspeed_config_file: "/mnt/home/liukai/code/LDA/lda/config/deepseeds/ds_config.yaml"
# 改：
deepspeed_config_file: "/root/workspace/LDA-1B/lda/config/deepseeds/ds_config.yaml"
```

**改动 2：GPU 进程数（修复问题 2）**

```yaml
# 原：num_processes: 8
# 改：
num_processes: 1
```

---

### 2.3 `lda/config/deepseeds/ds_config.yaml`

**与 per_device_batch_size 保持一致（修复问题 9）**

```json
// 原："train_micro_batch_size_per_gpu": 32
// 改：
"train_micro_batch_size_per_gpu": 4
```

---

### 2.4 `lda/model/modules/vlm/__init__.py`

**问题**：`get_vlm_model` 仅通过路径名字符串匹配 VLM 类型（检查是否含 `"Qwen3-VL"` 等关键字），本地目录名 `qwen` 不含任何关键字，导致 `NotImplementedError`。

**修复**：增加 `_infer_vlm_type()` 函数，当路径名匹配失败时，读取本地目录的 `config.json` 判断 `model_type`。

```python
def _infer_vlm_type(vlm_name: str) -> str:
    import os, json
    if "Qwen2.5-VL" in vlm_name or "nora" in vlm_name.lower():
        return "qwen2.5-vl"
    if "Qwen3-VL" in vlm_name:
        return "qwen3-vl"
    if "florence" in vlm_name.lower():
        return "florence"
    # fallback：读 config.json 识别本地模型
    if os.path.isdir(vlm_name):
        cfg_path = os.path.join(vlm_name, "config.json")
        if os.path.isfile(cfg_path):
            with open(cfg_path) as f:
                model_type = json.load(f).get("model_type", "")
            if "qwen3_vl" in model_type:
                return "qwen3-vl"
            if "qwen2_5_vl" in model_type or "qwen2.5_vl" in model_type:
                return "qwen2.5-vl"
    return "unknown"
```

---

### 2.5 `lda/model/modules/vlm/QWen3.py`

**问题**：写死 `attn_implementation="flash_attention_2"`，flash_attn 未安装时直接报错；另有 `dtype=` 拼写错误。

**修复**：自动检测 flash_attn 是否可用，不可用时降级为 PyTorch 内置的 `sdpa`。

```python
# 原：
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_id,
    attn_implementation="flash_attention_2",
    dtype=torch.bfloat16,          # 拼写错误，应为 torch_dtype
)

# 改：
try:
    import flash_attn  # noqa: F401
    _attn_impl = "flash_attention_2"
except ImportError:
    _attn_impl = "sdpa"
    logger.warning("flash_attn not installed, falling back to sdpa attention. "
                   "Install with: pip install flash-attn --no-build-isolation")

model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_id,
    attn_implementation=_attn_impl,
    torch_dtype=torch.bfloat16,    # 修正拼写
)
```

---

### 2.6 `lda/model/modules/vlm/QWen2_5.py`

同 QWen3.py，添加 flash_attn 自动检测：

```python
# 原：
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    attn_implementation="flash_attention_2",
    torch_dtype="auto",
)

# 改：
try:
    import flash_attn  # noqa: F401
    _attn_impl = "flash_attention_2"
except ImportError:
    _attn_impl = "sdpa"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    attn_implementation=_attn_impl,
    torch_dtype="auto",
)
```

---

### 2.7 `lda/training/train_LDA.py`

**问题**：`_train_step` 的 `finally` 块无条件 `del output_dict`，但当 forward 阶段发生 OOM 时 `output_dict` 尚未赋值，引发 `UnboundLocalError`。

```python
# 原：
finally:
    del output_dict

# 改：
finally:
    if 'output_dict' in locals():
        del output_dict
```

---

## 三、修改后的关键配置值汇总

| 配置项 | 原值 | 改后 |
|--------|------|------|
| `num_processes`（脚本 + yaml） | 8 | **1** |
| `base_vlm` | `.../Qwen3-VL-4B`（不存在） | `.../qwen` |
| `vision_encoder_path` | 含模型名（重复拼接） | 只含父目录 |
| `data_root_dir` | 含数据集名（重复拼接） | 只含父目录 |
| `per_device_batch_size` | 64 | **4** |
| `train_micro_batch_size_per_gpu` | 32 | **4** |
| `NCCL_SOCKET_IFNAME` | eth0 | **eno1** |
| `deepspeed_config_file` | `/mnt/home/liukai/...` | `/root/workspace/LDA-1B/...` |
| `attn_implementation` | 写死 flash_attention_2 | **自动检测，fallback sdpa** |

---

## 四、完整启动命令

```bash
cd /root/workspace/LDA-1B/scripts
bash run_scripts/run_lerobot_datasets_LDA.sh
```

脚本开头的 `cd "$(dirname "$0")/../.."` 会自动切换到项目根目录，因此从任意目录都可以调用。

---

## 五、多卡训练时的恢复方法

若后续换回多卡训练，需同步修改：

1. `scripts/run_scripts/run_lerobot_datasets_LDA.sh`：
   - `--num_processes` 改为实际 GPU 数
   - `NCCL_SOCKET_IFNAME` 改为对应 IB/以太网网卡名
   - `per_device_batch_size` 相应调大（至少 = 任务数 / GPU 数）

2. `lda/config/deepseeds/deepspeed_zero2.yaml`：
   - `num_processes` 改为实际 GPU 数

3. `lda/config/deepseeds/ds_config.yaml`：
   - `train_micro_batch_size_per_gpu` 与脚本 batch_size 保持一致

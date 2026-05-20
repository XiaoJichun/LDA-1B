# LDA-1B 训练架构分析

> 基于 `scripts/run_scripts/run_lerobot_datasets_LDA.sh` 及相关源码整理

---

## 一、整体思路

LDA-1B 是一个机器人基础模型，核心思想是**用统一的多模态扩散 Transformer（MMDiT）同时学习四类任务**，让不同质量的数据各司其职：

| 数据质量 | 对应任务 |
|---------|---------|
| 高质量遥操作数据（有精确动作标注） | Policy（动作预测） |
| 低质量脚本/半自动数据（动作标注不完整） | Forward / Inverse Dynamics |
| 无标注视频 | Video Generation（视觉预测） |

训练时不做模型切换，所有任务共享同一个骨干网络，通过 **task embedding** 区分当前样本属于哪类任务。

---

## 二、模型结构（Framework: QwenMMDiT）

入口文件：[lda/model/framework/QwenMMDiT.py](../../lda/model/framework/QwenMMDiT.py)

```
输入图像 + 语言指令
        │
        ▼
┌─────────────────────────────┐
│   VLM：Qwen3-VL-4B-Instruct │   ← 语言/视觉语义编码器
│   （默认 freeze）            │
│   输出: last_hidden_states   │
│   形状: [B, seq_len, H]      │
└──────────────┬──────────────┘
               │ vl_embs (Cross-Attn 条件)
               ▼
┌─────────────────────────────────────────────────────┐
│             MMDiT Action Head                       │
│                                                     │
│  视觉 tokens (obs_tokens)                           │
│  ┌──────────────────────────────────────────────┐   │
│  │  DINOv3-ViT-S（freeze）                      │   │
│  │  当前帧 + 未来帧 → patch embeddings           │   │
│  │  obs_merger(Linear): concat → [B, N_img, D]  │   │
│  └──────────────────────────────────────────────┘   │
│                                                     │
│  动作 tokens (action_features)                      │
│  ┌──────────────────────────────────────────────┐   │
│  │  ActionEncoder: action_dim → D               │   │
│  │  + 正弦时间步嵌入（sinusoidal t encoding）    │   │
│  │  可选前缀: state_encoder / history_encoder    │   │
│  └──────────────────────────────────────────────┘   │
│                                                     │
│  ┌──────────────────────────────────────────────┐   │
│  │            MMDiT Backbone (DiT-L)            │   │
│  │   hidden_dim=1536, num_heads=32, layers=8+   │   │
│  │                                              │   │
│  │  Self-Attn: image_tokens || action_tokens    │   │
│  │  Cross-Attn: vl_embs → image/action tokens  │   │
│  │  AdaLN-Zero: timestep t + task_embedding     │   │
│  └──────────────────────────────────────────────┘   │
│                                                     │
│  输出头：                                            │
│   action_decoder → pred_action [B, T, action_dim]  │
│   obs_projector  → pred_obs    [B, N_img, D_dino]  │
└─────────────────────────────────────────────────────┘
```

---

## 三、子模块详解

### 3.1 VLM 接口（qwen_vl_interface）

- 文件：[lda/model/modules/vlm/QWen2_5.py](../../lda/model/modules/vlm/QWen2_5.py) / `QWen3.py`
- 加载：`Qwen3-VL-4B-Instruct`（`base_vlm` 参数指定路径）
- 调用方式：`build_qwenvl_inputs(images, instructions)` → 得到 tokenized 输入
- 前向输出：取 `hidden_states[-1]`，即最后一层隐状态 `[B, seq_len, H]`
- **训练时默认 freeze**（`freeze_module_list` 包含 `qwen_vl_interface`）

### 3.2 视觉编码器（vision_encoder）

- 文件：[lda/model/modules/dinov3_vit/](../../lda/model/modules/dinov3_vit/)
- 默认：`DINOv3-ViT-S16`（`vision_encoder_type=dinov3`，`vision_encoder_size=s`）
- 支持三种：`dinov3` / `vjepa2` / `vae`
- 用途：对当前帧（`curr_imgs`）和未来帧（`future_imgs`）分别提取 patch 特征
- **训练时 freeze**（`freeze_module_list` 包含 `action_model.vision_encoder`）
- 输出经 `obs_merger`（Linear）与 `noisy_next_obs` 拼接后作为 `obs_tokens`

### 3.3 MMDiT 骨干（DiT Backbone）

- 文件：[lda/model/modules/action_model/flow_matching_head/mmdit/mmdit/mmdit_cross_attn.py](../../lda/model/modules/action_model/flow_matching_head/mmdit/mmdit/mmdit_cross_attn.py)
- 规格配置（`DIT_TYPE=DiT-L`）：

  | 参数 | 值 |
  |------|-----|
  | `input_embedding_dim` | 1536 |
  | `num_attention_heads` | 32 |
  | `attention_head_dim` | 48 |
  | `num_layers` | 8（config 默认，可调） |
  | `cross_attention_dim` | 与 VLM hidden_size 对齐 |
  | `norm_type` | `ada_norm`（AdaLN-Zero） |

- **Self-Attention**：`image_tokens || action_tokens` 拼接后做 joint self-attention，再分别输出
- **Cross-Attention**：`vl_embs`（VLM 语义）作为条件，注入图像流和动作流
- **AdaLN-Zero 调制**：每层的 scale/shift 由 `timestep_emb + task_embedding` 动态生成

### 3.4 动作编码器 / 解码器

```
ActionEncoder:
  Linear(action_dim → D)
  + SinusoidalPositionalEncoding(timestep)
  → concat → Linear(2D → D) → swish → Linear(D → D)

ActionDecoder (单具身):
  MLP: output_dim → hidden_size → action_dim

ActionDecoder (多具身):
  CategorySpecificMLP: 每个 embodiment_id 有独立权重
```

### 3.5 Task Embedding（4 类）

每个样本被分配到一个任务，对应一个可学习的向量，加到 timestep embedding 上送入 AdaLN：

| 任务 | 参数名 | 动作输入 | 未来观测输入 | 预测目标 |
|------|-------|---------|-------------|---------|
| Policy | `policy_embedding` | 加噪动作 | 可学习 token（遮蔽） | 动作速度场 |
| Inverse Dynamics | `id_embedding` | 加噪动作 | GT 未来帧特征 | 动作速度场（前N步） |
| Forward Dynamics | `fd_embedding` | GT 动作（t=1清洁） | 加噪未来帧 | 未来帧速度场 |
| Video Generation | `vg_embedding` | 可学习 token | 加噪未来帧 | 未来帧速度场 |

---

## 四、训练流程

### 4.1 数据流

```
LeRobot 格式数据集
    │
    ▼
TaskBatchSampler（task_weights=[1,1,1,1]）
    │  按权重随机分配每个样本的 assigned_task
    ▼
collate_fn → List[dict] per sample
    {image, lang, action, action_mask,
     future_image, state, embodiment_id, assigned_task, ...}
```

### 4.2 前向计算

```python
# Step 1: VLM 编码（BF16）
qwen_inputs = qwen_vl_interface.build_qwenvl_inputs(images, instructions)
vl_embs = qwen_vl_interface(**qwen_inputs).hidden_states[-1]  # [B, L, H]

# Step 2: 重复扩展（repeated_diffusion_steps 倍，增加扩散样本数）
vl_embs_repeated = vl_embs.repeat(repeated_diffusion_steps, 1, 1)

# Step 3: MMDiT Action Head 前向
output_dict = action_model(
    vl_embs=vl_embs_repeated,
    actions=actions_target_repeated,   # [B*r, T_future+1, action_dim]
    curr_imgs=curr_images_repeated,    # [B*r, V*T, C, H, W]
    future_imgs=future_images_repeated,
    embodiment_id=embodiment_ids_repeated,
    assigned_tasks=tasks * repeated_diffusion_steps,
    ...
)
```

### 4.3 Flow Matching 训练目标

采用 **Beta 分布** 采样噪声时间步 `t ∈ [0, 1)`：
```
t ~ Beta(α=1.5, β=1.0)，rescaled by noise_s=0.999
noisy_x = (1 - t) * noise + t * x_gt
velocity = x_gt - noise
loss = MSE(pred_velocity, velocity)
```

### 4.4 损失组成

```
total_loss = policy_act_loss          # 必有
           + inverse_act_loss         # 有 inverse dynamics 样本时
           + obs_loss                 # 有 forward_dynamics / video_gen 样本时
```

### 4.5 推理（Action Prediction）

Euler 积分，`num_inference_timesteps` 步（默认 4 步）：
```
x_T ~ N(0, I)  # 从噪声出发
for t in range(0, T):
    v = DiT(x_t, curr_obs, vl_embs, t, task=policy)
    x_{t+1} = x_t + (1/T) * v
return x_T  # 预测动作序列
```

---

## 五、训练配置

关键超参数（来自 `run_lerobot_datasets_LDA.sh` + `LDA_pretrain.yaml`）：

| 参数 | 值 |
|------|----|
| 框架名 | `QwenMMDiT` |
| VLM | Qwen3-VL-4B-Instruct（freeze） |
| 视觉编码器 | DINOv3-ViT-S16（freeze） |
| DiT 规格 | DiT-L（1536 dim，32 heads） |
| action_dim | 138 |
| action_horizon | 16（future_action_window_size+1） |
| obs_horizon | 1 |
| max_num_embodiments | 1（单具身） |
| repeated_diffusion_steps | 1（每 batch 扩展倍数） |
| per_device_batch_size | 64 |
| num_processes | 8（8 GPU） |
| learning_rate | 4e-5（base） |
| lr_scheduler | cosine_with_min_lr |
| max_train_steps | 200,000 |
| save_interval | 10,000 |
| optimizer | AdamW（β=[0.9,0.95]） |
| gradient_clipping | 1.0 |
| 分布式框架 | Accelerate + DeepSpeed ZeroStage2 |
| 混合精度 | BF16 |
| 梯度检查点 | 启用 |

---

## 六、冻结策略

```
freeze_module_list = 'qwen_vl_interface, action_model.vision_encoder'
```

- **Frozen**：Qwen3-VL（VLM）+ DINOv3（视觉编码器）
- **Trainable**：MMDiT backbone、ActionEncoder/Decoder、obs_merger、register_tokens、task_embeddings、positional_embeddings 等所有新增参数

> 说明：直接在 RoboCasa 数据集上训练时，解冻 VLM 可获得更好效果（注释提示）。

---

## 七、关键文件索引

| 文件 | 作用 |
|------|------|
| [lda/model/framework/QwenMMDiT.py](../../lda/model/framework/QwenMMDiT.py) | 顶层 Framework：组装 VLM + ActionHead，定义 forward/predict_action |
| [lda/model/modules/action_model/MMDiT_ActionHeader.py](../../lda/model/modules/action_model/MMDiT_ActionHeader.py) | FlowmatchingActionHead：4 任务逻辑、流匹配 loss、推理循环 |
| [lda/model/modules/vlm/QWen2_5.py](../../lda/model/modules/vlm/QWen2_5.py) | Qwen2.5-VL 接口封装 |
| [lda/model/modules/vlm/QWen3.py](../../lda/model/modules/vlm/QWen3.py) | Qwen3-VL 接口封装 |
| [lda/model/modules/dinov3_vit/](../../lda/model/modules/dinov3_vit/) | DINOv3 ViT 视觉编码器 |
| [lda/model/modules/action_model/flow_matching_head/mmdit/](../../lda/model/modules/action_model/flow_matching_head/mmdit/) | MMDiT 骨干实现 |
| [lda/config/training/LDA_pretrain.yaml](../../lda/config/training/LDA_pretrain.yaml) | 默认训练超参配置 |
| [lda/training/train_LDA.py](../../lda/training/train_LDA.py) | 训练主入口：VLATrainer |
| [scripts/run_scripts/run_lerobot_datasets_LDA.sh](../../scripts/run_scripts/run_lerobot_datasets_LDA.sh) | 启动脚本（覆盖 yaml 参数） |

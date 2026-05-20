# pip install huggingface-hub --upgrade
# pip install chardet charset-normalizer

from huggingface_hub import login, snapshot_download, whoami

import os
import sys

os.environ["LANG"] = "en_US.UTF-8"
os.environ["LC_ALL"] = "en_US.UTF-8"
sys.stdout.reconfigure(encoding='utf-8')

# 👇 只加这一行，临时加速，不污染环境
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"



# ===================== 配置区域 =====================
HF_TOKEN   = ""
NAME       = "robbyant/robotwin-clean-and-aug-lerobot"
LOCAL_DIR  = "/root/workspace/vla_repo/xiaojichun/checkpoints/lda_1b_checkpoints"
REPO_TYPE  = "dataset"   # dataset   or model
MAX_WORKERS = 16
# ====================================================

# 登录并校验是否成功
try:
    login(token=HF_TOKEN)
    user_info = whoami()
    print(f"✅ 登录成功！用户：{user_info['name']}")
except Exception as e:
    print(f"❌ 登录失败：{str(e)}")
    print("将以未登录模式继续下载（速度可能较慢）")

print(f"\n开始下载 {NAME} 模型...")
snapshot_download(
    repo_id=NAME,
    repo_type=REPO_TYPE,
    local_dir=LOCAL_DIR,
    # allow_patterns=["dit4dit_robocasa_gr1/**/*"],  # 只下载你需要的文件夹
    max_workers=MAX_WORKERS,
    resume_download=True,
)

print(f"\n🎉 {NAME} 模型下载完成！")

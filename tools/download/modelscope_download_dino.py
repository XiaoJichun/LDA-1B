#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ModelScope DINOv3 模型下载脚本
模型：facebook/dinov3-vits16-pretrain-lvd1689m
"""

import os
import sys
from pathlib import Path

# 1. 自动安装依赖
try:
    from modelscope import snapshot_download
except ImportError:
    print("正在安装 modelscope 依赖...")
    os.system(f"{sys.executable} -m pip install modelscope -q")
    from modelscope import snapshot_download

# ===================== 配置项 =====================
MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"  # 模型ID
SAVE_DIR = "/root/workspace/vla_repo/vla_data_repo/model_data/dinov3-vits16"  # 本地保存路径
FORCE_DOWNLOAD = False  # 强制重新下载
# ==================================================

def download_model():
    """下载DINOv3模型"""
    # 创建保存目录
    save_path = Path(SAVE_DIR).absolute()
    save_path.mkdir(exist_ok=True, parents=True)
    
    print(f"模型ID: {MODEL_ID}")
    print(f"保存路径: {save_path}")
    print("="*50)
    
    try:
        # 下载模型
        model_dir = snapshot_download(
            model_id=MODEL_ID,
            cache_dir=str(save_path),
            # force_download=FORCE_DOWNLOAD
        )
        
        print(f"\n✅ 下载完成！")
        print(f"模型路径: {model_dir}")
        
        # 列出文件
        print("\n📁 下载的文件：")
        for root, _, files in os.walk(model_dir):
            for f in files:
                file_path = Path(root) / f
                rel_path = file_path.relative_to(model_dir)
                size = file_path.stat().st_size / 1024 / 1024
                print(f"- {rel_path}  ({size:.2f} MB)")
                
        return model_dir
        
    except Exception as e:
        print(f"\n❌ 下载失败: {str(e)}")
        return None

if __name__ == "__main__":
    download_model()

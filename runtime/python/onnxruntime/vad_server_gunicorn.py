#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
FunASR Offline VAD FastAPI 服务 - Gunicorn 启动脚本

使用 Gunicorn 作为 WSGI/ASGI 服务器，比 uvicorn 的多 worker 模式更稳定
"""

import os
import sys

# 将当前目录添加到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 设置环境变量（如果需要）
# 这些变量会在 Gunicorn worker 启动时被读取
if len(sys.argv) > 1:
    # 从命令行参数解析（简化版）
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == '--model_dir' and i + 1 < len(sys.argv):
            os.environ['FUNASR_VAD_MODEL_DIR'] = sys.argv[i + 1]
        elif arg == '--thread_num' and i + 1 < len(sys.argv):
            os.environ['FUNASR_VAD_THREAD_NUM'] = sys.argv[i + 1]
        elif arg == '--quantize':
            os.environ['FUNASR_VAD_QUANTIZE'] = 'true'

# 导入 FastAPI 应用
from vad_server_fastapi import app

# Gunicorn 需要这个变量
application = app

if __name__ == "__main__":
    # 如果直接运行，提供使用说明
    print("=" * 60)
    print("FunASR Offline VAD FastAPI 服务 - Gunicorn 启动")
    print("=" * 60)
    print("\n使用方法:")
    print("  gunicorn -k uvicorn.workers.UvicornWorker \\")
    print("           -w 2 \\")
    print("           --bind 0.0.0.0:8000 \\")
    print("           --threads 2 \\")
    print("           --timeout 120 \\")
    print("           --access-logfile - \\")
    print("           --error-logfile - \\")
    print("           vad_server_gunicorn:application")
    print("\n参数说明:")
    print("  -k uvicorn.workers.UvicornWorker: 使用 Uvicorn worker")
    print("  -w 2: worker 进程数")
    print("  --bind 0.0.0.0:8000: 监听地址和端口")
    print("  --threads 2: 每个 worker 的线程数")
    print("  --timeout 120: 请求超时时间（秒）")
    print("=" * 60)


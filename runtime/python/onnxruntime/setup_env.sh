#!/bin/bash
# 设置运行环境变量（如果未使用 rpath）

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONNXRUNTIME_DIR=${ONNXRUNTIME_DIR:-""}
FFMPEG_DIR=${FFMPEG_DIR:-""}
FUNASR_BUILD_DIR=${FUNASR_BUILD_DIR:-"$SCRIPT_DIR/../../onnxruntime/build"}

# 设置库路径
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH

# 添加 FunASR 库路径
if [ -d "$FUNASR_BUILD_DIR/src" ]; then
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$FUNASR_BUILD_DIR/src
    echo "添加 FunASR 库路径: $FUNASR_BUILD_DIR/src"
fi

# 添加 ONNX Runtime 库路径
if [ -n "$ONNXRUNTIME_DIR" ] && [ -d "$ONNXRUNTIME_DIR/lib" ]; then
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ONNXRUNTIME_DIR/lib
    echo "添加 ONNX Runtime 库路径: $ONNXRUNTIME_DIR/lib"
fi

# 添加 FFmpeg 库路径
if [ -n "$FFMPEG_DIR" ] && [ -d "$FFMPEG_DIR/lib" ]; then
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$FFMPEG_DIR/lib
    echo "添加 FFmpeg 库路径: $FFMPEG_DIR/lib"
fi

echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo ""
echo "环境变量已设置。现在可以运行 Python 脚本了。"
echo "或者使用: source setup_env.sh"







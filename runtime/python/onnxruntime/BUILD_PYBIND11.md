# FunASR Online VAD Python Binding 构建指南

本文档介绍如何构建和使用 pybind11 封装的 FunASR Online VAD Python 扩展模块。

## ⚠️ 重要说明：Python 包 vs C++ 运行时库

FunASR 包含两个独立的组件：

1. **Python 包** (`funasr`): 
   - 通过 `pip install -e ./` 或 `pip install funasr` 安装
   - 包含 Python 代码和模型推理逻辑
   - **不会生成 C++ 运行时库**

2. **C++ 运行时库** (`libfunasr.so`):
   - 通过 CMake 编译生成
   - 位于 `runtime/onnxruntime/build/src/` 目录
   - **必须单独编译**，不会通过 `pip install` 生成

**要使用 pybind11 扩展，你需要：**
1. ✅ 安装 Python 包: `pip install -e ./` （你已经完成）
2. ✅ 编译 C++ 运行时库: `cd runtime/onnxruntime && mkdir build && cd build && cmake .. && make` （需要额外执行）
3. ✅ 编译 Python 扩展: `python setup_pybind11.py build_ext --inplace` （需要额外执行）

**简单来说**: `pip install` 只安装 Python 部分，C++ 库需要单独编译！

## 前置要求

1. **C++ 编译器**: 
   - Linux: GCC 7+ 或 Clang 5+
   - macOS: Xcode Command Line Tools
   - Windows: Visual Studio 2019+

2. **CMake**: 3.16 或更高版本

3. **Python**: 3.6 或更高版本

4. **pybind11**: 
   ```bash
   pip install pybind11
   ```
   或者通过 CMake 自动下载

5. **FunASR 运行时库**: 需要先编译 FunASR 的 C++ 运行时库

**重要说明**: 
- `pip install -e ./` 或 `pip install funasr` **只安装 Python 包**，不会编译 C++ 运行时库
- Python 包和 C++ 运行时库是**两个独立的组件**
- 要使用 pybind11 扩展，必须**额外编译** C++ 运行时库（通过 CMake）
- 编译后会生成 `runtime/onnxruntime/build/` 目录和 `libfunasr.so` 库文件

## 构建步骤

### 方法对比

| 特性 | 方法 1: CMake | 方法 2: setuptools |
|------|---------------|-------------------|
| **构建系统** | CMake (C++ 标准) | setuptools (Python 标准) |
| **适用场景** | 生产环境、CI/CD | 开发环境、快速迭代 |
| **构建目录** | 需要单独的 build 目录 | 可在源码目录直接构建 (`--inplace`) |
| **依赖管理** | 通过 CMake 变量配置 | 通过环境变量或 setup.py 配置 |
| **安装方式** | 手动复制 .so 文件 | 支持 `pip install` 或 `setup.py install` |
| **集成度** | 与 C++ 项目构建流程一致 | 更符合 Python 包管理习惯 |
| **灵活性** | 更灵活，可精确控制编译选项 | 相对简单，但配置选项较少 |
| **前置要求** | 需要先编译 FunASR runtime | **同样需要先编译 FunASR runtime** |

**重要说明：**
- **两种方法都需要先编译 FunASR 运行时库**，因为 pybind11 扩展需要链接 `libfunasr` 库
- 区别在于：方法 1 在同一个 CMake 项目中一起编译，方法 2 需要先单独编译 runtime，再编译 Python 扩展

**选择建议：**
- **方法 1 (CMake)**: 适合生产环境、需要与现有 CMake 项目集成、需要精确控制编译选项、希望统一构建流程
- **方法 2 (setuptools)**: 适合开发调试、快速测试、需要打包为 Python 包分发、更符合 Python 开发习惯

### 方法 1: 使用 CMake 构建

1. **编译 FunASR 运行时库** (如果还没有编译):
   ```bash
   cd runtime/onnxruntime
   mkdir build && cd build
   cmake .. -DONNXRUNTIME_DIR=/path/to/onnxruntime -DFFMPEG_DIR=/path/to/ffmpeg
   make -j4
   ```

2. **编译 Python 扩展模块**:
   ```bash
   cd runtime/onnxruntime/python
   mkdir build && cd build
   cmake .. \
     -DCMAKE_BUILD_TYPE=Release \
     -DONNXRUNTIME_DIR=/path/to/onnxruntime \
     -DFFMPEG_DIR=/path/to/ffmpeg \
     -DCMAKE_PREFIX_PATH=/path/to/funasr/build
   make -j4
   ```

3. **安装模块** (可选):
   ```bash
   # 将编译好的 .so (Linux/macOS) 或 .pyd (Windows) 文件复制到 Python 路径
   cp funasr_vad_online*.so /path/to/python/site-packages/
   ```

### 方法 2: 使用 setuptools 构建 (推荐用于开发)

**重要提示**: 使用 setuptools 构建**仍然需要提前用 CMake 编译 FunASR 运行时库**，因为：
- pybind11 扩展需要链接 `libfunasr` 库（libfunasr.so / libfunasr.dylib / funasr.dll）
- 这个库文件是通过 CMake 编译生成的，位于 `runtime/onnxruntime/build/src/` 目录
- setuptools 只负责编译 Python 扩展模块本身，不会编译 FunASR 运行时库

**构建步骤：**

1. **首先编译 FunASR 运行时库**（必需）:
   ```bash
   cd runtime/onnxruntime
   mkdir build && cd build
   cmake .. -DONNXRUNTIME_DIR=/path/to/onnxruntime -DFFMPEG_DIR=/path/to/ffmpeg
   make -j4
   # 这会生成 libfunasr.so (Linux) 或 libfunasr.dylib (macOS) 在 build/src/ 目录
   ```

2. **然后使用 setuptools 编译 Python 扩展**:
   ```bash
   cd runtime/python/onnxruntime
   export ONNXRUNTIME_DIR=/path/to/onnxruntime
   export FFMPEG_DIR=/path/to/ffmpeg
   export FUNASR_BUILD_DIR=/path/to/funasr/build  # 可选，默认是 ../build
   python setup_pybind11.py build_ext --inplace
   ```

**注意**: 如果使用默认路径，`setup_pybind11.py` 会自动查找 `runtime/onnxruntime/build/src/` 目录中的 `libfunasr` 库。

### 如何确定环境变量路径

#### 1. `ONNXRUNTIME_DIR` - ONNX Runtime 库路径

**含义**: ONNX Runtime 的安装目录（包含 `include/` 和 `lib/` 子目录）

**获取方法**:

**方法 A: 下载预编译版本（推荐）**
```bash
# Linux
wget https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/dep_libs/onnxruntime-linux-x64-1.14.0.tgz
tar -zxvf onnxruntime-linux-x64-1.14.0.tgz
export ONNXRUNTIME_DIR=$(pwd)/onnxruntime-linux-x64-1.14.0

# macOS (如果可用)
# 下载对应的 macOS 版本

# Windows
# 下载: https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/dep_libs/onnxruntime-win-x64-1.16.1.zip
# 解压到: d:/onnxruntime-win-x64-1.16.1
# export ONNXRUNTIME_DIR=d:/onnxruntime-win-x64-1.16.1
```

**方法 B: 从源码编译**
```bash
# 如果从源码编译了 ONNX Runtime
export ONNXRUNTIME_DIR=/path/to/onnxruntime/build/install
```

**验证路径是否正确**:
```bash
# 检查目录结构
ls $ONNXRUNTIME_DIR/include/onnxruntime/  # 应该存在
ls $ONNXRUNTIME_DIR/lib/libonnxruntime.*   # 应该存在 .so 或 .dylib 文件
```

**典型路径示例**:
- Linux: `/home/user/onnxruntime-linux-x64-1.14.0`
- macOS: `/Users/user/onnxruntime-osx-x64-1.14.0`
- Windows: `d:/onnxruntime-win-x64-1.16.1`

---

#### 2. `FFMPEG_DIR` - FFmpeg 库路径

**含义**: FFmpeg 的安装目录（包含 `include/` 和 `lib/` 子目录）

**获取方法**:

**方法 A: 下载预编译版本（推荐）**
```bash
# Linux
wget https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/dep_libs/ffmpeg-master-latest-linux64-gpl-shared.tar.xz
tar -xvf ffmpeg-master-latest-linux64-gpl-shared.tar.xz
export FFMPEG_DIR=$(pwd)/ffmpeg-master-latest-linux64-gpl-shared

# Windows
# 下载: https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/dep_libs/ffmpeg-master-latest-win64-gpl-shared.zip
# 解压到: d:/ffmpeg-master-latest-win64-gpl-shared
# export FFMPEG_DIR=d:/ffmpeg-master-latest-win64-gpl-shared
```

**方法 B: 系统安装的 FFmpeg**
```bash
# 如果通过包管理器安装（不推荐，可能版本不匹配）
# Ubuntu/Debian: sudo apt-get install ffmpeg libavcodec-dev libavformat-dev libavutil-dev
# 但通常需要指定开发库路径
export FFMPEG_DIR=/usr  # 如果头文件在 /usr/include，库在 /usr/lib
```

**验证路径是否正确**:
```bash
# 检查目录结构
ls $FFMPEG_DIR/include/libavcodec/  # 应该存在
ls $FFMPEG_DIR/lib/libavcodec.*      # 应该存在 .so 或 .dylib 文件
```

**典型路径示例**:
- Linux: `/home/user/ffmpeg-master-latest-linux64-gpl-shared`
- macOS: `/Users/user/ffmpeg-master-latest-osx64-gpl-shared`
- Windows: `d:/ffmpeg-master-latest-win64-gpl-shared`

---

#### 3. `FUNASR_BUILD_DIR` - FunASR 构建目录（可选）

**含义**: FunASR 运行时库的构建输出目录（包含 `src/` 子目录，其中有 `libfunasr.so`）

**重要**: 这个目录**不是通过 `pip install` 生成的**，必须通过 CMake 编译生成！

**获取方法**:

**步骤 1: 编译 FunASR C++ 运行时库**（必需）:
```bash
# 假设你已经 git clone 了 FunASR
cd FunASR/runtime/onnxruntime
mkdir build && cd build
cmake .. -DONNXRUNTIME_DIR=/path/to/onnxruntime -DFFMPEG_DIR=/path/to/ffmpeg
make -j4
# 编译后会在 build/src/ 目录生成 libfunasr.so
```

**步骤 2: 设置环境变量**:
```bash
# 默认值: 如果不设置，会自动使用 runtime/onnxruntime/build
# 手动设置（如果构建目录在其他位置）:
export FUNASR_BUILD_DIR=/path/to/FunASR/runtime/onnxruntime/build
```

**验证路径是否正确**:
```bash
# 检查 libfunasr 库是否存在
ls $FUNASR_BUILD_DIR/src/libfunasr.*  # 应该存在 .so 或 .dylib 文件

# 或者使用默认路径检查（从 runtime/python/onnxruntime 目录）
ls ../../onnxruntime/build/src/libfunasr.*
```

**典型路径示例**:
- 默认: `runtime/onnxruntime/build`（相对于 `setup_pybind11.py` 的父目录）
- 完整路径: `/home/user/FunASR/runtime/onnxruntime/build`

**常见问题**:
- ❌ "找不到 libfunasr.so" → 需要先运行 CMake 编译步骤
- ❌ "build 目录不存在" → 需要先创建 build 目录并运行 cmake + make
- ✅ 正确流程: `git clone` → `pip install -e ./` (安装 Python 包) → `cmake + make` (编译 C++ 库) → `python setup_pybind11.py` (编译 Python 扩展)

---

#### 完整示例

假设项目结构如下：
```
/home/user/
├── FunASR/                    # FunASR 项目根目录
│   └── runtime/
│       └── onnxruntime/
│           └── build/         # FunASR 构建目录
├── onnxruntime-linux-x64-1.14.0/  # ONNX Runtime
└── ffmpeg-master-latest-linux64-gpl-shared/  # FFmpeg
```

设置环境变量：
```bash
cd /home/user/FunASR/runtime/python/onnxruntime

export ONNXRUNTIME_DIR=/home/user/onnxruntime-linux-x64-1.14.0
export FFMPEG_DIR=/home/user/ffmpeg-master-latest-linux64-gpl-shared
# FUNASR_BUILD_DIR 可以不设置，使用默认值

python setup_pybind11.py build_ext --inplace
```

#### 快速检查脚本

创建以下脚本检查所有路径：
```bash
#!/bin/bash
echo "检查环境变量设置..."

if [ -z "$ONNXRUNTIME_DIR" ]; then
    echo "❌ ONNXRUNTIME_DIR 未设置"
else
    if [ -d "$ONNXRUNTIME_DIR/include" ] && [ -d "$ONNXRUNTIME_DIR/lib" ]; then
        echo "✅ ONNXRUNTIME_DIR=$ONNXRUNTIME_DIR (正确)"
    else
        echo "❌ ONNXRUNTIME_DIR=$ONNXRUNTIME_DIR (目录结构不正确)"
    fi
fi

if [ -z "$FFMPEG_DIR" ]; then
    echo "❌ FFMPEG_DIR 未设置"
else
    if [ -d "$FFMPEG_DIR/include" ] && [ -d "$FFMPEG_DIR/lib" ]; then
        echo "✅ FFMPEG_DIR=$FFMPEG_DIR (正确)"
    else
        echo "❌ FFMPEG_DIR=$FFMPEG_DIR (目录结构不正确)"
    fi
fi

FUNASR_BUILD_DIR=${FUNASR_BUILD_DIR:-"../build"}
if [ -f "$FUNASR_BUILD_DIR/src/libfunasr.so" ] || [ -f "$FUNASR_BUILD_DIR/src/libfunasr.dylib" ]; then
    echo "✅ FUNASR_BUILD_DIR=$FUNASR_BUILD_DIR (libfunasr 存在)"
else
    echo "❌ FUNASR_BUILD_DIR=$FUNASR_BUILD_DIR (libfunasr 不存在，需要先编译 FunASR runtime)"
fi
```

---

**以下是一个简化的 setup.py 示例**（实际项目中已提供 `setup_pybind11.py`）:

```python
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, Extension
import pybind11

ext_modules = [
    Pybind11Extension(
        "funasr_vad_online",
        ["../src/funasr_vad_online_pybind.cpp"],
        include_dirs=[
            "../include",
            "../src",
            "../third_party",
            pybind11.get_include(),
        ],
        libraries=["funasr", "onnxruntime"],
        library_dirs=["../build/src"],  # FunASR 库路径
        language="c++",
    ),
]

setup(
    name="funasr_vad_online",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
```

然后运行:
```bash
python setup_pybind11.py build_ext --inplace
```

**注意**: 
- 实际项目中已经提供了 `setup_pybind11.py` 文件，可以直接使用
- **必须先完成步骤 1（编译 FunASR 运行时库）**，否则会链接失败

### 两种方法的主要区别详解

#### 1. **构建系统差异**

**方法 1 (CMake)**:
- 使用 CMake 作为构建系统，这是 C++ 项目的标准选择
- 构建过程分为配置（cmake）和编译（make）两个阶段
- 生成标准的构建文件（Makefile 或 Visual Studio 项目文件）
- 可以更好地与现有的 C++ 项目集成

**方法 2 (setuptools)**:
- 使用 Python 的 setuptools，这是 Python 扩展模块的标准构建方式
- 构建过程由 Python 脚本控制，更符合 Python 开发者的习惯
- 可以直接使用 `pip install` 进行安装
- 更容易集成到 Python 包的发布流程中

#### 2. **构建目录和输出位置**

**方法 1 (CMake)**:
```bash
# 构建在单独的 build 目录
cd runtime/onnxruntime/python
mkdir build && cd build
cmake .. && make
# 输出文件在: build/funasr_vad_online*.so
```

**方法 2 (setuptools)**:
```bash
# 使用 --inplace 直接在源码目录生成
python setup_pybind11.py build_ext --inplace
# 输出文件在: 当前目录/funasr_vad_online*.so
```

#### 3. **依赖配置方式**

**方法 1 (CMake)**:
- 通过 CMake 变量传递路径：`-DONNXRUNTIME_DIR=...`
- 在 CMakeLists.txt 中统一管理依赖
- 适合复杂的依赖关系

**方法 2 (setuptools)**:
- 通过环境变量或 setup.py 中的硬编码路径
- 更灵活，但需要手动配置
- 适合简单的依赖关系

#### 4. **安装和分发**

**方法 1 (CMake)**:
- 需要手动复制 .so 文件到 Python 路径
- 或编写安装脚本
- 适合内部使用或特定环境

**方法 2 (setuptools)**:
- 支持 `python setup.py install` 自动安装
- 可以打包为 wheel 或 source distribution
- 可以通过 pip 安装：`pip install .`
- 适合公开发布

#### 5. **开发体验**

**方法 1 (CMake)**:
- 需要了解 CMake 语法
- 修改配置后需要重新运行 cmake
- 但可以精确控制编译选项（优化级别、链接库等）

**方法 2 (setuptools)**:
- 对 Python 开发者更友好
- 修改后直接运行即可
- 但编译选项相对固定

## 运行时库路径配置

### 问题：找不到 libfunasr.so

如果运行时遇到 `ImportError: libfunasr.so: cannot open shared object file`，需要配置库路径。

### 解决方案

**方案 1: 使用 rpath（推荐，已自动配置）**

`setup_pybind11.py` 已配置 `-rpath`，库路径会嵌入到 `.so` 文件中，通常不需要额外配置。

如果仍有问题，重新编译：
```bash
python setup_pybind11.py build_ext --inplace --force
```

**方案 2: 设置 LD_LIBRARY_PATH**

```bash
# 设置环境变量
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/FunASR/runtime/onnxruntime/build/src
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ONNXRUNTIME_DIR/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$FFMPEG_DIR/lib

# 或者使用提供的脚本
source setup_env.sh
```

**方案 3: 使用脚本自动设置**

```bash
# 使用提供的环境设置脚本
source setup_env.sh

# 然后运行 Python
python -c "import funasr_vad_online; print('✅ 成功！')"
```

## 使用方法

### 基本使用

```python
import funasr_vad_online
import numpy as np

# 初始化模型
model_path = {
    "model-dir": "/path/to/vad/model",
    "quantize": "false"  # 或 "true" 使用量化模型
}
vad = funasr_vad_online.FsmnVadOnline(model_path, thread_num=1)

# 方法 1: 从文件推理
segments = vad.infer_file("/path/to/audio.wav", sampling_rate=16000)

# 方法 2: 从 numpy 数组推理
audio_data = np.array([...], dtype=np.int16)  # int16 音频数据
segments = vad.infer_buffer(
    audio_data,
    is_final=True,
    sampling_rate=16000,
    wav_format="pcm"
)

# 方法 3: 从 bytes 推理
audio_bytes = b"..."  # 音频字节数据
segments = vad.infer_buffer_bytes(
    audio_bytes,
    is_final=True,
    sampling_rate=16000,
    wav_format="pcm"
)

# segments 格式: [[start1_ms, end1_ms], [start2_ms, end2_ms], ...]
for seg in segments:
    start_ms, end_ms = seg[0], seg[1]
    print(f"语音段: {start_ms}ms - {end_ms}ms")
```

### 流式处理示例

```python
import funasr_vad_online
import numpy as np
import librosa

# 初始化
model_path = {"model-dir": "/path/to/model", "quantize": "false"}
vad = funasr_vad_online.FsmnVadOnline(model_path)

# 加载音频
waveform, sr = librosa.load("audio.wav", sr=16000)
waveform_int16 = (waveform * 32767).astype(np.int16)

# 分块处理
chunk_size = int(sr * 0.05)  # 50ms 块
all_segments = []

for i in range(0, len(waveform_int16), chunk_size):
    chunk = waveform_int16[i:i + chunk_size]
    is_final = (i + chunk_size >= len(waveform_int16))
    
    segments = vad.infer_buffer(
        chunk,
        is_final=is_final,
        sampling_rate=sr,
        wav_format="pcm"
    )
    
    # 处理 segments...
```

## API 参考

### FsmnVadOnline 类

#### 构造函数

```python
FsmnVadOnline(model_path: dict, thread_num: int = 1)
```

- `model_path`: 字典，包含模型配置
  - `"model-dir"`: 模型目录路径（必需）
  - `"quantize"`: "true" 或 "false"（可选，默认 "false"）
- `thread_num`: 线程数（默认: 1）

#### 方法

##### infer_file(wav_path, sampling_rate=16000)

从音频文件推理。

- `wav_path`: 音频文件路径（支持 wav 和 pcm）
- `sampling_rate`: 采样率（默认: 16000）
- 返回: `List[List[int]]` - 语音段列表，格式为 `[[start_ms, end_ms], ...]`

##### infer_buffer(audio_data, is_final=True, sampling_rate=16000, wav_format="pcm")

从 numpy 数组推理。

- `audio_data`: numpy 数组，dtype 为 `np.int16`
- `is_final`: 是否为最后一块音频（默认: True）
- `sampling_rate`: 采样率（默认: 16000）
- `wav_format`: 音频格式，"pcm" 或 "wav"（默认: "pcm"）
- 返回: `List[List[int]]` - 语音段列表

##### infer_buffer_bytes(audio_bytes, is_final=True, sampling_rate=16000, wav_format="pcm")

从 bytes 对象推理。

- `audio_bytes`: bytes 对象，包含音频数据
- `is_final`: 是否为最后一块音频（默认: True）
- `sampling_rate`: 采样率（默认: 16000）
- `wav_format`: 音频格式，"pcm" 或 "wav"（默认: "pcm"）
- 返回: `List[List[int]]` - 语音段列表

## 故障排除

### CMake 编译错误

#### GTest 相关错误

**错误信息**:
```
CMake Error: The imported target "GTest::gtest" references the file
"/usr/lib/libgtest.a" but this file does not exist.
```

**原因**: glog 尝试查找 GTest 库，但系统上的 GTest 配置不完整。

**解决方案**（按优先级排序）:

**方案 1: 禁用 GTest（推荐）**
```bash
# 在运行 cmake 时禁用 GTest
cd runtime/onnxruntime/build
cmake .. \
  -DONNXRUNTIME_DIR=/path/to/onnxruntime \
  -DFFMPEG_DIR=/path/to/ffmpeg \
  -DWITH_GTEST=OFF

# 或者修改 CMakeLists.txt（已自动修复）
# 在 runtime/onnxruntime/CMakeLists.txt 中已添加:
# set(WITH_GTEST OFF CACHE BOOL "Use Google Test" FORCE)
```

**方案 2: 安装 GTest 开发库**
```bash
# Ubuntu/Debian
sudo apt-get install libgtest-dev

# CentOS/RHEL
sudo yum install gtest-devel

# macOS (使用 Homebrew)
brew install googletest
```

**方案 3: 完全禁用 glog（不推荐，除非确定不需要）**
```bash
cmake .. \
  -DONNXRUNTIME_DIR=/path/to/onnxruntime \
  -DFFMPEG_DIR=/path/to/ffmpeg \
  -DENABLE_GLOG=OFF
```

**注意**: 如果使用方案 3，某些功能可能不可用，因为 glog 用于日志记录。

#### ONNX Runtime 头文件错误

**错误信息 1**:
```
fatal error: onnxruntime_run_options_config_keys.h: No such file or directory
```

**错误信息 2**:
```
fatal error: onnxruntime/onnxruntime_cxx_api.h: No such file or directory
或
fatal error: onnxruntime_cxx_api.h: No such file or directory
```

**原因**: 
- 不同 ONNX Runtime 版本的头文件位置可能不同
- `ONNXRUNTIME_DIR` 可能设置不正确
- 头文件目录结构不匹配

**解决方案**:

**步骤 1: 检查 ONNX Runtime 目录结构**
```bash
# 检查 ONNXRUNTIME_DIR 是否正确设置
echo $ONNXRUNTIME_DIR

# 检查头文件位置（不同版本可能在不同位置）
ls $ONNXRUNTIME_DIR/include/onnxruntime_cxx_api.h          # 方式 1
ls $ONNXRUNTIME_DIR/include/onnxruntime/onnxruntime_cxx_api.h  # 方式 2

# 检查库文件
ls $ONNXRUNTIME_DIR/lib/libonnxruntime.*
```

**步骤 2: 根据实际目录结构调整**

如果头文件在 `include/onnxruntime_cxx_api.h`（直接在 include 目录下）:
```bash
# 这是标准结构，应该可以正常工作
export ONNXRUNTIME_DIR=/path/to/onnxruntime
```

如果头文件在 `include/onnxruntime/onnxruntime_cxx_api.h`（在 onnxruntime 子目录下）:
```bash
# 需要确保 CMakeLists.txt 正确设置了包含路径
# 代码已更新以支持这种情况
export ONNXRUNTIME_DIR=/path/to/onnxruntime
```

**步骤 3: 使用推荐的 ONNX Runtime 版本（推荐）**
```bash
# 下载推荐的版本（已验证兼容）
wget https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/dep_libs/onnxruntime-linux-x64-1.14.0.tgz
tar -zxvf onnxruntime-linux-x64-1.14.0.tgz
export ONNXRUNTIME_DIR=$(pwd)/onnxruntime-linux-x64-1.14.0

# 验证结构
ls $ONNXRUNTIME_DIR/include/onnxruntime_cxx_api.h  # 应该存在
ls $ONNXRUNTIME_DIR/lib/libonnxruntime.so          # 应该存在
```

**步骤 4: 清理并重新编译**
```bash
cd runtime/onnxruntime/build
rm -rf *
cmake .. \
  -DONNXRUNTIME_DIR=$ONNXRUNTIME_DIR \
  -DFFMPEG_DIR=$FFMPEG_DIR \
  -DWITH_GTEST=OFF
make -j4
```

#### FFmpeg 头文件错误

**错误信息**:
```
fatal error: libavutil/opt.h: No such file or directory
或
fatal error: libavcodec/avcodec.h: No such file or directory
```

**原因**: FFmpeg 的头文件路径配置不正确，或者 `FFMPEG_DIR` 设置错误。

**解决方案**:

**步骤 1: 检查 FFMPEG_DIR 和目录结构**
```bash
# 检查 FFMPEG_DIR 是否正确设置
echo $FFMPEG_DIR

# 检查 FFmpeg 头文件是否存在（应该存在）
ls $FFMPEG_DIR/include/libavutil/opt.h
ls $FFMPEG_DIR/include/libavcodec/avcodec.h
ls $FFMPEG_DIR/include/libavformat/avformat.h

# 检查库文件
ls $FFMPEG_DIR/lib/libavutil.*
ls $FFMPEG_DIR/lib/libavcodec.*
```

**步骤 2: 如果头文件不存在，下载推荐的 FFmpeg 版本**
```bash
# 下载推荐的 FFmpeg 版本
wget https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/dep_libs/ffmpeg-master-latest-linux64-gpl-shared.tar.xz
tar -xvf ffmpeg-master-latest-linux64-gpl-shared.tar.xz
export FFMPEG_DIR=$(pwd)/ffmpeg-master-latest-linux64-gpl-shared

# 验证结构
ls $FFMPEG_DIR/include/libavutil/opt.h  # 应该存在
ls $FFMPEG_DIR/lib/libavutil.so         # 应该存在
```

**步骤 3: 确保 CMake 配置正确**
```bash
cd runtime/onnxruntime/build
rm -rf *
cmake .. \
  -DONNXRUNTIME_DIR=$ONNXRUNTIME_DIR \
  -DFFMPEG_DIR=$FFMPEG_DIR \
  -DWITH_GTEST=OFF

# 检查 CMake 输出，确认 FFMPEG_DIR 被正确识别
# 应该看到类似: "FFMPEG_DIR: /path/to/ffmpeg"
```

**步骤 4: 如果使用系统安装的 FFmpeg（不推荐）**
```bash
# Ubuntu/Debian
sudo apt-get install libavcodec-dev libavformat-dev libavutil-dev libswresample-dev

# 然后设置 FFMPEG_DIR 指向系统路径
export FFMPEG_DIR=/usr  # 头文件在 /usr/include，库在 /usr/lib
```

**常见问题**:
- ❌ "FFMPEG_DIR 未设置" → 需要在 cmake 命令中指定 `-DFFMPEG_DIR=...`
- ❌ "头文件在错误的位置" → 确保 FFmpeg 目录结构是 `include/libavutil/` 而不是 `include/ffmpeg/libavutil/`
- ✅ 推荐使用预编译的 FFmpeg 版本，而不是系统包管理器安装的版本

#### Python 头文件错误（编译运行时库时）

**错误信息**:
```
fatal error: Python.h: No such file or directory
在编译 funasr_vad_online_pybind.cpp 时出现
```

**原因**: `funasr_vad_online_pybind.cpp` 是 Python 扩展模块，不应该在编译 FunASR 主运行时库时编译。它应该单独编译为 Python 扩展。

**解决方案**:

**方案 1: 使用修复后的代码（推荐）**
代码已更新，`src/CMakeLists.txt` 会自动排除 `funasr_vad_online_pybind.cpp`。如果仍有问题：
1. 确保使用最新版本的代码
2. 清理并重新编译：
   ```bash
   cd runtime/onnxruntime/build
   rm -rf *
   cmake .. \
     -DONNXRUNTIME_DIR=$ONNXRUNTIME_DIR \
     -DFFMPEG_DIR=$FFMPEG_DIR \
     -DWITH_GTEST=OFF
   make -j4
   ```

**方案 2: 手动排除（如果方案 1 不行）**
在 `runtime/onnxruntime/src/CMakeLists.txt` 中，确保排除了 pybind11 文件：
```cmake
list(REMOVE_ITEM files1 "${CMAKE_CURRENT_SOURCE_DIR}/funasr_vad_online_pybind.cpp")
```

**重要说明**:
- `funasr_vad_online_pybind.cpp` 是 Python 扩展模块，需要单独编译
- 它不应该包含在 `libfunasr.so` 的编译中
- Python 扩展应该使用 `setup_pybind11.py` 或 `python/CMakeLists.txt` 单独编译

#### 其他 CMake 错误

如果遇到其他 CMake 配置错误:
1. 检查所有必需的依赖是否已安装
2. 确保 `ONNXRUNTIME_DIR` 和 `FFMPEG_DIR` 路径正确
3. 清理 build 目录后重新配置: `rm -rf build && mkdir build && cd build`
4. 检查 ONNX Runtime 版本兼容性
5. 验证所有依赖库的头文件和库文件都存在
6. 确保 Python 扩展模块文件没有被包含在主库的编译中

### 导入错误

如果遇到 `ImportError: No module named 'funasr_vad_online'`:

1. 确保模块已正确编译
2. 检查 `.so` (Linux/macOS) 或 `.pyd` (Windows) 文件是否在 Python 路径中
3. 检查依赖库是否正确链接

### 链接错误

如果编译时遇到链接错误:

1. 确保 FunASR 运行时库已编译
2. 检查 `ONNXRUNTIME_DIR` 和 `FFMPEG_DIR` 是否正确设置
3. 确保所有依赖库都在链接路径中
4. 检查 `libfunasr.so` 是否存在: `ls runtime/onnxruntime/build/src/libfunasr.*`

### 运行时错误

如果运行时出错:

1. 检查模型路径是否正确
2. 确保音频格式和采样率正确
3. 查看错误消息中的详细信息
4. 检查动态库路径: `ldd funasr_vad_online*.so` (Linux) 或 `otool -L funasr_vad_online*.so` (macOS)

## 性能优化

1. **使用量化模型**: 设置 `"quantize": "true"` 可以使用更小的量化模型
2. **调整线程数**: 根据 CPU 核心数调整 `thread_num`
3. **批量处理**: 对于多个文件，可以复用同一个 `FsmnVadOnline` 实例

## 许可证

MIT License - 与 FunASR 项目相同


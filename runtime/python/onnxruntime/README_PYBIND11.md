# FunASR Online VAD Python Binding (pybind11)

这是一个使用 pybind11 将 FunASR Online VAD C API 封装为 Python 类的扩展模块。

## ⚠️ 重要：Python 包 vs C++ 运行时库

**如果你已经通过 `pip install -e ./` 安装了 FunASR，这还不够！**

- ✅ `pip install -e ./` 只安装 **Python 包**（funasr 目录下的代码）
- ❌ **不会生成** C++ 运行时库（`libfunasr.so`）
- ✅ 要使用 pybind11 扩展，需要**额外编译** C++ 运行时库

**完整流程**:
```bash
# 1. 安装 Python 包（你已经完成）
git clone https://github.com/alibaba-damo-academy/FunASR.git
cd FunASR
pip install -e ./

# 2. 编译 C++ 运行时库（需要额外执行）
cd runtime/onnxruntime
mkdir build && cd build
cmake .. -DONNXRUNTIME_DIR=/path/to/onnxruntime -DFFMPEG_DIR=/path/to/ffmpeg
make -j4  # 这会生成 libfunasr.so

# 3. 编译 Python 扩展（需要额外执行）
cd ../../python/onnxruntime
export ONNXRUNTIME_DIR=/path/to/onnxruntime
export FFMPEG_DIR=/path/to/ffmpeg
python setup_pybind11.py build_ext --inplace
```

## 文件结构

```
runtime/
├── onnxruntime/
│   ├── src/
│   │   └── funasr_vad_online_pybind.cpp  # pybind11 封装源码
│   └── python/
│       └── CMakeLists.txt                 # CMake 构建配置
└── python/onnxruntime/
    ├── setup_pybind11.py                  # setuptools 构建脚本
    ├── demo_vad_online_pybind.py         # 使用示例
    ├── BUILD_PYBIND11.md                  # 详细构建指南
    └── README_PYBIND11.md                 # 本文件
```

## 快速开始

### 1. 编译 FunASR 运行时库

首先需要编译 FunASR 的 C++ 运行时库（如果还没有编译）：

```bash
cd runtime/onnxruntime
mkdir build && cd build
cmake .. -DONNXRUNTIME_DIR=/path/to/onnxruntime -DFFMPEG_DIR=/path/to/ffmpeg
make -j4
```

### 2. 编译 Python 扩展

**重要**: 两种方法都需要先完成步骤 1（编译 FunASR 运行时库），因为 Python 扩展需要链接 `libfunasr` 库。

#### 方法 A: 使用 CMake

```bash
cd runtime/onnxruntime/python
mkdir build && cd build
cmake .. \
  -DONNXRUNTIME_DIR=/path/to/onnxruntime \
  -DFFMPEG_DIR=/path/to/ffmpeg \
  -DCMAKE_PREFIX_PATH=/path/to/funasr/build
make -j4
```

#### 方法 B: 使用 setuptools

```bash
cd runtime/python/onnxruntime
export ONNXRUNTIME_DIR=/path/to/onnxruntime
export FFMPEG_DIR=/path/to/ffmpeg
export FUNASR_BUILD_DIR=/path/to/funasr/build  # 可选，默认是 ../build
python setup_pybind11.py build_ext --inplace
```

**注意**: 如果使用默认路径，`setup_pybind11.py` 会自动查找 `runtime/onnxruntime/build/src/` 目录中的 `libfunasr` 库。

### 3. 使用示例

```python
import funasr_vad_online
import numpy as np

# 初始化模型
model_path = {
    "model-dir": "/path/to/vad/model",
    "quantize": "false"
}
vad = funasr_vad_online.FsmnVadOnline(model_path, thread_num=1)

# 从文件推理
segments = vad.infer_file("audio.wav", sampling_rate=16000)
print(f"检测到 {len(segments)} 个语音段")

# 从 numpy 数组推理
audio = np.array([...], dtype=np.int16)
segments = vad.infer_buffer(audio, is_final=True, sampling_rate=16000)
```

## API 文档

### FsmnVadOnline 类

#### 初始化

```python
vad = FsmnVadOnline(
    model_path={"model-dir": "/path/to/model", "quantize": "false"},
    thread_num=1
)
```

#### 方法

- `infer_file(wav_path, sampling_rate=16000)` - 从文件推理
- `infer_buffer(audio_data, is_final=True, sampling_rate=16000, wav_format="pcm")` - 从 numpy 数组推理
- `infer_buffer_bytes(audio_bytes, is_final=True, sampling_rate=16000, wav_format="pcm")` - 从 bytes 推理

所有方法返回格式为 `[[start_ms, end_ms], ...]` 的列表。

## 详细文档

更多信息请参考 [BUILD_PYBIND11.md](BUILD_PYBIND11.md)

## 示例代码

运行示例：

```bash
python demo_vad_online_pybind.py /path/to/model /path/to/audio.wav
```

## 注意事项

1. 需要先编译 FunASR 运行时库
2. 确保所有依赖库（onnxruntime, ffmpeg 等）已正确安装
3. 模型目录应包含 `model.onnx`、`config.yaml` 和 `am.mvn` 文件

## 许可证

MIT License - 与 FunASR 项目相同


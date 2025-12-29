# FunASR Offline VAD FastAPI 服务

基于 pybind11 封装的 C API 的 FastAPI 服务，提供离线 VAD 检测接口。

## 功能特性

- ✅ 基于 FastAPI 的 RESTful API
- ✅ 支持多种音频格式（wav, mp3, flac 等）
- ✅ 单文件检测和批量检测
- ✅ 自动音频格式转换和重采样
- ✅ 健康检查接口
- ✅ 完整的错误处理
- ✅ 支持多 worker 部署

## 前置要求

1. **编译 FunASR 运行时库**
   ```bash
   cd runtime/onnxruntime
   mkdir build && cd build
   cmake .. -DONNXRUNTIME_DIR=/path/to/onnxruntime -DFFMPEG_DIR=/path/to/ffmpeg
   make -j4
   ```

2. **编译 Python 扩展模块**
   ```bash
   cd runtime/python/onnxruntime
   export ONNXRUNTIME_DIR=/path/to/onnxruntime
   export FFMPEG_DIR=/path/to/ffmpeg
   python setup_pybind11.py build_ext --inplace
   ```

3. **安装 Python 依赖**
   ```bash
   pip install fastapi uvicorn[standard] aiofiles av numpy pydantic
   ```

## 启动服务

### 基本用法

```bash
python vad_server_fastapi.py --model_dir /path/to/vad/model
```

### 完整参数

```bash
python vad_server_fastapi.py \
    --model_dir /path/to/vad/model \
    --host 0.0.0.0 \
    --port 8000 \
    --thread_num 4 \
    --workers 2 \
    --temp_dir /tmp/funasr_vad \
    --log_level info
```

### 参数说明

- `--model_dir`: VAD 模型目录路径（必需）
- `--host`: 服务监听地址（默认: 0.0.0.0）
- `--port`: 服务端口（默认: 8000）
- `--thread_num`: 推理线程数（默认: 1）
- `--quantize`: 使用量化模型（可选）
- `--temp_dir`: 临时文件目录（默认: 系统临时目录）
- `--workers`: Uvicorn worker 数量（默认: 1，生产环境建议设置为 CPU 核心数）
- `--log_level`: 日志级别（debug/info/warning/error，默认: info）

## API 接口

### 1. 健康检查

**GET** `/health`

检查服务状态和模型加载情况。

**响应示例：**
```json
{
    "status": "healthy",
    "model_loaded": true,
    "model_dir": "/path/to/vad/model"
}
```

### 2. VAD 检测（单文件）

**POST** `/vad/detect`

上传音频文件进行 VAD 检测。

**请求参数：**
- `audio`: 音频文件（multipart/form-data）
- `sampling_rate`: 采样率（可选，默认: 16000）
- `wav_format`: 音频格式（可选，默认: auto）

**响应示例：**
```json
{
    "code": 0,
    "message": "success",
    "segments": [
        [100, 2500],
        [3000, 5500],
        [6000, 8500]
    ],
    "total_segments": 3,
    "audio_duration": 10.5,
    "processing_time": 0.234
}
```

**字段说明：**
- `code`: 状态码（0: 成功，非0: 失败）
- `message`: 状态消息
- `segments`: 语音段列表，每个段为 `[start_ms, end_ms]`
- `total_segments`: 检测到的语音段总数
- `audio_duration`: 音频时长（秒）
- `processing_time`: 处理耗时（秒）

### 3. VAD 检测（批量）

**POST** `/vad/detect_batch`

批量上传音频文件进行 VAD 检测。

**请求参数：**
- `audio_files`: 音频文件列表（multipart/form-data）
- `sampling_rate`: 采样率（可选，默认: 16000）

**响应示例：**
```json
{
    "code": 0,
    "message": "batch processing completed",
    "total_files": 2,
    "results": [
        {
            "code": 0,
            "message": "success",
            "segments": [[100, 2500]],
            "total_segments": 1,
            "audio_duration": 5.0,
            "processing_time": 0.123
        },
        {
            "code": 0,
            "message": "success",
            "segments": [[200, 3000], [4000, 6000]],
            "total_segments": 2,
            "audio_duration": 8.0,
            "processing_time": 0.234
        }
    ]
}
```

## 使用示例

### Python 客户端示例

```python
import requests

# 单文件检测
url = "http://localhost:8000/vad/detect"
files = {"audio": open("audio.wav", "rb")}
data = {"sampling_rate": 16000}
response = requests.post(url, files=files, data=data)
result = response.json()
print(result)

# 批量检测
url = "http://localhost:8000/vad/detect_batch"
files = [
    ("audio_files", open("audio1.wav", "rb")),
    ("audio_files", open("audio2.wav", "rb"))
]
data = {"sampling_rate": 16000}
response = requests.post(url, files=files, data=data)
result = response.json()
print(result)
```

### cURL 示例

```bash
# 单文件检测
curl -X POST "http://localhost:8000/vad/detect" \
    -F "audio=@audio.wav" \
    -F "sampling_rate=16000"

# 健康检查
curl "http://localhost:8000/health"
```

### JavaScript/TypeScript 示例

```javascript
// 单文件检测
const formData = new FormData();
formData.append('audio', audioFile);
formData.append('sampling_rate', '16000');

const response = await fetch('http://localhost:8000/vad/detect', {
    method: 'POST',
    body: formData
});

const result = await response.json();
console.log(result);
```

## 部署建议

### 开发环境

```bash
python vad_server_fastapi.py --model_dir /path/to/model --port 8000
```

### 生产环境

使用 Gunicorn + Uvicorn workers：

```bash
pip install gunicorn

gunicorn vad_server_fastapi:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 120
```

或者直接使用 Uvicorn（推荐）：

```bash
uvicorn vad_server_fastapi:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4 \
    --log-level info
```

### Docker 部署

创建 `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 复制代码
COPY . .

# 安装 Python 依赖
RUN pip install fastapi uvicorn[standard] aiofiles av numpy pydantic

# 暴露端口
EXPOSE 8000

# 启动服务
CMD ["python", "vad_server_fastapi.py", "--model_dir", "/app/models", "--host", "0.0.0.0", "--port", "8000"]
```

构建和运行：

```bash
docker build -t funasr-vad-server .
docker run -p 8000:8000 -v /path/to/models:/app/models funasr-vad-server
```

## 性能优化

1. **多 Worker 部署**: 使用 `--workers` 参数设置多个 worker 进程
2. **线程数调整**: 使用 `--thread_num` 参数调整推理线程数
3. **使用量化模型**: 添加 `--quantize` 参数使用量化模型以提升速度
4. **临时目录优化**: 使用 SSD 作为临时目录以提升 I/O 性能

## 故障排查

### 模型加载失败

- 检查模型目录路径是否正确
- 确认模型文件完整性
- 查看日志中的详细错误信息

### 音频处理失败

- 确认音频文件格式支持
- 检查音频文件是否损坏
- 查看临时目录权限

### 服务无法启动

- 确认端口未被占用
- 检查 Python 扩展模块是否已编译
- 确认所有依赖已安装

## API 文档

启动服务后，访问以下地址查看自动生成的 API 文档：

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 许可证

MIT License


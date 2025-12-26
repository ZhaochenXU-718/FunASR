mode=debug #[debug|release]
onnxruntime_dir=`pwd`/../onnxruntime/third_party/onnxruntime-linux-x64-1.14.0
ffmpeg_dir=`pwd`/../onnxruntime/third_party/ffmpeg-N-111383-g20b8688092-linux64-gpl-shared

# 检查依赖目录是否存在
if [ ! -d "$onnxruntime_dir" ]; then
    echo "Error: onnxruntime directory not found at $onnxruntime_dir"
    echo "Please download onnxruntime first"
    exit 1
fi

if [ ! -d "$ffmpeg_dir" ]; then
    echo "Error: ffmpeg directory not found at $ffmpeg_dir"
    echo "Please download ffmpeg first"
    exit 1
fi

# 检查系统json库
if ! pkg-config --exists nlohmann_json; then
    echo "Error: nlohmann-json3-dev not found"
    echo "Please install: sudo apt install nlohmann-json3-dev"
    exit 1
fi

echo "Found nlohmann/json: $(pkg-config --modversion nlohmann_json)"

# 清理并重新构建
rm -rf build
mkdir -p build
cd build

echo "Configuring CMake..."
cmake -DCMAKE_BUILD_TYPE=$mode ../ \
  -DONNXRUNTIME_DIR=$onnxruntime_dir \
  -DFFMPEG_DIR=$ffmpeg_dir \
  -DCMAKE_VERBOSE_MAKEFILE=ON

if [ $? -ne 0 ]; then
    echo "CMake configuration failed!"
    exit 1
fi

echo "Building..."
cmake --build . -j 4

if [ $? -eq 0 ]; then
    echo "Build server successfully!"
    echo "VAD server binary: ./build/bin/funasr-vad-server"
    ls -la bin/
else
    echo "Build failed!"
    exit 1
fi
# CMakeLists.txt for pybind11 Python extension
cmake_minimum_required(VERSION 3.16)

# 查找 pybind11
find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
    # 如果找不到 pybind11，尝试使用 FetchContent
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# 设置 Python 扩展模块
pybind11_add_module(funasr_vad_online
    ../src/funasr_vad_online_pybind.cpp
)

pybind11_add_module(funasr_vad
    ../src/funasr_vad_pybind.cpp
)

# 包含目录
target_include_directories(funasr_vad_online PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/third_party
    ${ONNXRUNTIME_DIR}/include
    ${FFMPEG_DIR}/include
)

# 链接库
if(WIN32)
    target_link_directories(funasr_vad_online PRIVATE
        ${ONNXRUNTIME_DIR}/lib
        ${FFMPEG_DIR}/lib
    )
    target_link_libraries(funasr_vad_online PRIVATE
        funasr
        onnxruntime
        avutil
        avcodec
        avformat
        swresample
    )
    target_compile_definitions(funasr_vad_online PRIVATE
        -DNOMINMAX
    )
else()
    target_link_directories(funasr_vad_online PRIVATE
        ${ONNXRUNTIME_DIR}/lib
        ${FFMPEG_DIR}/lib
    )
    target_link_libraries(funasr_vad_online PRIVATE
        funasr
        onnxruntime
        avutil
        avcodec
        avformat
        swresample
        pthread
    )
endif()

# 设置输出目录（可选，用于安装到 Python 包目录）
set_target_properties(funasr_vad_online PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN ON
    POSITION_INDEPENDENT_CODE ON
)

# 设置输出名称（Python 模块名）
set_target_properties(funasr_vad_online PROPERTIES
    OUTPUT_NAME "funasr_vad_online"
    PREFIX ""
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)

# 为离线 VAD 模块设置相同的包含目录和链接库
target_include_directories(funasr_vad PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/third_party
    ${ONNXRUNTIME_DIR}/include
    ${FFMPEG_DIR}/include
)

if(WIN32)
    target_link_directories(funasr_vad PRIVATE
        ${ONNXRUNTIME_DIR}/lib
        ${FFMPEG_DIR}/lib
    )
    target_link_libraries(funasr_vad PRIVATE
        funasr
        onnxruntime
        avutil
        avcodec
        avformat
        swresample
    )
    target_compile_definitions(funasr_vad PRIVATE
        -DNOMINMAX
    )
else()
    target_link_directories(funasr_vad PRIVATE
        ${ONNXRUNTIME_DIR}/lib
        ${FFMPEG_DIR}/lib
    )
    target_link_libraries(funasr_vad PRIVATE
        funasr
        onnxruntime
        avutil
        avcodec
        avformat
        swresample
        pthread
    )
endif()

set_target_properties(funasr_vad PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN ON
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME "funasr_vad"
    PREFIX ""
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)


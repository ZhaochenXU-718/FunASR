#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
setup.py for building FunASR Online VAD pybind11 extension

使用方法:
    python setup_pybind11.py build_ext --inplace
    python setup_pybind11.py install
"""

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, Extension
import pybind11
import os
import sys

# 获取项目根目录
project_root = os.path.dirname(os.path.abspath(__file__))
onnxruntime_dir = os.path.join(project_root, "..", "..", "onnxruntime")
src_dir = os.path.join(onnxruntime_dir, "src")
include_dir = os.path.join(onnxruntime_dir, "include")
third_party_dir = os.path.join(onnxruntime_dir, "third_party")

# 检查必要的目录
if not os.path.exists(src_dir):
    print(f"错误: 找不到源码目录: {src_dir}")
    sys.exit(1)

# 获取环境变量中的库路径
onnxruntime_lib_dir = os.environ.get("ONNXRUNTIME_DIR", "")
ffmpeg_lib_dir = os.environ.get("FFMPEG_DIR", "")
funasr_build_dir = os.environ.get("FUNASR_BUILD_DIR", os.path.join(onnxruntime_dir, "build"))

# 构建扩展模块
def create_ext_module(name, source_file):
    """创建扩展模块的辅助函数"""
    library_dirs = [
        os.path.join(funasr_build_dir, "src"),
        os.path.join(onnxruntime_lib_dir, "lib") if onnxruntime_lib_dir else "",
        os.path.join(ffmpeg_lib_dir, "lib") if ffmpeg_lib_dir else "",
    ]
    # 过滤掉空字符串
    library_dirs = [d for d in library_dirs if d]
    
    extra_link_args = [
        f"-Wl,-rpath,{os.path.join(funasr_build_dir, 'src')}",
    ] + ([f"-Wl,-rpath,{os.path.join(onnxruntime_lib_dir, 'lib')}"] if onnxruntime_lib_dir else []) + \
      ([f"-Wl,-rpath,{os.path.join(ffmpeg_lib_dir, 'lib')}"] if ffmpeg_lib_dir else [])
    
    return Pybind11Extension(
        name,
        [
            os.path.join(src_dir, source_file),
        ],
        include_dirs=[
            include_dir,
            src_dir,
            third_party_dir,
            os.path.join(third_party_dir, "kaldi-native-fbank"),
            os.path.join(third_party_dir, "yaml-cpp", "include"),
            os.path.join(third_party_dir, "jieba", "include"),
            os.path.join(third_party_dir, "kaldi"),
            pybind11.get_include(),
        ],
        libraries=[
            "funasr",
            "onnxruntime",
            "avutil",
            "avcodec",
            "avformat",
            "swresample",
            "pthread",
        ],
        library_dirs=library_dirs,
        extra_link_args=extra_link_args,
        language="c++",
        cxx_std=14,
    )

ext_modules = [
    create_ext_module("funasr_vad_online", "funasr_vad_online_pybind.cpp"),
    create_ext_module("funasr_vad", "funasr_vad_pybind.cpp"),
]

setup(
    name="funasr_vad",
    version="0.1.0",
    description="FunASR VAD Python Binding (Offline and Online)",
    long_description="Python binding for FunASR VAD (Offline and Online) using pybind11",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.6",
)


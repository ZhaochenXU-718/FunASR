#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
FunASR Offline VAD FastAPI 服务

基于 pybind11 封装的 C API 的 FastAPI 服务，提供 VAD 检测接口
"""

import argparse
import logging
import os
import uuid
import tempfile
from pathlib import Path
from typing import Optional, List

import aiofiles
import uvicorn
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from concurrent.futures import ProcessPoolExecutor
import asyncio

# 导入必要的模块
try:
    import funasr_vad
except ImportError:
    print("错误: 无法导入 funasr_vad 模块")
    print("请确保已经编译了 Python 扩展模块:")
    print("  python setup_pybind11.py build_ext --inplace")
    exit(1)

# 导入音频处理函数
import av
from av.audio.fifo import AudioFifo
from av.audio.resampler import AudioResampler
from av.audio.format import AudioFormat
import io

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 全局变量
vad_model = None
temp_dir = None
args = None  # 将在 main 中设置


def decode_fifo(data: bytes, sampling_rate: int) -> np.ndarray:
    """
    解码音频数据并直接转换为 int16 格式
    
    返回 int16 类型的 numpy 数组，范围 [-32768, 32767]
    这样可以减少后续格式转换的开销，同时不损失精度
    """
    buf = io.BytesIO(data)
    container = av.open(buf)
    # 初始化重采样器与 FIFO
    fmt = AudioFormat('flt') 
    resampler = AudioResampler(format=fmt.packed, layout='mono', rate=sampling_rate)
    fifo = AudioFifo()
    # 解码并写入 FIFO
    for frame in container.decode(audio=0):
        for of in resampler.resample(frame):
            of.pts = None         # 清除 PTS
            fifo.write(of)
    # 从 FIFO 中一次性读取所有样本
    output_frame = fifo.read(samples=0, partial=True)
    if output_frame is None:
        raise ValueError("AudioFifo 读取失败或无数据。")
    # 将输出帧转换为 NumPy (float32, 范围 [-1.0, 1.0])
    pcm2d = output_frame.to_ndarray()
    pcm1d = pcm2d.reshape(-1).astype(np.float32)
    
    # 直接转换为 int16，避免后续转换开销
    # 使用 clip 确保值在有效范围内，防止溢出
    # 注意：正常音频数据应该在 [-1.0, 1.0] 范围内，但为了安全起见进行裁剪
    pcm_int16 = np.clip(pcm1d * 32768.0, -32768.0, 32767.0).astype(np.int16)
    
    return pcm_int16

async def parallel_decode(payloads: list[bytes], sampling_rate: int):
    """
    并行解码一组音频负载，返回对应的 int16 PCM 数组列表。
    
    Returns:
        List[np.ndarray]: int16 类型的 numpy 数组列表
    """
    loop = asyncio.get_running_loop()  # 获取当前事件循环
    # 根据 CPU 核心数配置进程池
    with ProcessPoolExecutor(max_workers=4) as executor:
        # 将 decode_fifo 调度到进程池
        tasks = [
            loop.run_in_executor(executor, decode_fifo, data, sampling_rate)
            for data in payloads
        ]
        # 并发等待所有任务完成
        results = await asyncio.gather(*tasks)
    return results  # List[np.ndarray] (int16 dtype)

async def decode_chunks_with_resample(chunks: List[UploadFile], sampling_rate: int = 16000):
    """
    并行解码多个音频文件
    
    Args:
        chunks: UploadFile 列表
        sampling_rate: 采样率，默认16000
    
    Returns:
        List[np.ndarray]: int16 类型的 numpy 数组列表
    """
    chunk_datas = []
    for chunk in chunks:
        audio_data = await chunk.read()
        chunk_datas.append(audio_data)
    audio_datas = await parallel_decode(chunk_datas, sampling_rate)
    return audio_datas

def load_audio(file_path):
    """
    加载音频文件并返回 int16 格式的 numpy 数组
    
    Args:
        file_path: 音频文件路径
        
    Returns:
        np.ndarray: int16 类型的音频数据，范围 [-32768, 32767]
    """
    with open(file_path, "rb") as f:
        file = f.read()
    sr = 16000
    y = decode_fifo(file, sr)
    return y


# Pydantic 模型定义
class VADResponse(BaseModel):
    """VAD 检测响应模型"""
    code: int
    message: str
    segments: List[List[int]]  # [[start_ms, end_ms], ...]
    total_segments: int
    audio_duration: float  # 秒
    processing_time: float  # 秒


class HealthResponse(BaseModel):
    """健康检查响应模型"""
    status: str
    model_loaded: bool
    model_dir: Optional[str] = None


# FastAPI 应用
app = FastAPI(
    title="FunASR Offline VAD API",
    description="基于 pybind11 C API 的离线 VAD 检测服务",
    version="1.0.0"
)


# 注意：这里使用 on_event 是为了兼容性
# 在 FastAPI 0.100+ 版本中，推荐使用 lifespan 上下文管理器
@app.on_event("startup")
async def startup_event():
    """应用启动时初始化模型"""
    global vad_model, temp_dir, args
    if args is None:
        logger.warning("参数未初始化，跳过模型加载")
        return
    if vad_model is None and args.model_dir:
        logger.info(f"正在加载 VAD 模型: {args.model_dir}")
        try:
            model_path = {
                "model-dir": args.model_dir,
                "quantize": "false" if not args.quantize else "true"
            }
            vad_model = funasr_vad.FsmnVad(model_path, thread_num=args.thread_num)
            logger.info("VAD 模型加载成功")
        except Exception as e:
            logger.error(f"VAD 模型加载失败: {e}")
            raise


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理资源"""
    global vad_model, temp_dir
    if vad_model is not None:
        vad_model = None
        logger.info("VAD 模型已释放")
    
    # 清理临时目录（可选，通常保留以便调试）
    # if temp_dir and os.path.exists(temp_dir):
    #     import shutil
    #     try:
    #         shutil.rmtree(temp_dir)
    #         logger.info(f"临时目录已清理: {temp_dir}")
    #     except Exception as e:
    #         logger.warning(f"清理临时目录失败: {e}")


@app.get("/", tags=["Root"])
async def root():
    """根路径"""
    return {
        "service": "FunASR Offline VAD API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """健康检查接口"""
    global args
    return HealthResponse(
        status="healthy" if vad_model is not None else "unhealthy",
        model_loaded=vad_model is not None,
        model_dir=args.model_dir if (vad_model is not None and args is not None) else None
    )


@app.post("/vad/detect", response_model=VADResponse, tags=["VAD"])
async def vad_detect(
    audio: UploadFile = File(..., description="音频文件 (支持 wav, mp3, flac 等格式)"),
    sampling_rate: int = Form(16000, description="采样率，默认16000"),
    wav_format: str = Form("auto", description="音频格式，'auto' 自动检测，'pcm' 或 'wav'")
):
    """
    VAD 检测接口
    
    上传音频文件，返回检测到的语音段列表
    
    Args:
        audio: 音频文件
        sampling_rate: 采样率（默认16000）
        wav_format: 音频格式（默认auto，自动检测）
    
    Returns:
        VADResponse: 包含检测到的语音段信息
    """
    import time
    start_time = time.time()
    
    if vad_model is None:
        raise HTTPException(status_code=503, detail="VAD 模型未加载")
    
    try:
        # 直接读取上传的文件内容到内存（无需保存到临时文件）
        content = await audio.read()
        logger.info(f"收到音频文件: {audio.filename}, 大小: {len(content)} 字节")
        
        # 直接从内存解码音频（已直接返回 int16 格式）
        try:
            audio_int16 = decode_fifo(content, sampling_rate)
            audio_duration = len(audio_int16) / sampling_rate
            
            logger.info(f"音频解码成功: {len(audio_int16)} 采样点, 时长: {audio_duration:.3f}s, dtype: {audio_int16.dtype}")
            
        except Exception as e:
            logger.error(f"音频解码失败: {e}")
            raise HTTPException(status_code=400, detail=f"音频解码失败: {str(e)}")
        
        # VAD 推理
        try:
            segments = vad_model.infer_buffer(
                audio_int16,
                sampling_rate=sampling_rate,
                wav_format="pcm"
            )
            
            processing_time = time.time() - start_time
            
            logger.info(f"VAD 检测完成: 检测到 {len(segments)} 个语音段, 耗时: {processing_time:.3f}s")
            
            return VADResponse(
                code=0,
                message="success",
                segments=segments,
                total_segments=len(segments),
                audio_duration=audio_duration,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"VAD 推理失败: {e}")
            raise HTTPException(status_code=500, detail=f"VAD 推理失败: {str(e)}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"处理请求时发生错误: {e}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")


@app.post("/vad/detect_batch", tags=["VAD"])
async def vad_detect_batch(
    audio_files: List[UploadFile] = File(..., description="音频文件列表"),
    sampling_rate: int = Form(16000, description="采样率，默认16000")
):
    """
    批量 VAD 检测接口（优化版：使用并行解码）
    
    上传多个音频文件，使用并行解码和批量推理，返回每个文件的检测结果
    
    Args:
        audio_files: 音频文件列表
        sampling_rate: 采样率（默认16000）
    
    Returns:
        dict: 包含所有文件的检测结果
    """
    import time
    batch_start_time = time.time()
    
    if vad_model is None:
        raise HTTPException(status_code=503, detail="VAD 模型未加载")
    
    if not audio_files:
        raise HTTPException(status_code=400, detail="音频文件列表为空")
    
    logger.info(f"收到批量 VAD 检测请求: {len(audio_files)} 个文件")
    
    results = []
    file_names = [f.filename for f in audio_files]
    
    try:
        # 步骤1: 并行解码所有音频文件
        decode_start_time = time.time()
        try:
            audio_int16_list = await decode_chunks_with_resample(audio_files, sampling_rate)
            decode_time = time.time() - decode_start_time
            logger.info(f"并行解码完成: {len(audio_int16_list)} 个文件, 耗时: {decode_time:.3f}s")
        except Exception as e:
            logger.error(f"批量音频解码失败: {e}")
            raise HTTPException(status_code=400, detail=f"批量音频解码失败: {str(e)}")
        
        # 步骤2: 批量进行 VAD 推理
        infer_start_time = time.time()
        for idx, (audio_int16, filename) in enumerate(zip(audio_int16_list, file_names)):
            file_start_time = time.time()
            try:
                # 计算音频时长
                audio_duration = len(audio_int16) / sampling_rate
                
                # VAD 推理
                segments = vad_model.infer_buffer(
                    audio_int16,
                    sampling_rate=sampling_rate,
                    wav_format="pcm"
                )
                
                file_processing_time = time.time() - file_start_time
                
                logger.info(f"[{idx+1}/{len(audio_files)}] {filename}: "
                          f"检测到 {len(segments)} 个语音段, "
                          f"时长: {audio_duration:.3f}s, "
                          f"处理耗时: {file_processing_time:.3f}s")
                
                results.append({
                    "code": 0,
                    "message": "success",
                    "filename": filename,
                    "segments": segments,
                    "total_segments": len(segments),
                    "audio_duration": audio_duration,
                    "processing_time": file_processing_time
                })
                
            except Exception as e:
                # 如果某个文件处理失败，记录错误但继续处理其他文件
                logger.error(f"处理文件 {filename} 失败: {e}")
                results.append({
                    "code": 1,
                    "message": f"处理文件 {filename} 失败: {str(e)}",
                    "filename": filename,
                    "segments": [],
                    "total_segments": 0,
                    "audio_duration": 0.0,
                    "processing_time": 0.0
                })
        
        infer_time = time.time() - infer_start_time
        total_time = time.time() - batch_start_time
        
        successful_count = sum(1 for r in results if r["code"] == 0)
        failed_count = len(results) - successful_count
        
        logger.info(f"批量 VAD 检测完成: "
                  f"成功 {successful_count}/{len(audio_files)}, "
                  f"失败 {failed_count}, "
                  f"总耗时: {total_time:.3f}s "
                  f"(解码: {decode_time:.3f}s, 推理: {infer_time:.3f}s)")
        
        return {
            "code": 0,
            "message": "batch processing completed",
            "total_files": len(audio_files),
            "successful_files": successful_count,
            "failed_files": failed_count,
            "total_processing_time": total_time,
            "decode_time": decode_time,
            "infer_time": infer_time,
            "results": results
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"批量处理请求时发生错误: {e}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")


# 解析命令行参数
parser = argparse.ArgumentParser(description="FunASR Offline VAD FastAPI 服务")
parser.add_argument(
    "--model_dir",
    type=str,
    required=True,
    help="VAD 模型目录路径"
)
parser.add_argument(
    "--host",
    type=str,
    default="0.0.0.0",
    help="服务监听地址 (默认: 0.0.0.0)"
)
parser.add_argument(
    "--port",
    type=int,
    default=8000,
    help="服务端口 (默认: 8000)"
)
parser.add_argument(
    "--thread_num",
    type=int,
    default=1,
    help="推理线程数 (默认: 1)"
)
parser.add_argument(
    "--quantize",
    action="store_true",
    help="使用量化模型"
)
parser.add_argument(
    "--temp_dir",
    type=str,
    default=None,
    help="临时文件目录 (默认: 系统临时目录)"
)
parser.add_argument(
    "--workers",
    type=int,
    default=1,
    help="Uvicorn worker 数量 (默认: 1)"
)
parser.add_argument(
    "--log_level",
    type=str,
    default="info",
    choices=["debug", "info", "warning", "error"],
    help="日志级别 (默认: info)"
)

def main():
    """主函数"""
    global args, temp_dir
    
    args = parser.parse_args()
    
    # 设置临时目录
    if args.temp_dir:
        temp_dir = args.temp_dir
        os.makedirs(temp_dir, exist_ok=True)
    else:
        temp_dir = tempfile.mkdtemp(prefix="funasr_vad_")
        logger.info(f"使用临时目录: {temp_dir}")
    
    logger.info("=" * 60)
    logger.info("FunASR Offline VAD FastAPI 服务")
    logger.info("=" * 60)
    logger.info(f"模型目录: {args.model_dir}")
    logger.info(f"监听地址: {args.host}:{args.port}")
    logger.info(f"线程数: {args.thread_num}")
    logger.info(f"临时目录: {temp_dir}")
    logger.info("=" * 60)
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level=args.log_level
    )


if __name__ == "__main__":
    main()
    logger.info("=" * 60)
    logger.info("FunASR Offline VAD FastAPI 服务")
    logger.info("=" * 60)
    logger.info(f"模型目录: {args.model_dir}")
    logger.info(f"监听地址: {args.host}:{args.port}")
    logger.info(f"线程数: {args.thread_num}")
    logger.info(f"临时目录: {temp_dir}")
    logger.info("=" * 60)
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level=args.log_level
    )


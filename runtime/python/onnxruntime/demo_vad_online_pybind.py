#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
FunASR Online VAD Python Binding 使用示例

这个示例展示了如何使用 pybind11 封装的 FunASR Online VAD C API
"""

import numpy as np
import sys
import os
import time
from collections import defaultdict
from pathlib import Path
import librosa  # 提前导入，避免冷启动影响性能测试
import av
from av.audio.fifo import AudioFifo
from av.audio.resampler import AudioResampler
from av.audio.format import AudioFormat
import io
# 导入 pybind11 封装的模块
try:
    import funasr_vad_online
except ImportError:
    print("错误: 无法导入 funasr_vad_online 模块")
    print("请确保已编译并安装 pybind11 扩展模块")
    sys.exit(1)

def decode_fifo(data: bytes, sampling_rate: int) -> np.ndarray:
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
    # 将输出帧转换为 NumPy
    pcm2d = output_frame.to_ndarray()
    pcm1d = pcm2d.reshape(-1)
    return pcm1d.astype(np.float32)

def load_audio(file_path):
    with open(file_path, "rb") as f:
        file = f.read()
    sr = 16000
    y = decode_fifo(file, sr)
    return y

class PerformanceTimer:
    """性能计时器类"""
    def __init__(self):
        self.timings = defaultdict(list)
        self.start_times = {}
    
    def start(self, name):
        """开始计时"""
        self.start_times[name] = time.perf_counter()
    
    def end(self, name):
        """结束计时并记录"""
        if name in self.start_times:
            elapsed = time.perf_counter() - self.start_times[name]
            self.timings[name].append(elapsed)
            del self.start_times[name]
            return elapsed
        return 0.0
    
    def get_total(self, name):
        """获取某个操作的总耗时"""
        return sum(self.timings.get(name, []))
    
    def get_avg(self, name):
        """获取某个操作的平均耗时"""
        timings = self.timings.get(name, [])
        return sum(timings) / len(timings) if timings else 0.0
    
    def get_count(self, name):
        """获取某个操作的执行次数"""
        return len(self.timings.get(name, []))
    
    def print_summary(self, audio_duration=None):
        """打印性能摘要"""
        print("\n" + "=" * 60)
        print("性能统计")
        print("=" * 60)
        
        total_init_time = self.get_total("init")
        total_infer_time = self.get_total("infer")
        total_load_time = self.get_total("load_audio")
        
        print(f"\n【初始化】")
        if total_init_time > 0:
            print(f"  总耗时: {total_init_time*1000:.2f} ms")
            print(f"  执行次数: {self.get_count('init')}")
        
        print(f"\n【音频加载】")
        if total_load_time > 0:
            print(f"  总耗时: {total_load_time*1000:.2f} ms")
            print(f"  平均耗时: {self.get_avg('load_audio')*1000:.2f} ms")
            print(f"  执行次数: {self.get_count('load_audio')}")
        
        print(f"\n【推理】")
        if total_infer_time > 0:
            infer_count = self.get_count('infer')
            avg_infer_time = self.get_avg('infer')
            print(f"  总耗时: {total_infer_time*1000:.2f} ms ({total_infer_time:.3f} s)")
            print(f"  平均耗时: {avg_infer_time*1000:.2f} ms")
            print(f"  执行次数: {infer_count}")
            if infer_count > 1:
                print(f"  最快: {min(self.timings.get('infer', []))*1000:.2f} ms")
                print(f"  最慢: {max(self.timings.get('infer', []))*1000:.2f} ms")
            
            if audio_duration and audio_duration > 0:
                # RTF 基于平均推理时间计算
                avg_rtf = avg_infer_time / audio_duration
                print(f"  音频时长: {audio_duration:.3f} s")
                print(f"  RTF (Real-Time Factor, 基于平均): {avg_rtf:.4f}")
                if avg_rtf < 1.0:
                    print(f"  ✅ 实时性能: 可以处理 {1.0/avg_rtf:.2f}x 实时速度")
                else:
                    print(f"  ⚠️  处理速度: {avg_rtf:.2f}x 实时速度")
        
        # 显示分块推理的统计
        chunk_times = self.timings.get("infer_chunk", [])
        if chunk_times:
            print(f"\n【分块推理统计】")
            print(f"  总块数: {len(chunk_times)}")
            print(f"  总耗时: {sum(chunk_times)*1000:.2f} ms")
            print(f"  平均每块: {np.mean(chunk_times)*1000:.2f} ms")
            print(f"  最快: {np.min(chunk_times)*1000:.2f} ms")
            print(f"  最慢: {np.max(chunk_times)*1000:.2f} ms")
        
        print("=" * 60)


def python_infer_file(model_dir, wav_path, sampling_rate=16000, repeat=1):

    print("=" * 60)
    print("python-onnxruntime: 从音频文件推理")
    print("=" * 60)
    if repeat > 1:
        print(f"重复推理次数: {repeat}")
    timer = PerformanceTimer()
    from funasr_onnx import Fsmn_vad
    timer.start("init")
    model = Fsmn_vad(model_dir, device_id=-1, speech_noise_thres=0.85)
    timer.end("init")
    
    # 重复推理
    for i in range(repeat):
        timer.start("infer")
        result = model(wav_path)
        timer.end("infer")
    
    timer.print_summary()

def python_infer_bytes(model_dir, wav_path, sampling_rate=16000, repeat=1):
    print("=" * 60)
    print("python-onnxruntime: 从音频字节推理")
    print("=" * 60)
    if repeat > 1:
        print(f"重复推理次数: {repeat}")
    timer = PerformanceTimer()
    from funasr_onnx import Fsmn_vad
    timer.start("init")
    model = Fsmn_vad(model_dir, device_id=-1, speech_noise_thres=0.85)
    timer.end("init")
    timer.start("load_audio")
    data = load_audio(wav_path)
    timer.end("load_audio")
    audio_duration = len(data) / sampling_rate

    print(f"\n音频文件: {wav_path}")
    print(f"音频数据形状: {data.shape}, dtype: {data.dtype}")
    print(f"音频时长: {audio_duration:.3f} s")
    # 重复推理
    for i in range(repeat):
        timer.start("infer")
        result = model(data)
        timer.end("infer")
    print(result)
    timer.print_summary(audio_duration)

def example_infer_file(model_dir, wav_path, sampling_rate=16000, repeat=1):
    """
    示例 1: 从文件推理
    
    Args:
        model_dir: 模型目录路径
        wav_path: 音频文件路径
        sampling_rate: 采样率，默认16000
        repeat: 重复推理次数，默认1
    """
    print("=" * 60)
    print("示例 1: 从音频文件推理")
    print("=" * 60)
    if repeat > 1:
        print(f"重复推理次数: {repeat}")
    
    timer = PerformanceTimer()
    
    # 初始化模型
    model_path = {
        "model-dir": model_dir,
        "quantize": "false"  # 或 "true" 使用量化模型
    }
    
    try:
        timer.start("init")
        vad = funasr_vad_online.FsmnVadOnline(model_path, thread_num=1)
        init_time = timer.end("init")
        print(f"模型初始化成功: {model_dir}")
        print(f"初始化耗时: {init_time*1000:.2f} ms")
        
        # 计算音频时长（从文件）
        timer.start("load_audio")
        waveform, sr = librosa.load(wav_path, sr=sampling_rate)
        audio_duration = len(waveform) / sr
        timer.end("load_audio")
        
        # 重复推理
        segments = None
        for i in range(repeat):
            timer.start("infer")
            segments = vad.infer_file(wav_path, sampling_rate=sampling_rate)
            timer.end("infer")

        
        # print(f"\n音频文件: {wav_path}")
        # print(f"音频时长: {audio_duration:.3f} s")
        # if segments:
        #     print(f"检测到的语音段数量: {len(segments)}")
        #     for i, seg in enumerate(segments[:5]):  # 只显示前5个
        #         if len(seg) >= 2:
        #             start_ms = seg[0]  # 毫秒
        #             end_ms = seg[1]    # 毫秒
        #             print(f"  段 {i+1}: [{start_ms}ms, {end_ms}ms] (时长: {end_ms - start_ms}ms)")
        #     if len(segments) > 5:
        #         print(f"  ... 还有 {len(segments) - 5} 个段")
        
        # 打印本示例的性能统计
        timer.print_summary(audio_duration)
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


def example_infer_buffer_streaming(model_dir, wav_path, sampling_rate=16000, chunk_size_ms=50, repeat=1):
    """
    示例 2: 流式推理（分块处理）
    
    Args:
        model_dir: 模型目录路径
        wav_path: 音频文件路径
        sampling_rate: 采样率，默认16000
        chunk_size_ms: 分块大小（毫秒），默认50
        repeat: 重复推理次数，默认1
    """
    print("\n" + "=" * 60)
    print("示例 2: 流式推理（分块处理）")
    print("=" * 60)
    if repeat > 1:
        print(f"重复推理次数: {repeat}")
    
    timer = PerformanceTimer()
    
    # 初始化模型
    model_path = {
        "model-dir": model_dir,
        "quantize": "false"
    }
    
    try:
        timer.start("init")
        vad = funasr_vad_online.FsmnVadOnline(model_path, thread_num=1)
        init_time = timer.end("init")
        print(f"模型初始化成功: {model_dir}")
        print(f"初始化耗时: {init_time*1000:.2f} ms")
        
        # 加载音频
        timer.start("load_audio")
        waveform, sr = librosa.load(wav_path, sr=sampling_rate)
        waveform_int16 = (waveform * 32767).astype(np.int16)
        audio_duration = len(waveform_int16) / sr
        timer.end("load_audio")
        
        print(f"\n音频文件: {wav_path}")
        print(f"采样率: {sr} Hz")
        print(f"总时长: {audio_duration:.3f} 秒")
        print(f"分块大小: {chunk_size_ms} ms")
        
        # 重复推理
        all_segments = []
        for repeat_idx in range(repeat):
            if repeat > 1:
                print(f"\n--- 第 {repeat_idx + 1}/{repeat} 次完整推理 ---")
            
            # 分块处理
            chunk_samples = int(sampling_rate * chunk_size_ms / 1000)
            repeat_segments = []
            
            timer.start("infer")
            for i in range(0, len(waveform_int16), chunk_samples):
                chunk = waveform_int16[i:i + chunk_samples]
                is_final = (i + chunk_samples >= len(waveform_int16))
                
                # 推理
                chunk_start = time.perf_counter()
                segments = vad.infer_buffer(
                    chunk,
                    is_final=is_final,
                    sampling_rate=sampling_rate,
                    wav_format="pcm"
                )
                chunk_time = time.perf_counter() - chunk_start
                timer.timings["infer_chunk"].append(chunk_time)
                
                if segments:
                    # 调整时间戳（加上当前块的偏移）
                    offset_ms = int(i / sampling_rate * 1000)
                    for seg in segments:
                        if len(seg) >= 2:
                            adjusted_seg = [seg[0] + offset_ms, seg[1] + offset_ms]
                            repeat_segments.append(adjusted_seg)
                
                if (i // chunk_samples) % 10 == 0 and repeat_idx == 0:
                    print(f"  处理进度: {i / len(waveform_int16) * 100:.1f}%")
            
            infer_time = timer.end("infer")
            all_segments = repeat_segments  # 保存最后一次的结果
        
        print(f"\n检测到的语音段数量: {len(all_segments)}")
        for i, seg in enumerate(all_segments[:10]):  # 只显示前10个
            print(f"  段 {i+1}: [{seg[0]}ms, {seg[1]}ms] (时长: {seg[1] - seg[0]}ms)")
        if len(all_segments) > 10:
            print(f"  ... 还有 {len(all_segments) - 10} 个段")
        
        # 打印本示例的性能统计
        timer.print_summary(audio_duration)
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


def example_infer_buffer_numpy(model_dir, wav_path, sampling_rate=16000, repeat=1):
    """
    示例 3: 使用 numpy 数组推理（对应 python_infer_bytes 的 C++ 实现）
    
    Args:
        model_dir: 模型目录路径
        wav_path: 音频文件路径
        sampling_rate: 采样率，默认16000
        repeat: 重复推理次数，默认1
    """
    print("\n" + "=" * 60)
    print("示例 3: 使用 numpy 数组推理 (C++ pybind11)")
    print("=" * 60)
    if repeat > 1:
        print(f"重复推理次数: {repeat}")
    
    timer = PerformanceTimer()
    
    # 初始化模型
    model_path = {
        "model-dir": model_dir,
        "quantize": "false"
    }
    
    try:
        timer.start("init")
        vad = funasr_vad_online.FsmnVadOnline(model_path, thread_num=1)
        init_time = timer.end("init")
        print(f"模型初始化成功: {model_dir}")
        print(f"初始化耗时: {init_time*1000:.2f} ms")
        
        # # 加载音频并转换为 int16 numpy 数组
        # timer.start("load_audio")
        # waveform, sr = librosa.load(wav_path, sr=sampling_rate)
        # waveform_int16 = (waveform * 32767).astype(np.int16)
        # audio_duration = len(waveform_int16) / sr
        # timer.end("load_audio")
        
        # print(f"\n音频文件: {wav_path}")
        # print(f"音频数据形状: {waveform_int16.shape}, dtype: {waveform_int16.dtype}")
        # print(f"音频时长: {audio_duration:.3f} s")

        audio = load_audio(wav_path)
        audio_duration = len(audio) / sampling_rate

        # 转换为 int16
        if audio.dtype == np.float32 or audio.dtype == np.float64:
            audio = (audio * 32768).astype(np.int16)
        else:
            audio = audio.astype(np.int16)

        print(f"\n音频文件: {wav_path}")
        print(f"音频数据形状: {audio.shape}, dtype: {audio.dtype}")
        print(f"音频时长: {audio_duration:.3f} s")

        # 重复推理
        segments = None
        for i in range(repeat):
            timer.start("infer")
            segments = vad.infer_buffer(
                audio,  # 直接传入 numpy 数组
                is_final=True,
                sampling_rate=sampling_rate,
                wav_format="pcm"
            )
            timer.end("infer")
        print(segments)
        # 打印本示例的性能统计
        timer.print_summary(audio_duration)
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


def example_infer_buffer_bytes(model_dir, wav_path, sampling_rate=16000, repeat=1):
    """
    示例 4: 使用 bytes 类型推理
    
    Args:
        model_dir: 模型目录路径
        wav_path: 音频文件路径
        sampling_rate: 采样率，默认16000
        repeat: 重复推理次数，默认1
    """
    print("\n" + "=" * 60)
    print("示例 4: 使用 bytes 类型推理")
    print("=" * 60)
    if repeat > 1:
        print(f"重复推理次数: {repeat}")
    
    timer = PerformanceTimer()
    
    # 初始化模型
    model_path = {
        "model-dir": model_dir,
        "quantize": "false"
    }
    
    try:
        timer.start("init")
        vad = funasr_vad_online.FsmnVadOnline(model_path, thread_num=1)
        init_time = timer.end("init")
        print(f"模型初始化成功: {model_dir}")
        print(f"初始化耗时: {init_time*1000:.2f} ms")
        
        # 加载音频并转换为 bytes
        timer.start("load_audio")
        waveform, sr = librosa.load(wav_path, sr=sampling_rate)
        waveform_int16 = (waveform * 32767).astype(np.int16)
        audio_bytes = waveform_int16.tobytes()
        audio_duration = len(waveform_int16) / sr
        timer.end("load_audio")
        
        print(f"\n音频文件: {wav_path}")
        print(f"音频数据大小: {len(audio_bytes)} 字节")
        print(f"音频时长: {audio_duration:.3f} s")
        
        # 重复推理
        segments = None
        for i in range(repeat):
            timer.start("infer")
            segments = vad.infer_buffer_bytes(
                audio_bytes,
                is_final=True,
                sampling_rate=sampling_rate,
                wav_format="pcm"
            )
            timer.end("infer")

        
        # if segments:
        #     print(f"检测到的语音段数量: {len(segments)}")
        #     for i, seg in enumerate(segments[:5]):  # 只显示前5个
        #         if len(seg) >= 2:
        #             print(f"  段 {i+1}: [{seg[0]}ms, {seg[1]}ms] (时长: {seg[1] - seg[0]}ms)")
        #     if len(segments) > 5:
        #         print(f"  ... 还有 {len(segments) - 5} 个段")
        
        # 打印本示例的性能统计
        timer.print_summary(audio_duration)
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


# 用于存储Python模型的全局字典（每个example一个模型实例）
_python_models_cache = {}

def process_folder_with_example(example_name, example_func, model_dir, folder_path, sampling_rate=16000, **kwargs):
    """
    通用的文件夹处理函数，支持不同的example函数
    
    Args:
        example_name: example的名称（用于显示）
        example_func: example函数，接受 (model_dir, wav_path, sampling_rate, repeat=1) 参数
        model_dir: 模型目录路径
        folder_path: 包含wav文件的文件夹路径
        sampling_rate: 采样率
        **kwargs: 传递给example_func的其他参数（如chunk_size_ms等）
    """
    global _python_models_cache
    # 获取所有wav文件
    wav_files = []
    for ext in ['*.wav', '*.WAV']:
        wav_files.extend(Path(folder_path).glob(ext))
    
    if not wav_files:
        print(f"错误: 文件夹中没有找到wav文件: {folder_path}")
        return None
    
    wav_files = sorted(wav_files)
    print(f"\n{'='*60}")
    print(f"{example_name} - 文件夹模式")
    print(f"{'='*60}")
    print(f"找到 {len(wav_files)} 个wav文件")
    
    timer = PerformanceTimer()
    
    # 初始化模型（只初始化一次）
    model_path = {
        "model-dir": model_dir,
        "quantize": "false"
    }
    
    try:
        timer.start("init")
        vad = funasr_vad_online.FsmnVadOnline(model_path, thread_num=1)
        init_time = timer.end("init")
        print(f"✓ 模型初始化成功: {model_dir}")
        print(f"初始化耗时: {init_time*1000:.2f} ms\n")
    except Exception as e:
        print(f"✗ 模型初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 处理每个文件
    total_audio_duration = 0.0
    successful_files = 0
    failed_files = 0
    
    print("开始处理文件...")
    print("-" * 60)
    
    for idx, wav_file in enumerate(wav_files, 1):
        wav_path = str(wav_file)
        print(f"\n[{idx}/{len(wav_files)}] 处理: {wav_file.name}")
        
        try:
            # 根据不同的example函数调用不同的处理方式
            if example_func == example_infer_file:
                # 从文件推理
                timer.start("load_audio")
                waveform, sr = librosa.load(wav_path, sr=sampling_rate)
                audio_duration = len(waveform) / sr
                timer.end("load_audio")
                
                total_audio_duration += audio_duration
                
                timer.start("infer")
                segments = vad.infer_file(wav_path, sampling_rate=sampling_rate)
                infer_time = timer.end("infer")
                
            elif example_func == example_infer_buffer_numpy:
                # 从numpy数组推理
                timer.start("load_audio")
                audio = load_audio(wav_path)
                if audio.dtype == np.float32 or audio.dtype == np.float64:
                    audio_int16 = (audio * 32768).astype(np.int16)
                else:
                    audio_int16 = audio.astype(np.int16)
                audio_duration = len(audio_int16) / sampling_rate
                timer.end("load_audio")
                
                total_audio_duration += audio_duration
                
                timer.start("infer")
                segments = vad.infer_buffer(
                    audio_int16,
                    is_final=True,
                    sampling_rate=sampling_rate,
                    wav_format="pcm"
                )
                infer_time = timer.end("infer")
                
            elif example_func == example_infer_buffer_streaming:
                # 流式推理（分块处理）- 这个比较复杂，需要特殊处理
                chunk_size_ms = kwargs.get('chunk_size_ms', 50)
                timer.start("load_audio")
                waveform, sr = librosa.load(wav_path, sr=sampling_rate)
                waveform_int16 = (waveform * 32767).astype(np.int16)
                audio_duration = len(waveform_int16) / sr
                timer.end("load_audio")
                
                total_audio_duration += audio_duration
                
                # 分块处理
                timer.start("infer")
                chunk_samples = int(sampling_rate * chunk_size_ms / 1000)
                all_segments = []
                for i in range(0, len(waveform_int16), chunk_samples):
                    chunk = waveform_int16[i:i + chunk_samples]
                    is_final = (i + chunk_samples >= len(waveform_int16))
                    segments = vad.infer_buffer(
                        chunk,
                        is_final=is_final,
                        sampling_rate=sampling_rate,
                        wav_format="pcm"
                    )
                    if segments:
                        offset_ms = int(i / sampling_rate * 1000)
                        for seg in segments:
                            if len(seg) >= 2:
                                adjusted_seg = [seg[0] + offset_ms, seg[1] + offset_ms]
                                all_segments.append(adjusted_seg)
                infer_time = timer.end("infer")
                segments = all_segments
                
            elif example_func == example_infer_buffer_bytes:
                # 从bytes推理
                timer.start("load_audio")
                waveform, sr = librosa.load(wav_path, sr=sampling_rate)
                waveform_int16 = (waveform * 32767).astype(np.int16)
                audio_bytes = waveform_int16.tobytes()
                audio_duration = len(waveform_int16) / sr
                timer.end("load_audio")
                
                total_audio_duration += audio_duration
                
                timer.start("infer")
                segments = vad.infer_buffer_bytes(
                    audio_bytes,
                    is_final=True,
                    sampling_rate=sampling_rate,
                    wav_format="pcm"
                )
                infer_time = timer.end("infer")
                
            elif example_func == python_infer_file:
                # Python实现的文件推理
                from funasr_onnx import Fsmn_vad
                # 只在第一次初始化模型（每个example函数一个模型实例）
                cache_key = f"{example_func.__name__}_{model_dir}"
                if cache_key not in _python_models_cache:
                    _python_models_cache[cache_key] = Fsmn_vad(model_dir, device_id=-1, speech_noise_thres=0.85)
                model = _python_models_cache[cache_key]
                
                timer.start("load_audio")
                waveform, sr = librosa.load(wav_path, sr=sampling_rate)
                audio_duration = len(waveform) / sr
                timer.end("load_audio")
                
                total_audio_duration += audio_duration
                
                timer.start("infer")
                result = model(wav_path)
                infer_time = timer.end("infer")
                segments = result
                
            elif example_func == python_infer_bytes:
                # Python实现的bytes推理
                from funasr_onnx import Fsmn_vad
                # 只在第一次初始化模型（每个example函数一个模型实例）
                cache_key = f"{example_func.__name__}_{model_dir}"
                if cache_key not in _python_models_cache:
                    _python_models_cache[cache_key] = Fsmn_vad(model_dir, device_id=-1, speech_noise_thres=0.85)
                model = _python_models_cache[cache_key]
                
                timer.start("load_audio")
                data = load_audio(wav_path)
                audio_duration = len(data) / sampling_rate
                timer.end("load_audio")
                
                total_audio_duration += audio_duration
                
                timer.start("infer")
                result = model(data)
                infer_time = timer.end("infer")
                segments = result
                
            else:
                raise ValueError(f"不支持的example函数: {example_func}")
            
            successful_files += 1
            seg_count = len(segments) if isinstance(segments, (list, tuple)) else 0
            print(f"  ✓ 成功 - 时长: {audio_duration:.3f}s, 推理耗时: {infer_time*1000:.2f}ms, 检测到 {seg_count} 个语音段")
            
        except Exception as e:
            failed_files += 1
            print(f"  ✗ 失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 收集统计信息
    total_infer_time = timer.get_total("infer")
    total_load_time = timer.get_total("load_audio")
    avg_infer_time = timer.get_avg("infer")
    init_time = timer.get_total("init")
    total_time = init_time + total_load_time + total_infer_time
    
    stats = {
        "example_name": example_name,
        "total_files": len(wav_files),
        "successful_files": successful_files,
        "failed_files": failed_files,
        "total_audio_duration": total_audio_duration,
        "init_time": init_time,
        "total_load_time": total_load_time,
        "total_infer_time": total_infer_time,
        "total_time": total_time,
        "avg_infer_time": avg_infer_time,
        "avg_audio_duration": total_audio_duration / successful_files if successful_files > 0 else 0,
        "avg_rtf": avg_infer_time / (total_audio_duration / successful_files) if successful_files > 0 and total_audio_duration > 0 else 0
    }
    
    return stats


def print_all_statistics(all_stats):
    """
    统一打印所有example的统计信息
    
    Args:
        all_stats: 统计信息列表，每个元素是一个字典
    """
    if not all_stats:
        return
    
    print("\n" + "=" * 80)
    print("所有 Example 统计信息汇总")
    print("=" * 80)
    
    # 打印表格头部
    print(f"\n{'Example名称':<40} {'文件数':<8} {'成功':<8} {'失败':<8} {'总时长(s)':<12} {'总耗时(ms)':<12} {'平均RTF':<10}")
    print("-" * 80)
    
    # 打印每个example的统计
    for stats in all_stats:
        if stats is None:
            continue
        example_name = stats["example_name"]
        total_files = stats["total_files"]
        successful = stats["successful_files"]
        failed = stats["failed_files"]
        total_audio = stats["total_audio_duration"]
        total_time_ms = stats["total_time"] * 1000
        avg_rtf = stats["avg_rtf"]
        
        print(f"{example_name:<40} {total_files:<8} {successful:<8} {failed:<8} {total_audio:<12.3f} {total_time_ms:<12.2f} {avg_rtf:<10.4f}")
    
    print("\n" + "=" * 80)
    print("详细统计信息")
    print("=" * 80)
    
    # 打印每个example的详细信息
    for stats in all_stats:
        if stats is None:
            continue
        
        print(f"\n{'-'*80}")
        print(f"{stats['example_name']} - 详细统计")
        print(f"{'-'*80}")
        print(f"总文件数: {stats['total_files']}")
        print(f"成功处理: {stats['successful_files']}")
        print(f"失败: {stats['failed_files']}")
        print(f"总音频时长: {stats['total_audio_duration']:.3f} s")
        
        print(f"\n【总耗时】")
        print(f"  初始化: {stats['init_time']*1000:.2f} ms")
        print(f"  音频加载: {stats['total_load_time']*1000:.2f} ms")
        print(f"  推理: {stats['total_infer_time']*1000:.2f} ms ({stats['total_infer_time']:.3f} s)")
        print(f"  总计: {stats['total_time']*1000:.2f} ms")
        
        if stats['successful_files'] > 0:
            print(f"\n【平均每个文件】")
            print(f"  推理耗时: {stats['avg_infer_time']*1000:.2f} ms")
            print(f"  音频时长: {stats['avg_audio_duration']:.3f} s")
            if stats['avg_rtf'] > 0:
                print(f"  RTF: {stats['avg_rtf']:.4f}")
                if stats['avg_rtf'] < 1.0:
                    print(f"  ✅ 实时性能: 可以处理 {1.0/stats['avg_rtf']:.2f}x 实时速度")
                else:
                    print(f"  ⚠️  处理速度: {stats['avg_rtf']:.2f}x 实时速度")
    
    print("\n" + "=" * 80)


def main():
    if len(sys.argv) < 3:
        print("用法: python demo_vad_online_pybind.py <model_dir> <wav_path_or_folder> [sampling_rate] [repeat]")
        print("\n参数:")
        print("  model_dir:         VAD 模型目录路径")
        print("  wav_path_or_folder: 音频文件路径 或 包含wav文件的文件夹路径")
        print("  sampling_rate:     采样率 (默认: 16000)")
        print("  repeat:            重复推理次数，用于统计平均耗时 (默认: 1)")
        print("                     注意: 当输入是文件夹时，repeat参数无效")
        print("\n示例:")
        print("  单文件模式:")
        print("    python demo_vad_online_pybind.py ./model ./audio.wav 16000 10")
        print("  文件夹模式:")
        print("    python demo_vad_online_pybind.py ./model ./wav_folder 16000")
        sys.exit(1)
    
    model_dir = sys.argv[1]
    wav_path = sys.argv[2]
    sampling_rate = int(sys.argv[3]) if len(sys.argv) > 3 else 16000
    repeat = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    
    if not os.path.exists(model_dir):
        print(f"错误: 模型目录不存在: {model_dir}")
        sys.exit(1)
    
    if not os.path.exists(wav_path):
        print(f"错误: 音频文件或文件夹不存在: {wav_path}")
        sys.exit(1)
    
    # 判断是文件还是文件夹
    if os.path.isdir(wav_path):
        # 文件夹模式 - 运行所有example
        print("FunASR Online VAD Python Binding 示例")
        print("=" * 60)
        print(f"检测到文件夹输入: {wav_path}")
        print("=" * 60)
        
        # 收集所有example的统计信息
        all_stats = []
        
        # 运行所有example的文件夹处理版本
        all_stats.append(process_folder_with_example("示例 1: 从文件推理", example_infer_file, model_dir, wav_path, sampling_rate))
        all_stats.append(process_folder_with_example("示例 2: 流式推理(分块处理)", example_infer_buffer_streaming, model_dir, wav_path, sampling_rate, chunk_size_ms=50))
        all_stats.append(process_folder_with_example("示例 3: 使用numpy数组推理", example_infer_buffer_numpy, model_dir, wav_path, sampling_rate))
        all_stats.append(process_folder_with_example("示例 4: 使用bytes类型推理", example_infer_buffer_bytes, model_dir, wav_path, sampling_rate))
        all_stats.append(process_folder_with_example("Python实现: 从文件推理", python_infer_file, model_dir, wav_path, sampling_rate))
        all_stats.append(process_folder_with_example("Python实现: 从bytes推理", python_infer_bytes, model_dir, wav_path, sampling_rate))
        
        # 统一打印所有统计信息
        print_all_statistics(all_stats)
    else:
        # 单文件模式（原有逻辑）
        if repeat < 1:
            print(f"错误: 重复次数必须 >= 1，当前值: {repeat}")
            sys.exit(1)
        
        print("FunASR Online VAD Python Binding 示例")
        print("=" * 60)
        print(f"单文件模式: {wav_path}")
        print("=" * 60)
        
        # 运行示例（每个示例使用独立的计时器）
        #example_infer_file(model_dir, wav_path, sampling_rate, repeat=repeat)
        #example_infer_buffer_streaming(model_dir, wav_path, sampling_rate, repeat=repeat)
        example_infer_buffer_numpy(model_dir, wav_path, sampling_rate, repeat=repeat)  # 使用 numpy 数组
        #example_infer_buffer_bytes(model_dir, wav_path, sampling_rate, repeat=repeat)  # 使用 bytes
        python_infer_bytes(model_dir, wav_path, sampling_rate, repeat=repeat)  # Python 实现的对比


if __name__ == "__main__":
    main()


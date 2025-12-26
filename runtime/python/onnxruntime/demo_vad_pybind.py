#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
FunASR Offline VAD Python Binding 使用示例

这个示例展示了如何使用 pybind11 封装的 FunASR 离线 VAD 模块。
"""

import numpy as np
import soundfile as sf
import sys
import os
import time
from collections import defaultdict
from pathlib import Path
import av
from av.audio.fifo import AudioFifo
from av.audio.resampler import AudioResampler
from av.audio.format import AudioFormat
import io
try:
    import funasr_vad
except ImportError:
    print("错误: 无法导入 funasr_vad 模块")
    print("请确保已经编译了 Python 扩展模块:")
    print("  python setup_pybind11.py build_ext --inplace")
    exit(1)


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
        
        print("=" * 60)

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

def python_infer_numpy(model_dir, wav_path, sampling_rate=16000, repeat=1):
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
        "quantize": "false"
    }
    
    try:
        timer.start("init")
        vad = funasr_vad.FsmnVad(model_path, thread_num=1)
        init_time = timer.end("init")
        print(f"✓ 模型初始化成功: {model_dir}")
        print(f"初始化耗时: {init_time*1000:.2f} ms")
    except Exception as e:
        print(f"✗ 模型初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    if not os.path.exists(wav_path):
        print(f"✗ 音频文件不存在: {wav_path}")
        return
    
    try:
        # 计算音频时长
        timer.start("load_audio")
        audio, sr = sf.read(wav_path)
        audio_duration = len(audio) / sr
        timer.end("load_audio")
        
        print(f"\n音频文件: {wav_path}")
        print(f"音频时长: {audio_duration:.3f} s")
        
        # 重复推理
        segments = None
        for i in range(repeat):
            timer.start("infer")
            segments = vad.infer_file(wav_path, sampling_rate=sampling_rate)
            timer.end("infer")
        
        # if segments:
        #     print(f"\n✓ 推理成功，检测到 {len(segments)} 个语音段:")
        #     for i, seg in enumerate(segments[:10]):  # 只显示前10个
        #         if len(seg) >= 2:
        #             print(f"  段 {i+1}: [{seg[0]}ms, {seg[1]}ms] (时长: {seg[1] - seg[0]}ms)")
        #     if len(segments) > 10:
        #         print(f"  ... 还有 {len(segments) - 10} 个段")
        
        # 打印性能统计
        timer.print_summary(audio_duration)
        
    except Exception as e:
        print(f"✗ 推理失败: {e}")
        import traceback
        traceback.print_exc()


def example_infer_buffer(model_dir, wav_path, sampling_rate=16000, repeat=1):
    """
    示例 2: 从 numpy 数组推理
    
    Args:
        model_dir: 模型目录路径
        wav_path: 音频文件路径
        sampling_rate: 采样率，默认16000
        repeat: 重复推理次数，默认1
    """
    print("\n" + "=" * 60)
    print("示例 2: 从 numpy 数组推理")
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
        vad = funasr_vad.FsmnVad(model_path, thread_num=1)
        init_time = timer.end("init")
        print(f"✓ 模型初始化成功: {model_dir}")
        print(f"初始化耗时: {init_time*1000:.2f} ms")
    except Exception as e:
        print(f"✗ 模型初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    if not os.path.exists(wav_path):
        print(f"✗ 音频文件不存在: {wav_path}")
        return
    
    try:
        # 读取音频文件
        timer.start("load_audio")
        audio, sampling_rate = sf.read(wav_path)
        
        # 转换为 int16
        if audio.dtype == np.float32 or audio.dtype == np.float64:
            audio = (audio * 32768).astype(np.int16)
        else:
            audio = audio.astype(np.int16)
        
        audio_duration = len(audio) / sampling_rate
        timer.end("load_audio")
        

        
        print(f"\n音频文件: {wav_path}")
        print(f"✓ 音频加载成功: {len(audio)} 个采样点, 采样率: {sampling_rate} Hz")
        print(f"音频时长: {audio_duration:.3f} s")
        print(f"音频数据形状: {audio.shape}, dtype: {audio.dtype}")
        
        # 重复推理
        segments = None
        for i in range(repeat):
            timer.start("infer")
            segments = vad.infer_buffer(audio, sampling_rate=sampling_rate, wav_format="pcm")
            timer.end("infer")
        print(segments)
        # if segments:
        #     print(f"\n✓ 推理成功，检测到 {len(segments)} 个语音段:")
        #     for i, seg in enumerate(segments[:10]):  # 只显示前10个
        #         if len(seg) >= 2:
        #             print(f"  段 {i+1}: [{seg[0]}ms, {seg[1]}ms] (时长: {seg[1] - seg[0]}ms)")
        #     if len(segments) > 10:
        #         print(f"  ... 还有 {len(segments) - 10} 个段")
        
        # 打印性能统计
        timer.print_summary(audio_duration)
        
    except Exception as e:
        print(f"✗ 推理失败: {e}")
        import traceback
        traceback.print_exc()


def example_infer_buffer_newsample(model_dir, wav_path, sampling_rate=16000, repeat=1):
    """
    示例 2: 从 numpy 数组推理-newsample
    
    Args:
        model_dir: 模型目录路径
        wav_path: 音频文件路径
        sampling_rate: 采样率，默认16000
        repeat: 重复推理次数，默认1
    """
    print("\n" + "=" * 60)
    print("示例 2: 从 numpy 数组推理")
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
        vad = funasr_vad.FsmnVad(model_path, thread_num=1)
        init_time = timer.end("init")
        print(f"✓ 模型初始化成功: {model_dir}")
        print(f"初始化耗时: {init_time*1000:.2f} ms")
    except Exception as e:
        print(f"✗ 模型初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    if not os.path.exists(wav_path):
        print(f"✗ 音频文件不存在: {wav_path}")
        return
    
    try:
        # 读取音频文件
        # timer.start("load_audio")
        # audio, sampling_rate = sf.read(wav_path)
        
        # # 转换为 int16
        # if audio.dtype == np.float32 or audio.dtype == np.float64:
        #     audio = (audio * 32768).astype(np.int16)
        # else:
        #     audio = audio.astype(np.int16)
        
        # audio_duration = len(audio) / sample_rate
        # timer.end("load_audio")
        
        timer.start("load_audio")
        audio = load_audio(wav_path)
        
        # 转换为 int16
        if audio.dtype == np.float32 or audio.dtype == np.float64:
            audio = (audio * 32768).astype(np.int16)
        else:
            audio = audio.astype(np.int16)
        
        audio_duration = len(audio) / sampling_rate
        timer.end("load_audio")
        
        print(f"\n音频文件: {wav_path}")
        print(f"✓ 音频加载成功: {len(audio)} 个采样点, 采样率: {sampling_rate} Hz")
        print(f"音频时长: {audio_duration:.3f} s")
        print(f"音频数据形状: {audio.shape}, dtype: {audio.dtype}")
        
        # 重复推理
        segments = None
        for i in range(repeat):
            timer.start("infer")
            segments = vad.infer_buffer(audio, sampling_rate=sampling_rate, wav_format="pcm")
            timer.end("infer")
        print(segments)
        # if segments:
        #     print(f"\n✓ 推理成功，检测到 {len(segments)} 个语音段:")
        #     for i, seg in enumerate(segments[:10]):  # 只显示前10个
        #         if len(seg) >= 2:
        #             print(f"  段 {i+1}: [{seg[0]}ms, {seg[1]}ms] (时长: {seg[1] - seg[0]}ms)")
        #     if len(segments) > 10:
        #         print(f"  ... 还有 {len(segments) - 10} 个段")
        
        # 打印性能统计
        timer.print_summary(audio_duration)
        
    except Exception as e:
        print(f"✗ 推理失败: {e}")
        import traceback
        traceback.print_exc()

def example_infer_buffer_bytes(model_dir, wav_path, sampling_rate=16000, repeat=1):
    """
    示例 3: 从 bytes 数据推理
    
    Args:
        model_dir: 模型目录路径
        wav_path: 音频文件路径
        sampling_rate: 采样率，默认16000
        repeat: 重复推理次数，默认1
    """
    print("\n" + "=" * 60)
    print("示例 3: 从 bytes 数据推理")
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
        vad = funasr_vad.FsmnVad(model_path, thread_num=1)
        init_time = timer.end("init")
        print(f"✓ 模型初始化成功: {model_dir}")
        print(f"初始化耗时: {init_time*1000:.2f} ms")
    except Exception as e:
        print(f"✗ 模型初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    if not os.path.exists(wav_path):
        print(f"✗ 音频文件不存在: {wav_path}")
        return
    
    try:
        # 读取音频文件为 bytes
        timer.start("load_audio")
        with open(wav_path, "rb") as f:
            audio_bytes = f.read()
        
        # 估算音频时长（从文件大小粗略估算，实际应该解析 WAV 头）
        # 这里使用 soundfile 来获取准确的时长
        audio, sr = sf.read(wav_path)
        audio_duration = len(audio) / sr
        timer.end("load_audio")
        
        print(f"\n音频文件: {wav_path}")
        print(f"✓ 音频加载成功: {len(audio_bytes)} 字节")
        print(f"音频时长: {audio_duration:.3f} s")
        
        # 重复推理（注意：这里假设是 WAV 格式）
        segments = None
        for i in range(repeat):
            timer.start("infer")
            segments = vad.infer_buffer_bytes(audio_bytes, sampling_rate=sampling_rate, wav_format="wav")
            timer.end("infer")
        print(segments)
        # if segments:
        #     print(f"\n✓ 推理成功，检测到 {len(segments)} 个语音段:")
        #     for i, seg in enumerate(segments[:10]):  # 只显示前10个
        #         if len(seg) >= 2:
        #             print(f"  段 {i+1}: [{seg[0]}ms, {seg[1]}ms] (时长: {seg[1] - seg[0]}ms)")
        #     if len(segments) > 10:
        #         print(f"  ... 还有 {len(segments) - 10} 个段")
        
        # 打印性能统计
        timer.print_summary(audio_duration)
        
    except Exception as e:
        print(f"✗ 推理失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    if len(sys.argv) < 3:
        print("用法: python demo_vad_pybind.py <model_dir> <wav_path> [sampling_rate] [repeat]")
        print("\n参数:")
        print("  model_dir:      VAD 模型目录路径")
        print("  wav_path:       音频文件路径")
        print("  sampling_rate:  采样率 (默认: 16000)")
        print("  repeat:         重复推理次数，用于统计平均耗时 (默认: 1)")
        print("\n示例:")
        print("  python demo_vad_pybind.py ./model ./audio.wav 16000 10")
        sys.exit(1)
    
    model_dir = sys.argv[1]
    wav_path = sys.argv[2]
    sampling_rate = int(sys.argv[3]) if len(sys.argv) > 3 else 16000
    repeat = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    
    if not os.path.exists(model_dir):
        print(f"错误: 模型目录不存在: {model_dir}")
        sys.exit(1)
    
    if not os.path.exists(wav_path):
        print(f"错误: 音频文件不存在: {wav_path}")
        sys.exit(1)
    
    if repeat < 1:
        print(f"错误: 重复次数必须 >= 1，当前值: {repeat}")
        sys.exit(1)
    
    print("FunASR Offline VAD Python Binding 示例")
    print("=" * 60)
    
    # 运行示例
    #example_infer_file(model_dir, wav_path, sampling_rate, repeat=repeat)
    example_infer_buffer(model_dir, wav_path, sampling_rate, repeat=repeat)
    example_infer_buffer_newsample(model_dir, wav_path, sampling_rate, repeat=repeat)
    example_infer_buffer_bytes(model_dir, wav_path, sampling_rate, repeat=repeat)
    python_infer_numpy(model_dir, wav_path, sampling_rate, repeat=repeat)
    
    print("\n" + "=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()


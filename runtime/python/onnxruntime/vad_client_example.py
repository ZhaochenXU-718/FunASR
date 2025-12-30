#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
FunASR Offline VAD FastAPI 服务客户端示例

演示如何调用 VAD 服务的 API 接口
"""

import requests
import argparse
import json
from pathlib import Path


def test_health_check(base_url: str):
    """测试健康检查接口"""
    print("=" * 60)
    print("测试健康检查接口")
    print("=" * 60)
    
    try:
        response = requests.get(f"{base_url}/health")
        response.raise_for_status()
        result = response.json()
        print(f"状态: {result['status']}")
        print(f"模型已加载: {result['model_loaded']}")
        if result.get('model_dir'):
            print(f"模型目录: {result['model_dir']}")
        return True
    except Exception as e:
        print(f"健康检查失败: {e}")
        return False


def test_vad_detect(base_url: str, audio_path: str, sampling_rate: int = 16000):
    """测试单文件 VAD 检测"""
    print("\n" + "=" * 60)
    print("测试单文件 VAD 检测")
    print("=" * 60)
    print(f"音频文件: {audio_path}")
    
    if not Path(audio_path).exists():
        print(f"错误: 音频文件不存在: {audio_path}")
        return None
    
    try:
        with open(audio_path, "rb") as f:
            files = {"audio": (Path(audio_path).name, f, "audio/wav")}
            data = {
                "sampling_rate": sampling_rate,
                "wav_format": "auto"
            }
            response = requests.post(f"{base_url}/vad/detect", files=files, data=data)
            response.raise_for_status()
            result = response.json()
            
            print(f"\n检测结果:")
            print(f"  状态码: {result['code']}")
            print(f"  消息: {result['message']}")
            print(f"  检测到 {result['total_segments']} 个语音段")
            print(f"  音频时长: {result['audio_duration']:.3f} 秒")
            print(f"  处理耗时: {result['processing_time']:.3f} 秒")
            
            if result['segments']:
                print(f"\n语音段列表:")
                for i, seg in enumerate(result['segments'][:10], 1):
                    if len(seg) >= 2:
                        print(f"  段 {i}: [{seg[0]}ms, {seg[1]}ms] (时长: {seg[1] - seg[0]}ms)")
                if len(result['segments']) > 10:
                    print(f"  ... 还有 {len(result['segments']) - 10} 个段")
            
            return result
            
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
                print(f"错误详情: {error_detail}")
            except:
                print(f"响应内容: {e.response.text}")
        return None
    except Exception as e:
        print(f"处理失败: {e}")
        return None


def test_vad_detect_batch(base_url: str, audio_paths: list, sampling_rate: int = 16000):
    """测试批量 VAD 检测"""
    print("\n" + "=" * 60)
    print("测试批量 VAD 检测")
    print("=" * 60)
    print(f"音频文件数量: {len(audio_paths)}")
    
    try:
        files = []
        for audio_path in audio_paths:
            if not Path(audio_path).exists():
                print(f"警告: 跳过不存在的文件: {audio_path}")
                continue
            files.append(("audio_files", (Path(audio_path).name, open(audio_path, "rb"), "audio/wav")))
        
        if not files:
            print("错误: 没有有效的音频文件")
            return None
        
        data = {"sampling_rate": sampling_rate}
        response = requests.post(f"{base_url}/vad/detect_batch", files=files, data=data)
        response.raise_for_status()
        result = response.json()
        
        # 关闭文件
        for _, (_, file_obj, _) in files:
            file_obj.close()
        
        print(f"\n批量检测结果:")
        print(f"  状态码: {result['code']}")
        print(f"  消息: {result['message']}")
        print(f"  总文件数: {result['total_files']}")
        
        for i, file_result in enumerate(result['results'], 1):
            print(f"\n  文件 {i}:")
            print(f"    状态码: {file_result['code']}")
            print(f"    消息: {file_result['message']}")
            print(f"    检测到 {file_result['total_segments']} 个语音段")
            print(f"    音频时长: {file_result['audio_duration']:.3f} 秒")
            print(f"    处理耗时: {file_result['processing_time']:.3f} 秒")
        
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
                print(f"错误详情: {error_detail}")
            except:
                print(f"响应内容: {e.response.text}")
        return None
    except Exception as e:
        print(f"处理失败: {e}")
        return None

def test_detect_chunks(base_url: str, audio_paths: list, sampling_rate: int = 16000):
    """测试批量 VAD 检测"""
    print("\n" + "=" * 60)
    print("测试批量 VAD 检测")
    print("=" * 60)
    print(f"音频文件数量: {len(audio_paths)}")
    
    try:
        files = []
        for audio_path in audio_paths:
            if not Path(audio_path).exists():
                print(f"警告: 跳过不存在的文件: {audio_path}")
                continue
            files.append(("audio_files", (Path(audio_path).name, open(audio_path, "rb"), "audio/wav")))
        
        if not files:
            print("错误: 没有有效的音频文件")
            return None
        
        data = {"sampling_rate": sampling_rate}
        response = requests.post(f"{base_url}/vad/detect_chunks", files=files, data=data)
        response.raise_for_status()
        result = response.json()
        
        # 关闭文件
        for _, (_, file_obj, _) in files:
            file_obj.close()
        
        print(f'result: {result}')
        # print(f"\n批量检测结果:")
        # print(f"  状态码: {result['code']}")
        # print(f"  消息: {result['message']}")
        # print(f"  总文件数: {result['total_files']}")
        
        # for i, file_result in enumerate(result['results'], 1):
        #     print(f"\n  文件 {i}:")
        #     print(f"    状态码: {file_result['code']}")
        #     print(f"    消息: {file_result['message']}")
        #     print(f"    检测到 {file_result['total_segments']} 个语音段")
        #     print(f"    音频时长: {file_result['audio_duration']:.3f} 秒")
        #     print(f"    处理耗时: {file_result['processing_time']:.3f} 秒")
        
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
                print(f"错误详情: {error_detail}")
            except:
                print(f"响应内容: {e.response.text}")
        return None
    except Exception as e:
        print(f"处理失败: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="FunASR Offline VAD FastAPI 服务客户端")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="服务地址 (默认: http://localhost:8000)"
    )
    parser.add_argument(
        "--audio",
        type=str,
        help="音频文件路径（单文件模式）"
    )
    parser.add_argument(
        "--audio_list",
        type=str,
        help="包含 wav 文件的文件夹路径（批量模式）"
    )
    parser.add_argument(
        "--sampling_rate",
        type=int,
        default=16000,
        help="采样率 (默认: 16000)"
    )
    
    args = parser.parse_args()
    
    base_url = args.url.rstrip('/')
    
    print("FunASR Offline VAD FastAPI 客户端")
    print("=" * 60)
    print(f"服务地址: {base_url}")
    print("=" * 60)
    
    # 健康检查
    if not test_health_check(base_url):
        print("\n服务不可用，请检查服务是否已启动")
        return
    
    # 单文件检测
    if args.audio:
        test_vad_detect(base_url, args.audio, args.sampling_rate)
    
    # 批量检测
    if args.audio_list:
        # 从文件夹中读取所有 wav 文件
        folder_path = Path(args.audio_list)
        if not folder_path.exists():
            print(f"错误: 文件夹不存在: {args.audio_list}")
            return
        
        if not folder_path.is_dir():
            print(f"错误: 路径不是文件夹: {args.audio_list}")
            return
        
        # 获取所有 wav 文件
        wav_files = []
        for ext in ['*.wav', '*.WAV']:
            wav_files.extend(folder_path.glob(ext))
        
        wav_files = sorted(wav_files)
        
        if not wav_files:
            print(f"错误: 文件夹中没有找到 wav 文件: {args.audio_list}")
            return
        
        print(f"\n找到 {len(wav_files)} 个 wav 文件")
        audio_paths = [str(f) for f in wav_files]
        # test_vad_detect_batch(base_url, audio_paths, args.sampling_rate)
        test_detect_chunks(base_url, audio_paths, args.sampling_rate)
    # 如果都没有指定，只做健康检查
    if not args.audio and not args.audio_list:
        print("\n提示: 使用 --audio 或 --audio_list 参数来测试 VAD 检测功能")
        print("示例:")
        print("  python vad_client_example.py --url http://localhost:8000 --audio ./audio.wav")
        print("  python vad_client_example.py --url http://localhost:8000 --audio_list ./wav_folder")


if __name__ == "__main__":
    main()


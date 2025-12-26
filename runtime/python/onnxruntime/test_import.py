#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
快速测试 funasr_vad_online 模块是否可以正常导入
"""

import sys

print("测试 funasr_vad_online 模块导入...")

try:
    import funasr_vad_online
    print("✅ 成功导入 funasr_vad_online 模块")
    
    # 检查模块属性
    print(f"模块路径: {funasr_vad_online.__file__}")
    print(f"模块内容: {dir(funasr_vad_online)}")
    
    # 检查类是否存在
    if hasattr(funasr_vad_online, 'FsmnVadOnline'):
        print("✅ FsmnVadOnline 类存在")
        print(f"   类文档: {funasr_vad_online.FsmnVadOnline.__doc__}")
    else:
        print("❌ FsmnVadOnline 类不存在")
    
    print("\n模块构建成功！可以开始使用了。")
    print("\n使用示例:")
    print("  from funasr_vad_online import FsmnVadOnline")
    print("  vad = FsmnVadOnline({'model-dir': '/path/to/model'})")
    
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("\n可能的原因:")
    print("1. .so 文件不在当前目录或 Python 路径中")
    print("2. 依赖库（libfunasr.so）未找到")
    print("3. Python 版本不匹配")
    sys.exit(1)
except Exception as e:
    print(f"❌ 发生错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


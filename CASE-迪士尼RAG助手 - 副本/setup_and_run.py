#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
迪士尼RAG助手 - 安装和启动脚本
此脚本将安装必要的依赖并启动RAG助手
"""

import os
import sys
import subprocess
import time

def run_command(command, description="执行命令"):
    """执行命令并显示进度"""
    print(f"🔄 {description}: {' '.join(command)}")
    try:
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(f"✅ {description}成功")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description}失败")
        print(f"错误信息: {e.stderr}")
        return False, e.stderr

def install_packages(packages):
    """安装Python包"""
    for package in packages:
        success, _ = run_command(
            [sys.executable, '-m', 'pip', 'install', '-i', 'https://pypi.tuna.tsinghua.edu.cn/simple', package],
            f"安装 {package}"
        )
        if not success:
            print(f"⚠️ 包 {package} 安装失败，将继续尝试其他包")

def main():
    print("🔄 开始设置迪士尼RAG助手...")
    
    # 确保pip已升级
    run_command([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip', '-i', 'https://pypi.tuna.tsinghua.edu.cn/simple'], "升级pip")
    
    # 安装核心依赖
    core_packages = [
        'numpy',
        'faiss-cpu',
        'transformers>=4.30.0',
        'sentence-transformers>=2.2.0',
        'python-dotenv>=1.0.0'
    ]
    
    print("📦 安装核心依赖包...")
    install_packages(core_packages)
    
    # 安装兼容版本的依赖包
    print("🔄 安装兼容版本的依赖包...")
    
    # 先安装transformers特定版本，然后安装兼容的依赖
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "transformers", "sentence-transformers", "huggingface_hub"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "transformers==4.30.2", "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "sentence-transformers==2.2.2", "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"], check=True)
    
    # 安装兼容版本的huggingface_hub
    subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub==0.16.4", "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"], check=True)
    
    # 安装特定版本的OpenAI，避免proxies参数问题
    print("🔄 安装OpenAI 1.10.0版本以避免proxies参数问题...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "openai"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "openai==1.10.0", "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"], check=True)
    
    # 检查.env文件
    if not os.path.exists('.env'):
        print("📄 创建.env文件...")
        if os.path.exists('.env.example'):
            with open('.env.example', 'r', encoding='utf-8') as f:
                content = f.read()
            with open('.env', 'w', encoding='utf-8') as f:
                f.write(content)
            print("✅ .env文件已创建")
    
    # 启动RAG助手
    print("🚀 启动迪士尼RAG助手...")
    print("\n提示：")
    print("1. 程序将自动安装缺失的依赖")
    print("2. 使用关键词搜索功能无需API密钥")
    print("3. 要启用RAG生成功能，请在.env文件中配置有效的API密钥")
    print("\n正在启动...")
    
    # 运行主程序
    try:
        subprocess.run([sys.executable, '迪士尼RAG检索助手FAISS版.py'])
    except KeyboardInterrupt:
        print("\n🔚 程序已停止")
    except Exception as e:
        print(f"\n❌ 程序运行出错: {e}")
        input("按Enter键退出...")

if __name__ == "__main__":
    main()
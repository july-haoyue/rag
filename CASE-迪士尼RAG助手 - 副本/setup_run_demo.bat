@echo off

:: 迪士尼RAG助手 - 安装和启动脚本
:: 此脚本将安装必要的依赖并启动RAG助手

echo 🔄 开始设置迪士尼RAG助手...

:: 检查Python是否已安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ 未找到Python。请先安装Python 3.7+
    pause
    exit /b 1
)

echo ✅ Python已安装

:: 创建虚拟环境（可选，如果已存在则跳过）
if not exist "venv" (
    echo 📦 创建Python虚拟环境...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo ❌ 创建虚拟环境失败
        pause
        exit /b 1
    )
    
    echo ✅ 虚拟环境创建成功
)

:: 激活虚拟环境
echo 🚀 激活虚拟环境...
call venv\Scripts\activate

:: 升级pip
echo 🔄 升级pip...
pip install --upgrade pip

:: 安装必要的依赖
echo 📦 安装核心依赖包...
pip install numpy faiss-cpu transformers sentence-transformers huggingface_hub python-dotenv openai==1.10.0

:: 安装web界面依赖
if exist "web界面\requirements.txt" (
    echo 📦 安装Web界面依赖...
    cd web界面
    pip install -r requirements.txt
    cd ..
)

echo ✅ 依赖安装完成

:: 检查.env文件
if not exist ".env" (
    echo 📄 复制.env.example为.env...
    if exist ".env.example" (
        copy .env.example .env
        echo 💡 请编辑.env文件，配置您的API密钥
    ) else (
        echo ⚠️ 未找到.env.example文件，请手动创建.env文件
    )
)

:: 启动RAG助手
echo 🚀 启动迪士尼RAG助手...
python 迪士尼RAG检索助手FAISS版.py

:: 如果脚本退出，保持窗口打开
if %errorlevel% neq 0 (
    echo ❌ 程序运行出错
    pause
)

:: 停用虚拟环境
call venv\Scripts\deactivate
echo 🔚 程序已退出
pause
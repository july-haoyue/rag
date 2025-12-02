@echo off

REM 迪士尼RAG助手 - 一键安装与运行脚本
REM 适用于Windows系统

echo ====================================
echo 迪士尼RAG助手 - 安装与运行向导
echo ====================================

REM 检查Python版本
echo 检查Python版本...
python --version
if %ERRORLEVEL% neq 0 (
    echo 错误: 未找到Python。请先安装Python 3.8或更高版本。
    pause
    exit /b 1
)

REM 创建虚拟环境
echo 创建虚拟环境...
python -m venv venv
if %ERRORLEVEL% neq 0 (
    echo 错误: 创建虚拟环境失败。
    pause
    exit /b 1
)

REM 激活虚拟环境
echo 激活虚拟环境...
call venv\Scripts\activate.bat
if %ERRORLEVEL% neq 0 (
    echo 错误: 激活虚拟环境失败。
    pause
    exit /b 1
)

REM 升级pip
echo 升级pip...
python -m pip install --upgrade pip

REM 安装项目依赖
echo 安装项目依赖...
pip install -e .
if %ERRORLEVEL% neq 0 (
    echo 警告: 部分依赖安装可能失败，将尝试继续...
)

REM 创建.env文件（如果不存在）
if not exist .env (
    echo 创建.env配置文件...
    copy .env.example .env
    echo 提示: 请编辑.env文件配置必要的API密钥
)

REM 提示用户编辑配置文件
echo.
echo ====================================
echo 安装完成！
echo ====================================
echo 1. 请编辑.env文件，配置必要的API密钥
echo 2. 选择以下方式启动应用：
echo    - 运行 'python 迪士尼RAG检索助手FAISS版.py' 启动命令行版本
echo    - 运行 'python web界面\app.py' 或 'disney-rag-web' 启动Web版本
echo.
pause

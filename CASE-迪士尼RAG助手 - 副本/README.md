# 迪士尼RAG助手

这是一个基于RAG（检索增强生成）技术的迪士尼知识库问答助手，集成了本地向量检索和网络搜索功能，能够提供准确的迪士尼相关信息回答。

## 快速开始

### 1. 环境准备

```bash
# 克隆或下载项目
# cd 到项目目录

# 创建虚拟环境（推荐）
python -m venv venv

# 激活虚拟环境
# Windows
env\Scripts\activate
# Linux/MacOS
source venv/bin/activate

# 安装项目依赖
pip install -e .

# 安装完整版本（包括可选依赖）
pip install -e "[full]"
```

### 2. 配置环境变量

复制`.env.example`文件为`.env`，并填写必要的API密钥：

```bash
# Windows
copy .env.example .env
# Linux/MacOS
cp .env.example .env
```

编辑`.env`文件，配置以下内容：
```
# OpenAI API配置
OPENAI_API_KEY=your_openai_api_key

# 可选：其他API密钥配置
```

### 3. 运行应用

#### 方法一：直接运行Python文件

```bash
# 运行命令行版本
python 迪士尼RAG检索助手FAISS版.py

# 运行Web界面版本
cd web界面
python app.py
```

#### 方法二：使用安装后的命令（推荐）

```bash
# 运行命令行版本
disney-rag

# 运行Web界面版本
disney-rag-web
```

### 4. 访问Web界面

启动Web服务后，浏览器访问：`http://127.0.0.1:5000`

## 项目结构

```
├── final_index/              # 预构建的向量索引
├── final_processed_data/     # 处理后的知识库数据
├── web界面/                  # Web应用界面
│   ├── static/              # 静态资源
│   ├── templates/           # HTML模板
│   └── app.py              # Flask应用入口
├── hybrid_retriever.py       # 混合检索模块
├── web_searcher.py           # 网络搜索模块
├──迪士尼RAG检索助手FAISS版.py  # 主程序
├── .env.example              # 环境变量示例
├── requirements.txt          # 依赖列表
└── setup.py                  # 安装配置
```

## 系统要求

- Python 3.8+
- 足够的磁盘空间存储模型和索引文件
- 推荐至少8GB内存以支持模型运行

## 注意事项

1. **关于模型下载**：首次运行时会自动下载所需的预训练模型，这可能需要一些时间。

2. **关于API密钥**：确保正确配置API密钥，否则部分功能可能无法正常使用。

3. **关于可选依赖**：`retrieval_evaluator`相关功能是可选的，缺少这些依赖不会影响核心问答功能。

4. **关于知识库**：项目包含预构建的迪士尼知识库索引，无需额外构建即可使用。

## 常见问题

### Q: 运行时遇到 `No module named 'retrieval_evaluator'` 错误怎么办？
A: 这是预期的提示，因为该模块是可选的。系统会自动跳过相关功能，不影响核心问答功能的使用。

### Q: 如何更新知识库？
A: 将新的文档放入迪士尼RAG知识库目录，然后重新构建索引。

### Q: Web服务无法启动怎么办？
A: 确保所有依赖已正确安装，并检查端口5000是否被占用。

## 许可证

[MIT License](LICENSE)

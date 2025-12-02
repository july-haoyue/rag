from setuptools import setup, find_packages
import os

# 读取requirements.txt文件中的依赖
with open('web界面/requirements.txt', 'r', encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]

setup(
    name="disney-rag-assistant",
    version="1.0.0",
    description="迪士尼RAG助手 - 智能问答与知识库系统",
    author="AI大模型应用课程",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    extras_require={
        'full': [
            'evalml==0.62.1',
            'scikit-learn==1.3.2',
            'matplotlib==3.7.3'
        ]
    },
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'disney-rag=迪士尼RAG检索助手FAISS版:main',
            'disney-rag-web=web界面.app:main'
        ]
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    project_urls={
        'Source': 'https://github.com/your-username/disney-rag-assistant',
    },
)

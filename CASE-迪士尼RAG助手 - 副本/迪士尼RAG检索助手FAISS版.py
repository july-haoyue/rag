#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
迪士尼RAG检索助手 - FAISS向量版（简化版）
集成FAISS向量库和阿里云百炼API，实现检索和RAG生成
"""

# 应用huggingface_hub补丁，解决缺失split_torch_state_dict_into_shards函数的问题
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from patch_huggingface_hub import apply_patch
    apply_patch()
    print("✅ 已应用huggingface_hub补丁")
except ImportError as e:
    print(f"⚠️ 无法应用huggingface_hub补丁: {e}")

import re
import json
import time
import numpy as np
import faiss
from typing import List, Dict, Any, Tuple

# 导入混合检索相关模块
try:
    from hybrid_retriever import HybridRetriever
    from web_searcher import WebSearcher
    # retrieval_evaluator设为可选导入
    try:
        from retrieval_evaluator import RetrievalEvaluator
        RETRIEVAL_EVALUATOR_AVAILABLE = True
    except ImportError:
        print("⚠️ retrieval_evaluator模块缺失，将不使用检索评估功能")
        RETRIEVAL_EVALUATOR_AVAILABLE = False
    HAS_HYBRID_RETRIEVAL = True
    print("✅ 混合检索核心模块导入成功")
except ImportError as e:
    print(f"⚠️ 混合检索核心模块导入失败: {e}")
    HAS_HYBRID_RETRIEVAL = False
    RETRIEVAL_EVALUATOR_AVAILABLE = False

# 设置本地缓存目录，优先使用本地模型
cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache', 'huggingface')
os.makedirs(cache_dir, exist_ok=True)
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['SENTENCE_TRANSFORMERS_HOME'] = cache_dir

# 导入必要的库
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    from openai import OpenAI
    # 尝试加载.env文件
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # .env文件支持为可选
    
except ImportError as e:
    print(f"❌ 导入库失败: {e}")

class DisneyRAGAssistant:
    """迪士尼RAG检索助手类 - 简化版"""
    
    def __init__(self, index_dir: str = 'final_index', dashscope_api_key: str = None):
        """初始化RAG助手"""
        self.index_dir = index_dir
        self.inverted_index = {}
        self.chunk_mapping = []   
        self.vector_index = None  
        self.embeddings = None    
        self.embedding_model = None  
        self.rerank_model = None  
        self.ai_client = None  
        self.dashscope_api_key = dashscope_api_key  
        self.initialized = False
        
        # 会话历史管理
        self.conversation_history = []
        self.max_history_length = 5
        
        # 停用词表 - 简化版
        self.stop_words = {
            '的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', 
            '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会',
            'a', 'an', 'the', 'is', 'are', 'and', 'or', 'but', 'for', 'with',
            '吗', '呢', '吧', '啊', '请问', '能否', '是否', '怎么', '如何'
        }
        
        # 模型更新相关配置
        self.update_config_path = os.path.join(cache_dir, 'model_update_config.json')
        self.update_interval_days = 30  # 每月检查一次更新
        
        # 混合检索相关
        self.hybrid_retriever = None
        self.hybrid_retrieval_enabled = False
    
    def check_for_model_updates(self):
        """检查模型更新
        
        每月自动检查一次模型更新，通过记录上次更新时间来控制检查频率
        """
        import datetime
        import json
        
        # 读取或初始化更新配置
        last_update_time = None
        current_time = datetime.datetime.now()
        
        if os.path.exists(self.update_config_path):
            try:
                with open(self.update_config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    if 'last_update_time' in config:
                        last_update_time = datetime.datetime.fromisoformat(config['last_update_time'])
            except Exception as e:
                print(f"⚠️ 读取更新配置失败: {str(e)}")
        
        # 检查是否需要更新（首次运行或超过更新间隔）
        should_update = False
        if last_update_time is None:
            should_update = True
            print("🔄 首次运行，将检查模型更新")
        else:
            days_since_update = (current_time - last_update_time).days
            if days_since_update >= self.update_interval_days:
                should_update = True
                print(f"🔄 距离上次更新已超过 {days_since_update} 天，将检查模型更新")
            else:
                print(f"✅ 模型更新检查跳过，距离下次更新还有 {self.update_interval_days - days_since_update} 天")
        
        if should_update:
            try:
                print("🔄 正在检查并更新本地模型...")
                from huggingface_hub import snapshot_download
                
                # 模型名称映射
                models_to_update = [
                    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
                ]
                
                for model_name in models_to_update:
                    # 下载最新模型到缓存目录
                    model_path = os.path.join(cache_dir, model_name.replace('/', '_'))
                    if not os.path.exists(model_path):
                        print(f"  📥 正在下载模型: {model_name}")
                        try:
                            snapshot_download(
                                repo_id=model_name,
                                cache_dir=cache_dir,
                                local_dir=model_path
                            )
                            print(f"  ✅ 模型下载完成: {model_name}")
                        except Exception as e:
                            print(f"  ⚠️ 模型下载失败: {model_name}, 错误: {str(e)}")
                    else:
                        print(f"  ✅ 模型已存在: {model_name}")
                
                # 更新最后更新时间
                with open(self.update_config_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'last_update_time': current_time.isoformat(),
                        'update_interval_days': self.update_interval_days
                    }, f, ensure_ascii=False, indent=2)
                
                print("✅ 模型更新检查完成")
            except Exception as e:
                print(f"⚠️ 模型更新过程中发生错误: {str(e)}")
                print("   程序将继续使用已有的本地模型")
    
    def load_index(self):
        """加载搜索索引 - 简化版"""
        print(f"正在加载迪士尼RAG知识库索引...")
        
        # 1. 安装必要的依赖
        self._install_required_dependencies()
        
        # 2. 检查模型更新（每月一次）
        self.check_for_model_updates()
        
        # 2. 加载基本数据
        try:
            # 加载倒排索引和切片映射
            index_path = os.path.join(self.index_dir, 'simple_index.json')
            with open(index_path, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            
            self.inverted_index = index_data['inverted_index']
            
            # 加载切片映射
            mapping_path = os.path.join(self.index_dir, 'chunk_mapping.json')
            with open(mapping_path, 'r', encoding='utf-8') as f:
                self.chunk_mapping = json.load(f)
            
            print(f"✅ 基本数据加载成功")
            print(f"- 索引词数量: {len(self.inverted_index)}")
            print(f"- 知识库切片数: {len(self.chunk_mapping)}")
            
        except Exception as e:
            print(f"❌ 基本数据加载失败: {str(e)}")
            self.initialized = False
            return
        
        # 3. 初始化嵌入模型和向量索引
        vector_search_enabled = False
        try:
            # 尝试加载嵌入模型，使用本地模型路径
            print("正在初始化嵌入模型...")
            
            # 使用实例级缓存目录，或创建默认缓存目录
            instance_cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache', 'huggingface')
            os.makedirs(instance_cache_dir, exist_ok=True)
            
            # 尝试使用sentence-transformers库加载模型
            try:
                # 构建本地模型路径
                local_model_path = os.path.join(instance_cache_dir, 'sentence-transformers_paraphrase-multilingual-MiniLM-L12-v2')
                print(f"  优先尝试加载本地模型: {local_model_path}")
                
                # 尝试使用sentence-transformers直接加载本地模型
                self.embedding_model = SentenceTransformer(
                    local_model_path,
                    cache_folder=instance_cache_dir
                )
                print("  ✅ 成功加载本地模型")
            except Exception as e:
                print(f"⚠️ 模型加载失败: {str(e)}")
                print("  注意：由于网络连接问题或库版本不兼容，无法加载向量嵌入模型")
                print("  系统将回退到关键词搜索模式，这可能影响搜索质量但仍可使用基本功能")
                # 设置embedding_model为None以便系统能正确检测到模型加载失败并回退到关键词搜索
                self.embedding_model = None
                  
            # 根据实际加载状态打印正确的信息
            if self.embedding_model is not None:
                print(f"✅ 成功加载嵌入模型")
            else:
                print(f"⚠️ 嵌入模型加载失败，将使用关键词搜索模式")
            
            # 尝试加载或创建向量索引
            vector_index_path = os.path.join(self.index_dir, 'vector_index.faiss')
            embeddings_path = os.path.join(self.index_dir, 'embeddings.npy')
            
            if os.path.exists(vector_index_path) and os.path.exists(embeddings_path):
                try:
                    self.vector_index = faiss.read_index(vector_index_path)
                    self.embeddings = np.load(embeddings_path)
                    print("✅ 成功加载预构建的向量索引和嵌入向量")
                except Exception as load_e:
                    print(f"⚠️ 加载预构建向量索引失败: {str(load_e)}，将重新创建")
                    self._create_vector_index()
            else:
                print("正在创建FAISS向量索引...")
                self._create_vector_index()
            
            vector_search_enabled = True
            print("✅ 向量搜索功能已启用")
        except Exception as e:
            print(f"⚠️ 向量搜索初始化失败: {str(e)}")
            print("   将使用关键词搜索模式")
            self.vector_index = None
            self.embedding_model = None
        
        # 3. 初始化简单的TF-IDF重排序
        try:
            print("正在初始化TF-IDF重排序...")
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            # 准备文档内容
            documents = [chunk['content'] for chunk in self.chunk_mapping]
            
            # 初始化TF-IDF
            self.tfidf_vectorizer = TfidfVectorizer(
                token_pattern=r'[\u4e00-\u9fa5]+|[a-zA-Z]+|[0-9]+',
                max_features=5000
            )
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
            
            print("✅ TF-IDF重排序初始化成功")
        except Exception as e:
            print(f"⚠️ TF-IDF重排序初始化失败: {str(e)}")
        
        # 4. 初始化AI客户端（优先使用OpenAI，其次是阿里云百炼）
        print("🔄 开始初始化AI客户端...")
        self.ai_client = None
        
        try:
            # 导入httpx库，用于创建自定义HTTP客户端
            import httpx
            
            # 动态导入OpenAI，确保版本兼容
            try:
                from openai import OpenAI
                print("  ✅ OpenAI模块导入成功")
            except ImportError as e:
                print(f"  ⚠️ OpenAI模块导入失败: {str(e)}")
                print("  尝试安装或更新openai库...")
                import subprocess
                import sys
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "openai==1.12.0"])
                from openai import OpenAI
                print("  ✅ 重新导入OpenAI成功")
            
            # 直接使用阿里云百炼API
            print("  🔍 直接初始化阿里云百炼API客户端")
            
            # 尝试从环境变量获取API密钥
            self.dashscope_api_key = os.environ.get('ALIYUN_BAILIAN_API_KEY') or os.environ.get('DASHSCOPE_API_KEY')
            print(f"  🔍 检测到阿里云百炼API密钥: {'已设置' if self.dashscope_api_key and not (self.dashscope_api_key.startswith('test_') or self.dashscope_api_key.startswith('your_')) else '未设置或为测试值'}")
            
            if self.dashscope_api_key:
                try:
                    print(f"  🚀 尝试使用阿里云百炼API密钥: {self.dashscope_api_key[:8]}...")
                    # 创建自定义httpx客户端
                    custom_http_client = httpx.Client(timeout=30.0)
                    # 使用自定义http_client和阿里云密钥初始化
                    # 注意：httpx.Client不支持proxies参数，使用标准参数
                    self.ai_client = OpenAI(
                        api_key=self.dashscope_api_key,
                        http_client=custom_http_client,
                        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 直接在初始化时设置base_url
                    )
                    # base_url已在初始化时设置
                    print("  ✅ 已在初始化时设置阿里云百炼API基础URL")
                    print("✅ 阿里云百炼API客户端初始化成功")
                except Exception as e:
                    print(f"⚠️ 阿里云百炼API客户端初始化失败: {str(e)}")
                    self.ai_client = None
            else:
                print("⚠️ 阿里云百炼API密钥格式无效或为测试值")
                self.ai_client = None
        except Exception as e:
            print(f"⚠️ AI客户端初始化异常: {str(e)}")
            self.ai_client = None
        
        if not self.ai_client:
            print("💡 未设置有效的API密钥，RAG生成功能将不可用")
            print("   请在.env文件中配置正确的ALIYUN_BAILIAN_API_KEY")
        
        # 设置初始化状态
        self.initialized = True
        print("✅ 系统初始化完成")
        print(f"- 搜索模式: {'向量搜索+关键词搜索' if vector_search_enabled else '仅关键词搜索'}")
        print(f"- RAG生成功能: {'已启用' if self.ai_client else '已禁用'}")
    
    def _create_vector_index(self):
        """创建FAISS向量索引"""
        # 生成嵌入向量
        chunk_texts = [chunk['content'] for chunk in self.chunk_mapping]
        self.embeddings = self.embedding_model.encode(chunk_texts, show_progress_bar=True)
        
        # 创建并添加到FAISS索引
        dimension = self.embeddings.shape[1]
        self.vector_index = faiss.IndexFlatL2(dimension)
        self.vector_index.add(self.embeddings.astype(np.float32))
        
        print(f"✅ FAISS向量索引创建完成，包含 {self.embeddings.shape[0]} 个向量")
    
    def _install_required_dependencies(self):
        """安装必要的依赖包"""
        print("🔄 检查并安装必要的依赖...")
        try:
            import subprocess
            import sys
            
            # 安装必要的依赖包
            required_packages = [
                'transformers>=4.30.0',
                'sentence-transformers>=2.2.0',
                'huggingface_hub>=0.14.0',
                'python-dotenv>=1.0.0'
            ]
            
            for package in required_packages:
                try:
                    print(f"  检查 {package}...")
                    # 尝试导入包，如果成功则跳过安装
                    if package.split('>=')[0] in ['transformers', 'sentence-transformers', 'huggingface_hub']:
                        __import__(package.split('>=')[0])
                    elif package.startswith('python-dotenv'):
                        __import__('dotenv')
                    print(f"  ✅ {package} 已安装")
                except ImportError:
                    print(f"  📦 正在安装 {package}...")
                    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                    print(f"  ✅ {package} 安装完成")
            
            print("✅ 依赖检查完成")
        except Exception as e:
            print(f"⚠️ 依赖安装过程中发生错误: {str(e)}")
            print("   程序将继续执行，但某些功能可能不可用")
    
    def preprocess_query(self, query: str) -> List[str]:
        """预处理查询文本，提取有效关键词"""
        # 移除特殊字符并标准化
        query = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', ' ', query)
        query = query.strip().lower()
        
        # 提取所有可能的关键词
        keywords = []
        
        # 提取中文词组 (2-4字)
        chinese_chars = re.findall(r'[\u4e00-\u9fa5]', query)
        for i in range(len(chinese_chars)):
            # 生成2-4字词组
            for length in range(2, min(5, len(chinese_chars) - i + 1)):
                word = ''.join(chinese_chars[i:i+length])
                if word not in self.stop_words:
                    keywords.append(word)
        
        # 提取英文单词和数字
        en_num_pattern = r'[a-zA-Z]+|[0-9]+'
        en_num_words = re.findall(en_num_pattern, query)
        keywords.extend([w for w in en_num_words if w not in self.stop_words and len(w) > 1])
        
        # 添加原始查询词作为关键词
        if query and query not in self.stop_words:
            keywords.append(query)
        
        # 去重并过滤掉太短的关键词
        keywords = [k for k in list(set(keywords)) if len(k) >= 2 or k.isdigit()]
        
        # 确保至少有一个关键词
        if not keywords:
            keywords = [query[:10]]  # 使用原始查询的前10个字符
        
        return keywords
    
    def vector_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """使用FAISS向量库进行语义搜索"""
        if not self.initialized or not self.vector_index:
            print("⚠️ 向量索引未初始化")
            return []
        
        # 生成查询向量
        query_vector = self.embedding_model.encode([query])[0]
        
        # 执行向量搜索
        distances, indices = self.vector_index.search(np.array([query_vector]).astype(np.float32), top_k)
        
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            distance = distances[0][i]
            
            # 距离转相关性分数（距离越小越相关）
            score = 1.0 / (1.0 + distance)  # 转换为0-1之间的分数
            
            chunk = self.chunk_mapping[idx]
            result = {
                'content': chunk['content'],
                'metadata': chunk['metadata'],
                'score': score,
                'distance': distance,
                'chunk_id': idx
            }
            results.append(result)
        
        print(f"向量搜索完成，返回 {len(results)} 个结果")
        return results
    
    def rerank_results(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        """使用重排序模型对候选文档进行重排序
        
        Args:
            query: 用户查询
            candidates: 候选文档列表
            top_k: 返回的文档数量
            
        Returns:
            重排序后的文档列表
        """
        # 优先使用外部重排序模型
        if self.rerank_model:
            try:
                print(f"🔄 使用外部重排序模型对 {len(candidates)} 个候选文档进行重排序")
                start_time = time.time()
                
                # 准备查询-文档对，限制长度以避免模型错误
                pairs = []
                for doc in candidates:
                    # 限制文档长度，避免模型处理过长文本
                    content = doc['content'][:500]  # 限制为前500个字符
                    pairs.append((query[:100], content))  # 限制查询长度
                
                # 使用重排序模型计算相关性分数
                scores = self.rerank_model.predict(pairs)
                
                # 更新每个文档的分数
                for i, doc in enumerate(candidates):
                    doc['rerank_score'] = float(scores[i])
                
                # 按重排序分数排序
                reranked_results = sorted(candidates, key=lambda x: x.get('rerank_score', 0), reverse=True)
                
                elapsed_time = time.time() - start_time
                print(f"✅ 外部重排序完成，保留 {top_k} 个最相关文档 (耗时: {elapsed_time:.3f} 秒)")
                
                # 打印重排序后的分数
                for i, doc in enumerate(reranked_results[:top_k]):
                    print(f"  {i+1}. 重排序分数: {doc.get('rerank_score', 0):.4f}")
                
                return reranked_results[:top_k]
                
            except Exception as e:
                print(f"❌ 外部重排序过程发生错误: {str(e)}")
                # 无论如何都尝试使用TF-IDF重排序备选方案
                print("🔄 尝试使用TF-IDF重排序备选方案")
        
        # 使用TF-IDF重排序备选方案（主要的重排序逻辑）
        if hasattr(self, 'tfidf_vectorizer') and hasattr(self, 'tfidf_matrix'):
            print("🔄 使用TF-IDF重排序备选方案")
            return self._tfidf_rerank(query, candidates, top_k)
        
        # 最后的备选：使用简单的关键词匹配重排序
        print("⚠️ 所有重排序方案都不可用，使用简单关键词匹配重排序")
        try:
            # 简单的关键词匹配分数计算，优化中文支持
            # 提取中文关键词
            query_words = set(re.findall(r'[\u4e00-\u9fa5]+|[a-zA-Z]+|[0-9]+', query.lower()))
            
            for doc in candidates:
                # 提取文档中的词语
                content_words = set(re.findall(r'[\u4e00-\u9fa5]+|[a-zA-Z]+|[0-9]+', doc['content'].lower()))
                
                # 计算关键词匹配数量
                common_words = query_words.intersection(content_words)
                # 计算匹配比例
                keyword_match_score = len(common_words) / max(len(query_words), 1)
                
                # 结合原始相似度和关键词匹配分数
                original_score = doc.get('score', 0)
                doc['rerank_score'] = 0.6 * original_score + 0.4 * keyword_match_score
            
            # 按重排序分数排序
            reranked_results = sorted(candidates, key=lambda x: x.get('rerank_score', x.get('score', 0)), reverse=True)
            return reranked_results[:top_k]
        except Exception as e:
            print(f"❌ 简单重排序过程发生错误: {str(e)}")
            # 出错时返回原始候选的前几个
            return candidates[:top_k]
    
    def _tfidf_rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        """使用TF-IDF对候选文档进行重排序"""
        try:
            start_time = time.time()
            
            # 转换查询为TF-IDF向量
            query_vector = self.tfidf_vectorizer.transform([query])
            
            # 计算每个候选文档的TF-IDF相似度
            from sklearn.metrics.pairwise import cosine_similarity
            
            # 为每个候选文档计算TF-IDF相似度
            for i, doc in enumerate(candidates):
                chunk_idx = doc.get('chunk_id')
                if chunk_idx is not None and chunk_idx < self.tfidf_matrix.shape[0]:
                    # 使用chunk_id获取预计算的文档向量
                    doc_vector = self.tfidf_matrix[chunk_idx]
                    tfidf_score = cosine_similarity(query_vector, doc_vector)[0][0]
                else:
                    # 如果没有chunk_id，直接计算当前文档内容的TF-IDF
                    doc_vector = self.tfidf_vectorizer.transform([doc['content']])
                    tfidf_score = cosine_similarity(query_vector, doc_vector)[0][0]
                
                # 结合原始分数和TF-IDF分数
                original_score = doc.get('score', 0)
                doc['rerank_score'] = 0.6 * original_score + 0.4 * float(tfidf_score)
            
            # 按重排序分数排序并返回前top_k个结果
            reranked_results = sorted(candidates, key=lambda x: x.get('rerank_score', 0), reverse=True)
            return reranked_results[:top_k]
            
        except Exception:
            # 出错时使用原始分数排序作为备选方案
            try:
                return sorted(candidates, key=lambda x: x.get('score', 0), reverse=True)[:top_k]
            except:
                # 最后备选：返回原始候选的前几个
                return candidates[:top_k]
    
    def _basic_search(self, query: str, top_k: int = 5, min_score: float = 0.1, use_vector_search: bool = True, use_context: bool = True) -> List[Dict[str, Any]]:
        """基础搜索方法 - 用于混合检索器的本地检索"""
        # 预处理查询
        rewritten_query = query
        if use_context and self.conversation_history and self.ai_client:
            rewritten_query = self.rewrite_query_with_context(query)
            if rewritten_query != query:
                print(f"🔄 查询改写: '{query}' -> '{rewritten_query}'")
        
        # 预处理查询
        keywords = self.preprocess_query(rewritten_query)
        
        results = []
        recall_size = 10  # 粗召回数量
        
        # 1. 向量检索（如果启用）
        if use_vector_search and self.vector_index:
            vector_results = self.vector_search(rewritten_query, top_k=recall_size * 2)
            results.extend(vector_results)
        
        # 2. 关键词检索（作为补充）
        chunk_scores = {}
        
        # 基于索引的关键词匹配
        for keyword in keywords:
            if keyword in self.inverted_index:
                for chunk_id in self.inverted_index[keyword]:
                    chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + 1.0
        
        # 全文模糊匹配
        for chunk_id, chunk in enumerate(self.chunk_mapping):
            content = chunk['content'].lower()
            metadata = chunk['metadata']
            filename = metadata.get('filename', '').lower()
            
            # 计算内容和文件名匹配得分
            content_score = sum(content.count(keyword) * 0.5 for keyword in keywords if keyword in content)
            filename_score = sum(1.0 for keyword in keywords if keyword in filename)
            total_score = content_score + filename_score
            
            if total_score > 0:
                chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + total_score
        
        # 添加关键词检索结果
        keyword_results = []
        for chunk_id, score in chunk_scores.items():
            final_score = min(1.0, score / len(keywords))
            if final_score >= min_score and chunk_id < len(self.chunk_mapping):
                chunk = self.chunk_mapping[chunk_id]
                keyword_results.append({
                    'content': chunk['content'],
                    'metadata': chunk['metadata'],
                    'score': final_score,
                    'is_keyword_match': True
                })
        
        # 按得分排序并添加到结果中
        keyword_results.sort(key=lambda x: x['score'], reverse=True)
        results.extend(keyword_results[:recall_size])
        
        # 3. 合并并去重结果
        unique_results = []
        seen_contents = set()
        
        # 按得分排序所有结果
        results.sort(key=lambda x: x['score'], reverse=True)
        
        for result in results:
            # 使用内容的哈希值进行去重
            content_hash = hash(result['content'][:100])
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique_results.append(result)
                if len(unique_results) >= recall_size:
                    break
        
        # 4. 重排序
        reranked_results = self.rerank_results(rewritten_query, unique_results, top_k=top_k)
        
        return reranked_results
    
    def search(self, query: str, top_k: int = 5, min_score: float = 0.1, use_vector_search: bool = True, use_context: bool = True) -> List[Dict[str, Any]]:
        """搜索相关文档 - 支持混合检索"""
        # 如果启用了混合检索
        if self.hybrid_retrieval_enabled and self.hybrid_retriever:
            try:
                # 执行混合检索
                results, stats = self.hybrid_retriever.retrieve(
                    query,
                    top_k=top_k,
                    min_score=min_score,
                    use_vector_search=use_vector_search,
                    use_context=use_context
                )
                
                # 记录检索统计信息
                if stats.get('used_web_search', False):
                    print(f"🌐 混合检索 - 已使用网络搜索: {stats.get('trigger_reason', '')}")
                
                return results[:top_k]
            except Exception as e:
                print(f"❌ 混合检索失败，回退到本地检索: {str(e)}")
        
        # 回退到基本搜索
        return self._basic_search(query, top_k, min_score, use_vector_search, use_context)
    
    def rewrite_query_with_context(self, query: str) -> str:
        """使用大模型基于上下文历史改写查询"""
        if not self.ai_client or not self.conversation_history:
            return query
        
        try:
            # 构建历史上下文
            history_context = "\n".join([f"用户: {h['query']}\n助手: {h['response_summary']}" for h in self.conversation_history[-3:]])
            
            # 构建查询改写提示词
            prompt = f"""任务：将上下文依赖的用户查询改写为完整独立的查询。

上下文历史:
{history_context}

当前用户查询: {query}

请基于上下文历史，将当前查询改写为一个完整独立的查询，确保它包含所有必要的信息，不需要上下文即可理解。改写后的查询应该保留原始查询的核心意图，但要扩展它以包含上下文信息。

输出要求：仅返回改写后的完整查询文本，不添加任何解释或额外内容。"""
            
            # 调用大模型进行改写 - 兼容不同OpenAI版本
            response = None
            try:
                # 尝试新的API格式 (OpenAI v1.x)
                if hasattr(self.ai_client, 'chat') and hasattr(self.ai_client.chat, 'completions'):
                    response = self.ai_client.chat.completions.create(
                        model="qwen-turbo",
                        messages=[
                            {"role": "system", "content": "你是一个查询改写助手，擅长将上下文依赖的查询转换为独立完整的查询。"},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=200,
                        temperature=0.3,  # 较低的温度，保持一致性
                        timeout=15
                    )
                elif hasattr(self.ai_client, 'chat_completions'):
                    response = self.ai_client.chat_completions.create(
                        model="qwen-turbo",
                        messages=[
                            {"role": "system", "content": "你是一个查询改写助手，擅长将上下文依赖的查询转换为独立完整的查询。"},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=200,
                        temperature=0.3,  # 较低的温度，保持一致性
                        timeout=15
                    )
                else:
                    # 尝试旧的API格式 (OpenAI v0.x)
                    response = self.ai_client.ChatCompletion.create(
                        model="qwen-turbo",
                        messages=[
                            {"role": "system", "content": "你是一个查询改写助手，擅长将上下文依赖的查询转换为独立完整的查询。"},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=200,
                        temperature=0.3,
                        timeout=15
                    )
            except Exception as inner_e:
                print(f"❌ API调用过程中发生错误: {str(inner_e)}")
                return query
            
            rewritten = response.choices[0].message.content.strip()
            return rewritten if rewritten else query
        except Exception as e:
            print(f"❌ 查询改写失败: {str(e)}")
            return query
    
    def update_conversation_history(self, query: str, response_content: str):
        """更新会话历史
        
        Args:
            query: 用户查询
            response_content: 助手响应
        """
        # 为响应生成摘要，用于存储在历史记录中
        response_summary = response_content[:100] + "..." if len(response_content) > 100 else response_content
        
        self.conversation_history.append({
            'query': query,
            'response_summary': response_summary,
            'timestamp': time.time()
        })
        
        # 保持历史记录在最大长度内
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
    
    def clear_conversation_history(self):
        """清除会话历史"""
        self.conversation_history = []
        print("✅ 会话历史已清除")
    
    def setup_hybrid_retrieval(self, config: Dict = None):
        """设置混合检索功能
        
        Args:
            config: 混合检索配置参数
        """
        if not HAS_HYBRID_RETRIEVAL:
            print("❌ 混合检索模块不可用，无法设置")
            return False
        
        try:
            # 定义本地检索器包装器
            class RAGRetrieverWrapper:
                def __init__(self, assistant):
                    self.assistant = assistant
                
                def retrieve(self, query: str, **kwargs):
                    return self.assistant._basic_search(query, **kwargs)
            
            # 创建本地检索器包装器
            rag_retriever_wrapper = RAGRetrieverWrapper(self)
            
            # 创建混合检索器
            self.hybrid_retriever = HybridRetriever(
                rag_retriever=rag_retriever_wrapper,
                config=config
            )
            
            self.hybrid_retrieval_enabled = True
            print("✅ 混合检索功能已启用")
            return True
        except Exception as e:
            print(f"❌ 设置混合检索失败: {str(e)}")
            return False
    
    def toggle_hybrid_retrieval(self, enable: bool):
        """切换混合检索功能的开关
        
        Args:
            enable: 是否启用混合检索
        """
        if not HAS_HYBRID_RETRIEVAL:
            print("❌ 混合检索模块不可用")
            return False
        
        # 如果要启用但还没有初始化
        if enable and not self.hybrid_retriever:
            return self.setup_hybrid_retrieval()
        
        self.hybrid_retrieval_enabled = enable
        print(f"✅ 混合检索已{'启用' if enable else '禁用'}")
        return True
    
    def generate_rag_response(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """使用大模型生成RAG响应
        
        实现完整的RAG流程：
        1. 用户提问
        2. 检索相关文档
        3. 基于检索结果生成回答
        """
        if not self.ai_client:
            return {
                'success': False,
                'error': '客户端未初始化，请提供API密钥'
            }
        
        try:
            # 1. 检索相关文档
            search_results = self.search(query, top_k=top_k)
            
            if not search_results:
                # 如果混合检索启用但没有结果，尝试直接使用网络搜索
                if self.hybrid_retrieval_enabled and self.hybrid_retriever:
                    print("🔄 尝试直接使用网络搜索获取结果")
                    try:
                        # 强制使用网络搜索
                        _, stats = self.hybrid_retriever.retriever.evaluate_query_temporal_need(query)
                        if hasattr(self.hybrid_retriever, 'web_searcher'):
                            search_results = self.hybrid_retriever.web_searcher.search(query, num_results=top_k)
                            # 格式化结果
                            formatted_results = []
                            for i, result in enumerate(search_results):
                                formatted_results.append({
                                    'content': result.get('snippet', ''),
                                    'title': result.get('title', ''),
                                    'score': 0.7 - (i * 0.1),
                                    'source': 'web_search'
                                })
                            search_results = formatted_results
                    except Exception as e:
                        print(f"❌ 强制网络搜索失败: {str(e)}")
            
            if not search_results:
                return {
                    'success': False,
                    'error': '未找到相关信息'
                }
            
            # 2. 构建提示词
            # 区分本地和网络结果
            context_parts = []
            for i, result in enumerate(search_results):
                source = result.get('source', 'local')
                source_tag = "[网络]" if 'web' in source else "[本地]"
                title = result.get('title', '')
                content = result['content'][:300] if len(result['content']) > 300 else result['content']
                
                context_line = f"文档 {i+1} {source_tag}: {title}\n{content}"
                context_parts.append(context_line)
            
            context = "\n\n".join(context_parts)
            
            prompt = f"""你是迪士尼乐园的智能客服助手，请基于以下提供的信息回答用户问题。

已知信息:
{context}

用户问题: {query}

请根据已知信息，用中文回答用户问题。如果已知信息中没有相关内容，请说'我暂时无法回答这个问题'。回答要简洁明了，符合迪士尼乐园的服务风格。"""
            
            # 3. 调用API生成答案 - 兼容不同OpenAI版本
            model_name = "qwen-turbo"
            response = None
            try:
                # 尝试新的API格式 (OpenAI v1.x)
                if hasattr(self.ai_client, 'chat') and hasattr(self.ai_client.chat, 'completions'):
                    response = self.ai_client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": "你是迪士尼乐园的智能客服助手，专业、友好且乐于助人。"},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=500,
                        temperature=0.7,
                        timeout=30
                    )
                elif hasattr(self.ai_client, 'chat_completions'):
                    response = self.ai_client.chat_completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": "你是迪士尼乐园的智能客服助手，专业、友好且乐于助人。"},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=500,
                        temperature=0.7,
                        timeout=30
                    )
                else:
                    # 尝试旧的API格式 (OpenAI v0.x)
                    response = self.ai_client.ChatCompletion.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": "你是迪士尼乐园的智能客服助手，专业、友好且乐于助人。"},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=500,
                        temperature=0.7,
                        timeout=30
                    )
            except Exception as inner_e:
                raise inner_e  # 重新抛出异常，由外部try-except捕获
            
            # 4. 处理响应
            answer = response.choices[0].message.content
            
            # 更新会话历史
            self.update_conversation_history(query, answer)
            
            return {
                'success': True,
                'answer': answer,
                'sources': [
                    {
                        'filename': result['metadata'].get('filename', '未知文件'),
                        'content': result['content'][:200] + '...',
                        'score': result.get('rerank_score', result.get('score', 0))
                    }
                    for result in search_results
                ],
                'model_used': model_name
            }
            
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ["authentication", "invalid api key"]):
                return {'success': False, 'error': "API密钥认证失败，请检查密钥是否正确。"}
            elif any(keyword in error_msg for keyword in ["timeout", "connection"]):
                return {'success': False, 'error': "网络连接问题，请稍后重试。"}
            else:
                return {'success': False, 'error': f"生成回答时出错: {str(e)}"}

    
    def format_result(self, result: Dict[str, Any], index: int) -> str:
        """格式化搜索结果"""
        content = result['content']
        # 简化内容长度限制逻辑
        display_content = content[:200] + ("..." if len(content) > 200 else "")
        
        metadata = result['metadata']
        filename = metadata.get('filename', '未知文件')
        category = metadata.get('category', '未知分类')
        score = result['score']
        
        formatted = f"\n📄 结果 {index + 1} (相关性: {score:.2f})"
        formatted += f"\n   文件: {filename}"
        formatted += f"\n   分类: {category}"
        formatted += f"\n   内容: {display_content}"
        
        return formatted
    
    def search_and_display(self, query: str, top_k: int = 5):
        """搜索并显示结果"""
        results = self.search(query, top_k, use_context=True)
        
        if not results:
            print("❌ 未找到相关信息")
            return
        
        print(f"\n📋 搜索结果:")
        print("-" * 60)
        
        # 生成响应摘要用于更新会话历史
        response_summary = "找到相关信息" if results else "未找到相关信息"
        if results:
            first_result = results[0]['content'][:50] + "..." if len(results[0]['content']) > 50 else results[0]['content']
            response_summary = f"找到{len(results)}条结果，最相关的是: {first_result}"
        
        # 更新会话历史
        self.update_conversation_history(query, response_summary)
        
        for i, result in enumerate(results):
            print(self.format_result(result, i))
            print("-" * 60)
    
    def display_knowledge_base_stats(self):
        """显示知识库统计信息"""
        if not self.initialized:
            print("请先加载索引")
            return
        
        print("\n📊 迪士尼RAG知识库统计信息")
        print("=" * 60)
        print(f"总切片数: {len(self.chunk_mapping)}")
        print(f"索引词数量: {len(self.inverted_index)}")
        print(f"向量索引状态: {'已初始化' if self.vector_index else '未初始化'}")
        
        # 统计分类信息
        categories = {}
        for chunk in self.chunk_mapping:
            category = chunk['metadata'].get('category', '未知分类')
            categories[category] = categories.get(category, 0) + 1
        
        print("\n📂 分类统计:")
        for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {category}: {count} 个切片")

def interactive_mode(assistant):
    """交互式搜索模式"""
    print("\n=== 迪士尼RAG知识库检索助手 ===")
    print("💡 输入问题进行检索，'stats'查看统计，'clear'清空历史，'exit'退出")
    
    while True:
        try:
            # 获取用户输入
            query = input("\n您的问题: ").strip()
            
            # 检查特殊命令
            if query.lower() in ['exit', 'quit', 'q', '退出']:
                print("👋 感谢使用，再见！")
                break
                
            elif query.lower() in ['clear', 'c']:
                assistant.clear_conversation_history()
                print("✅ 对话历史已清空")
                continue
                
            elif query.lower() in ['stats', 's']:
                assistant.display_knowledge_base_stats()
                continue
                
            elif query.lower() in ['help', 'h', '?']:
                print("\n📋 命令说明:")
                print("- 直接输入问题: 自动进行RAG生成回答")
                print("- 'stats': 查看知识库统计")
                print("- 'clear': 清空对话历史")
                print("- 'exit': 退出系统")
                continue
                
            elif not query:
                print("💡 请输入有效的查询内容")
                continue
            
            # 执行RAG生成响应
            response = assistant.generate_rag_response(query)
            if response['success']:
                print(f"\n✅ 回答:")
                print(response['answer'])
                print(f"\n📚 参考来源:")
                for i, source in enumerate(response['sources']):
                    print(f"  {i+1}. {source['filename']} (相关度: {source['score']:.2f})")
            else:
                # 如果RAG失败，尝试普通搜索
                print(f"❌ {response['error']}")
                print("🔄 尝试普通搜索...")
                assistant.search_and_display(query)
                
        except KeyboardInterrupt:
            print("\n\n👋 感谢使用，再见！")
            break
        except Exception as e:
            print(f"❌ 发生错误: {str(e)}")

def main():
    """
    主函数，同时作为命令行工具入口点
    支持通过 'disney-rag' 命令调用
    """
    # 设置工作目录并加载环境变量
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    
    # 尝试加载.env文件
    if os.path.exists('.env'):
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass  # 静默处理缺失的库
    
    # 获取API密钥
    api_key = os.environ.get('ALIYUN_BAILIAN_API_KEY') or os.environ.get('DASHSCOPE_API_KEY')
    
    # 初始化并启动助手
    # 使用正确的索引目录，根据目录结构选择'final_index'
    index_dir = os.path.join(current_dir, 'final_index')
    assistant = DisneyRAGAssistant(index_dir=index_dir, dashscope_api_key=api_key)
    assistant.load_index()
    
    # 显示欢迎信息
    print("\n🚀 迪士尼RAG助手启动成功！")
    print("💡 您可以直接输入问题，或输入以下命令：")
    print("   - 'help': 显示帮助信息")
    print("   - 'exit'/'quit'/'q': 退出程序")
    print("   - 'stats': 显示知识库统计信息")
    print("   - 'clear': 清空对话历史")
    print("\n🔍 示例问题：")
    print("   1. 上海迪士尼乐园门票多少钱？")
    print("   2. 迪士尼乐园有哪些必玩项目？")
    print("\n" + "="*60)
    
    interactive_mode(assistant)

if __name__ == "__main__":
    main()
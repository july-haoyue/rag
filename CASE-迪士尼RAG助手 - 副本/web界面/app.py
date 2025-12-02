from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import sys
import os
import re  # 添加正则表达式模块导入
import math  # 添加数学模块导入，用于计算BM25相关分数

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 先应用huggingface_hub补丁，解决可能的API变更问题
try:
    from patch_huggingface_hub import apply_patch
    apply_patch()
    print("✅ 已应用huggingface_hub补丁")
except Exception as e:
    print(f"⚠️ 应用huggingface_hub补丁时出错: {str(e)}")
    # 即使补丁应用失败也继续运行

# 导入迪士尼RAG助手（使用FAISS向量版）
from 迪士尼RAG检索助手FAISS版 import DisneyRAGAssistant

# 获取当前文件所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 使用绝对路径来指定static_folder和template_folder
app = Flask(__name__, template_folder=os.path.join(current_dir, 'templates'), static_folder=os.path.join(current_dir, 'static'))
CORS(app)  # 启用CORS支持

# 初始化迪士尼RAG助手
rag_assistant = None

def init_assistant():
    global rag_assistant  # 全局变量声明必须在函数开头
    try:
        # 显式使用全局的os模块
        import os as global_os
        
        # 设置正确的索引文件路径（指向项目根目录的final_index）
        project_root = global_os.path.dirname(global_os.path.dirname(global_os.path.abspath(__file__)))
        index_path = global_os.path.join(project_root, 'final_index')
        
        print(f"项目根目录: {project_root}")
        print(f"索引路径: {index_path}")
        print(f"当前工作目录: {global_os.getcwd()}")
        
        # 确保索引目录存在
        if not global_os.path.exists(index_path):
            print(f"错误: 索引目录不存在: {index_path}")
            # 尝试使用其他可能的索引目录
            alternative_paths = [
                global_os.path.join(project_root, 'simple_index'),
                global_os.path.join(project_root, 'fixed_index')
            ]
            for alt_path in alternative_paths:
                if global_os.path.exists(alt_path):
                    index_path = alt_path
                    print(f"使用备用索引目录: {index_path}")
                    break
            else:
                print("无法找到有效的索引目录")
                return False
        
        # 修改当前工作目录为项目根目录，这样索引加载才能正确找到文件
        global_os.chdir(project_root)
        print(f"切换工作目录后: {global_os.getcwd()}")
        
        # 从环境变量读取阿里云百炼API密钥
        dashscope_api_key = global_os.getenv("DASHSCOPE_API_KEY", "")
        aliyun_api_key = global_os.getenv("ALIYUN_API_KEY", "")
        
        # 使用任意一个存在的API密钥
        api_key = dashscope_api_key if dashscope_api_key else aliyun_api_key
        
        if not api_key:
            print("警告: 环境变量DASHSCOPE_API_KEY和ALIYUN_API_KEY都未设置")
            print("提示: 您可以在Windows命令提示符中使用 set DASHSCOPE_API_KEY=your_api_key 或 set ALIYUN_API_KEY=your_api_key 来设置环境变量")
        else:
            print("成功获取API密钥")
        
        # 尝试创建完整版的DisneyRAGAssistant
        try:
            print("尝试创建完整版DisneyRAGAssistant...")
            rag_assistant = DisneyRAGAssistant(index_path, 
                                             dashscope_api_key=api_key)
            print("✅ 完整版DisneyRAGAssistant初始化成功")
            return True
        except Exception as full_error:
            print(f"⚠️ 创建完整版助手失败: {str(full_error)}")
            print("继续创建简化版助手...")
        
        # 创建一个基本版本的助手，跳过复杂的嵌入模型和向量索引初始化
        import json
        
        # 创建一个简化版本的助手类，只实现基本的关键词搜索功能
        class SimpleDisneyAssistant:
            def __init__(self, index_dir):
                self.index_dir = index_dir
                self.inverted_index = {}
                self.chunk_mapping = []
                self.initialized = False
                # 简单的停用词表
                self.stop_words = {
                    '的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', 
                    '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', 
                    '着', '没有', '看', '好', '自己', '这'
                }
            
            def load_index(self):
                try:
                    # 使用global_os来避免名称冲突
                    # 只加载必要的索引文件
                    index_path = global_os.path.join(self.index_dir, 'simple_index.json')
                    with open(index_path, 'r', encoding='utf-8') as f:
                        index_data = json.load(f)
                    self.inverted_index = index_data['inverted_index']
                    
                    # 加载切片映射
                    mapping_path = global_os.path.join(self.index_dir, 'chunk_mapping.json')
                    with open(mapping_path, 'r', encoding='utf-8') as f:
                        self.chunk_mapping = json.load(f)
                    
                    self.initialized = True
                    print("✅ 简化版索引加载成功")
                    print(f"- 索引词数量: {len(self.inverted_index)}")
                    print(f"- 知识库切片数: {len(self.chunk_mapping)}")
                    return True
                except Exception as e:
                    print(f"❌ 简化版索引加载失败: {str(e)}")
                    return False
            
            def preprocess_query(self, query):
                # 简单的关键词提取
                import re
                # 移除特殊字符
                query = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', ' ', query)
                query = query.strip().lower()
                
                # 提取关键词
                keywords = []
                # 简单的中文词组提取
                chinese_pattern = r'[\u4e00-\u9fa5]'
                chinese_chars = re.findall(chinese_pattern, query)
                
                # 提取2-4字的词组
                for i in range(len(chinese_chars)):
                    if i + 1 < len(chinese_chars):
                        word2 = chinese_chars[i] + chinese_chars[i+1]
                        if word2 not in self.stop_words:
                            keywords.append(word2)
                
                # 如果没有提取到关键词，使用原始查询的主要部分
                if not keywords and len(query) > 0:
                    keywords = [query[:10]]
                
                return keywords
            
            def search(self, query, top_k=5, min_score=0.1, use_vector_search=False):
                if not self.initialized:
                    print("⚠️ 请先加载索引")
                    return []
                
                keywords = self.preprocess_query(query)
                print(f"增强版搜索关键词: {keywords}")
                
                if not keywords:
                    return []
                
                # 计算文档总数
                total_docs = len(self.chunk_mapping)
                
                # 基于BM25启发的搜索算法
                chunk_scores = {}
                
                # 计算每个关键词的文档频率（DF）
                keyword_df = {}
                for keyword in keywords:
                    if keyword in self.inverted_index:
                        keyword_df[keyword] = len(self.inverted_index[keyword])
                    else:
                        keyword_df[keyword] = 0
                
                # 1. 倒排索引精确匹配
                for keyword in keywords:
                    if keyword in self.inverted_index:
                        # 计算IDF (逆文档频率)
                        if keyword_df[keyword] > 0:
                            idf = max(0.1, math.log((total_docs - keyword_df[keyword] + 0.5) / (keyword_df[keyword] + 0.5)))
                        else:
                            idf = 0.1
                        
                        for chunk_id in self.inverted_index[keyword]:
                            if chunk_id not in chunk_scores:
                                chunk_scores[chunk_id] = 0
                            
                            # 简单的TF (词频) 计算
                            chunk = self.chunk_mapping[chunk_id]
                            content = chunk.get('content', '').lower()
                            tf = content.count(keyword) / max(1, len(content.split()))
                            
                            # BM25启发的分数计算
                            k1 = 1.2  # BM25参数
                            b = 0.75  # BM25参数
                            avg_doc_length = sum(len(chunk.get('content', '').split()) for chunk in self.chunk_mapping) / max(1, total_docs)
                            doc_length = len(content.split())
                            
                            # 计算最终得分
                            score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_length / avg_doc_length))
                            chunk_scores[chunk_id] += score
                
                # 2. 全文模糊匹配作为补充（处理未在倒排索引中的情况）
                query_lower = query.lower()
                for chunk_id, chunk in enumerate(self.chunk_mapping):
                    content = chunk.get('content', '').lower()
                    metadata = chunk.get('metadata', {})
                    filename = metadata.get('filename', '').lower()
                    
                    # 计算多种匹配指标
                    content_score = 0
                    exact_matches = 0
                    partial_matches = 0
                    
                    # 计算精确匹配和部分匹配
                    for keyword in keywords:
                        if keyword in content:
                            exact_matches += 1
                            # 计算关键词在内容中的位置权重（前面的关键词更重要）
                            pos = content.find(keyword)
                            pos_weight = max(0.5, 1 - pos / max(1, len(content)))
                            content_score += 1.0 * pos_weight
                        elif any(keyword in word for word in content.split()):
                            partial_matches += 1
                            content_score += 0.3  # 部分匹配权重较低
                    
                    # 文件名匹配（权重更高）
                    filename_score = 0
                    for keyword in keywords:
                        if keyword in filename:
                            filename_score += 2.0  # 提高文件名匹配权重
                    
                    # 标题匹配（如果有）
                    title = metadata.get('title', '').lower()
                    title_score = 0
                    for keyword in keywords:
                        if keyword in title:
                            title_score += 1.8  # 标题匹配权重
                    
                    # 计算内容相关性（关键词覆盖率）
                    coverage = exact_matches / len(keywords)
                    
                    # 综合分数
                    total_score = (content_score * 0.5) + (filename_score * 0.3) + (title_score * 0.2)
                    
                    # 应用覆盖率提升
                    if coverage > 0:
                        total_score *= (1 + coverage * 0.5)
                    
                    if total_score > 0:
                        if chunk_id not in chunk_scores:
                            chunk_scores[chunk_id] = 0
                        chunk_scores[chunk_id] += total_score
                
                # 生成结果列表
                results = []
                for chunk_id, score in chunk_scores.items():
                    if score >= min_score:
                        try:
                            chunk = self.chunk_mapping[chunk_id]
                            content = chunk.get('content', '内容不可用')
                            metadata = chunk.get('metadata', {})
                            
                            # 计算额外的相关性指标
                            relevance_score = 0
                            
                            # 检查内容长度（适中的内容更可能是有用的）
                            content_length = len(content)
                            if 50 <= content_length <= 500:
                                relevance_score += 0.2
                            elif content_length > 500:
                                relevance_score += 0.1
                            
                            # 检查是否包含多个关键词
                            keyword_count = 0
                            for keyword in keywords:
                                if keyword in content.lower():
                                    keyword_count += 1
                            if keyword_count >= len(keywords) * 0.7:
                                relevance_score += 0.3
                            
                            # 应用相关性调整
                            final_score = score * (1 + relevance_score)
                            
                            # 归一化分数（使用log缩放避免分数过高）
                            normalized_score = min(1.0, math.log(final_score + 1) / 3)
                            
                            result = {
                                'content': content,
                                'metadata': metadata,
                                'score': normalized_score,
                                'is_keyword_match': True,
                                'keyword_count': keyword_count,
                                'coverage': keyword_count / len(keywords)
                            }
                            results.append(result)
                        except (IndexError, KeyError):
                            continue
                
                # 重排序：综合考虑分数、关键词覆盖率和内容质量
                results.sort(key=lambda x: (x['score'], x['coverage']), reverse=True)
                
                # 移除冗余结果（避免返回过于相似的内容）
                unique_results = []
                seen_contents = set()
                for result in results:
                    # 提取内容指纹用于去重
                    content_fingerprint = ' '.join(sorted([keyword for keyword in keywords if keyword in result['content'].lower()]))
                    content_preview = result['content'][:100]
                    fingerprint = f"{content_fingerprint}:{content_preview}"
                    
                    if fingerprint not in seen_contents:
                        seen_contents.add(fingerprint)
                        unique_results.append(result)
                        if len(unique_results) >= top_k:
                            break
                
                print(f"增强版搜索完成，返回 {len(unique_results)} 个结果，关键词覆盖率最高: {max([r['coverage'] for r in unique_results]) if unique_results else 0:.2f}")
                return unique_results
            
            def generate_rag_response(self, query, top_k=5):
                """
                生成基于检索结果的RAG回答
                参数:
                    query: 用户查询
                    top_k: 返回的结果数量
                返回:
                    基于检索结果的综合回答
                """
                # 首先执行搜索获取相关文档
                search_results = self.search(query, top_k=top_k)
                
                if not search_results:
                    return f"针对'{query}'的问题，我没有找到相关信息，请尝试使用其他关键词。"
                
                # 文本清理函数
                def clean_text(text):
                    import re
                    if not text:
                        return ""
                    
                    # 移除控制字符和乱码
                    cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
                    cleaned = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？.,!?:;；\s\(\)\[\]\{\}"\']', '', cleaned)
                    cleaned = re.sub(r'\s+', ' ', cleaned)
                    return cleaned.strip()
                
                # 尝试使用环境变量中的API密钥调用大模型进行RAG生成
                import os
                aliyun_api_key = os.getenv("ALIYUN_BAILIAN_API_KEY", "")
                
                # 如果有API密钥，尝试调用大模型生成RAG回答
                if aliyun_api_key:
                    try:
                        print("尝试使用阿里云百炼Qwen-Turbo v1模型生成RAG回答...")
                        
                        # 构建上下文信息
                        context = ""
                        source_info = []
                        
                        for i, result in enumerate(search_results[:3], 1):  # 使用前3个最相关的结果
                            content = clean_text(result.get('content', ''))
                            metadata = result.get('metadata', {})
                            filename = metadata.get('filename', '未知来源')
                            
                            context += f"【信息来源{i}: {filename}】\n{content}\n\n"
                            source_info.append(filename)
                        
                        # 构建提示词
                        prompt = f"""上下文构建策略， 
 
 -精准检索：每次问答将严格筛选并采用前3个最相关的检索结果作为信息基础，确保答案的核心性与准确性。 
 
 -内容清理：在构建答案前，系统会自动对原始内容进行净化，去除所有控制字符、乱码及不相关的冗余数据，保证信息的纯净与可读性。 
 
 
 角色定位 
 
 你是一个「迪士尼知识库助手」有丰富的关于迪士尼的资料储备，也能搜索最新状态的新闻，首要根据知识库的储备，次要和网络搜索相结合，总结归纳用户的问题。不单独显示资料来源。 
 
 -信息整合原则 
 我的回答将首要依据内部知识库的丰富资料。当遇到知识库中信息不完整或可能过时的情况，我会谨慎地结合网络上的权威新闻进行补充和更新，确保您获得的信息既准确又全面。 
 
 -角色与语言风格 
 我会全程扮演好"迪士尼知识库助手"的角色，使用热情、亲切且充满故事性的语言与您交流，就像在迪士尼乐园中与您对话一样，为您营造沉浸式的体验。 
 
 -内容呈现方式 
 在回答中，我会自然地融合信息，避免简单罗列要点。对于需要推理或步骤说明的问题，我会采用循序渐进的解释方式，让复杂的内容也变得清晰易懂。所有信息来源都会无缝融入回答，不会出现"根据资料显示"这类生硬的术语。 
 
 -质量保证 
 为了确保信息的准确性，尤其是在结合网络信息时，我会交叉验证多个来源，并优先采纳官方及权威媒体发布的信息。 
 
 -严谨胡乱答复：如果知识库中没有找到相关信息或信息不足，我会坦诚告知，绝不会随意编造或推测内容。

问题: {query}

上下文信息:
{context}

请开始回答:"""
                        
                        # 调用阿里云百炼API
                        import requests
                        import json
                        
                        url = "https://bailian.aliyuncs.com/v1/chat/completions"
                        headers = {
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {aliyun_api_key}"
                        }
                        
                        data = {
                            "model": "qwen-turbo",  # 使用Qwen-Turbo v1模型
                            "messages": [
                                {"role": "system", "content": "你是一个专业的迪士尼知识库助手。"},
                                {"role": "user", "content": prompt}
                            ],
                            "temperature": 0.3,
                            "max_tokens": 1000
                        }
                        
                        response = requests.post(url, headers=headers, json=data, timeout=30)
                        response.raise_for_status()
                        
                        # 解析响应
                        result = response.json()
                        print(f"调试信息：阿里云API响应内容: {json.dumps(result, ensure_ascii=False, indent=2)}")
                        
                        # 安全地获取响应内容
                        try:
                            if 'choices' in result and len(result['choices']) > 0:
                                if 'message' in result['choices'][0] and 'content' in result['choices'][0]['message']:
                                    rag_answer = result['choices'][0]['message']['content'].strip()
                                elif 'text' in result['choices'][0]:
                                    # 某些模型可能使用text字段
                                    rag_answer = result['choices'][0]['text'].strip()
                                else:
                                    raise KeyError("响应格式不包含预期的消息内容字段")
                            else:
                                raise KeyError("响应中没有choices字段或为空")
                        except (KeyError, IndexError) as e:
                            print(f"⚠️ 响应格式解析错误: {str(e)}")
                            # 尝试检查是否有其他可能的响应格式
                            if 'result' in result:
                                rag_answer = str(result['result']).strip()
                            elif 'content' in result:
                                rag_answer = str(result['content']).strip()
                            else:
                                raise KeyError("无法从响应中提取内容")
                        
                        # 添加来源信息
                        if source_info:
                            sources_text = "\n\n相关信息来源: " + "、".join(source_info)
                            if len(rag_answer) + len(sources_text) < 600:
                                rag_answer += sources_text
                        
                        print("✓ 成功使用大模型生成RAG回答")
                        return rag_answer
                        
                    except Exception as e:
                        print(f"⚠️ 阿里云百炼API调用失败: {str(e)}，回退到传统方法")
                        # 如果大模型调用失败，回退到传统的文本拼接方法
                
                # 传统方法：将检索到的内容进行拼接和组织
                # 构建回答
                answer = f"针对'{query}'的问题，根据知识库信息，以下是详细回答：\n\n"
                
                # 添加搜索结果中的关键信息
                seen_info = set()  # 用于去重
                info_count = 0
                max_info = 3  # 最多使用3个结果
                
                for result in search_results[:max_info]:
                    content = result.get('content', '')
                    metadata = result.get('metadata', {})
                    filename = metadata.get('filename', '未知来源')
                    score = result.get('score', 0)
                    
                    # 清理内容
                    cleaned_content = clean_text(content)
                    
                    # 如果内容太短或为空，跳过
                    if len(cleaned_content) < 20:
                        continue
                    
                    # 提取关键句子
                    import re
                    sentences = re.split(r'[.!?。！？\n]+', cleaned_content)
                    relevant_sentences = []
                    
                    # 只保留与查询相关的句子
                    query_lower = query.lower()
                    for sentence in sentences:
                        sentence_lower = sentence.lower()
                        # 检查句子是否包含查询关键词或相关信息
                        if any(keyword in sentence_lower for keyword in query_lower.split()) or len(relevant_sentences) < 2:
                            if len(sentence.strip()) > 10:
                                relevant_sentences.append(sentence.strip())
                                if len(relevant_sentences) >= 2:
                                    break
                    
                    # 添加到回答中
                    if relevant_sentences:
                        info_count += 1
                        answer += f"【{info_count}. {filename}】\n"
                        for sentence in relevant_sentences:
                            if sentence not in seen_info:
                                seen_info.add(sentence)
                                answer += f"{sentence}\n"
                        answer += "\n"
                
                # 如果没有找到足够的信息，使用第一个结果
                if info_count == 0 and search_results:
                    first_result = search_results[0]
                    content = clean_text(first_result.get('content', ''))
                    filename = first_result.get('metadata', {}).get('filename', '未知来源')
                    answer += f"【1. {filename}】\n"
                    answer += f"{content[:300]}...\n"
                
                # 添加结尾
                answer += "以上信息来源于迪士尼知识库，希望能帮助您更好地了解相关内容。"
                
                return answer
        
        # 优先使用简化版助手，避免网络依赖
        print("优先使用简化版迪士尼RAG助手...")
        
        # 检查索引文件是否存在
        chunk_mapping_path = global_os.path.join(index_path, 'chunk_mapping.json')
        simple_index_path = global_os.path.join(index_path, 'simple_index.json')
        if global_os.path.exists(chunk_mapping_path) and global_os.path.exists(simple_index_path):
            print(f"索引文件存在: {chunk_mapping_path}")
            print(f"索引文件存在: {simple_index_path}")
        else:
            print(f"警告: 索引文件不完整")
            print(f"chunk_mapping.json 存在: {global_os.path.exists(chunk_mapping_path)}")
            print(f"simple_index.json 存在: {global_os.path.exists(simple_index_path)}")
        
        # 直接使用简化版助手
        try:
            rag_assistant = SimpleDisneyAssistant(index_dir=index_path)
            load_success = rag_assistant.load_index()
            
            if load_success:
                print("✅ 简化版RAG助手初始化成功")
                print("📋 当前功能状态:")
                print("  - 关键词搜索: 已启用")
                print("  - 向量搜索: 不支持")
                print("  - RAG生成: 基于关键词搜索结果")
                return True
            else:
                print("❌ 简化版助手初始化失败")
                return False
        except Exception as e:
            print(f"❌ 创建简化版助手失败: {str(e)}")
            import traceback
            print(f"错误堆栈: {traceback.format_exc()}")
            return False
    except Exception as e:
        print(f"初始化失败: {e}")
        import traceback
        print(f"错误堆栈: {traceback.format_exc()}")
        return False

# 首页路由
@app.route('/')
def index():
    return render_template('index.html')

# 搜索API
@app.route('/api/search', methods=['POST'])
def search():
    if not rag_assistant:
        return jsonify({
            'success': False,
            'error': '助手尚未初始化'
        }), 503
    
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        use_vector_search = data.get('use_vector_search', True)  # 新增参数：是否使用向量搜索
        top_k = data.get('top_k', 5)  # 新增参数：返回结果数量
        
        if not query:
            return jsonify({
                'success': False,
                'error': '查询内容不能为空'
            }), 400
        
        # 执行搜索
        print(f"执行搜索查询: {query}, 向量搜索: {use_vector_search}")
        results = rag_assistant.search(query, top_k=top_k, use_vector_search=use_vector_search)
        
        # 检查结果是否为None
        if results is None:
            results = []
            print("注意：搜索返回None，已转换为空列表")
        
        print(f"搜索结果数量: {len(results)}")
        
        # 格式化结果
        formatted_results = []
        for i, result in enumerate(results):
            print(f"结果 {i+1}: {result.keys()}")
            # 确保结果中包含必要的键
            metadata = result.get('metadata', {})
            filename = metadata.get('filename', '未知文件')
            content = result.get('content', '内容不可用')
            score = result.get('score', 0)
            
            # 提取分类信息（从文件名）
            category = '通用'
            
            # 根据文件名简单分类
            if '门票' in filename or '票务' in filename:
                category = '门票信息'
            elif '酒店' in filename:
                category = '酒店服务'
            elif '项目' in filename or '游玩' in filename:
                category = '游乐项目'
            elif '餐饮' in filename:
                category = '餐饮服务'
            elif '会员' in filename or '尊享' in filename:
                category = '会员服务'
            elif '地图' in filename or '区域' in filename:
                category = '园区导览'
            elif '攻略' in filename:
                category = '游玩攻略'
            
            # 添加搜索类型标记
            search_type = '混合检索'
            if 'is_keyword_match' in result and not use_vector_search:
                search_type = '关键词检索'
            elif 'distance' in result:
                search_type = '向量检索'
            
            formatted_results.append({
                'score': f"{score:.4f}",
                'filename': filename,
                'content': content,
                'category': category,
                'search_type': search_type
            })
        
        return jsonify({
            'success': True,
            'results': formatted_results,
            'search_type': '混合检索' if use_vector_search else '关键词检索'
        })
        
    except Exception as e:
        print(f"搜索错误详细信息: {str(e)}")
        import traceback
        print(f"错误堆栈: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# RAG生成API - 隐藏AI相关描述
@app.route('/api/generate', methods=['POST'])
def generate_rag():
    """生成知识库回答"""
    global rag_assistant  # 声明使用全局变量
    if not rag_assistant:
        return jsonify({
            'success': False,
            'error': '助手尚未初始化'
        }), 503
    
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        top_k = data.get('top_k', 5)  # 增加默认返回的结果数量
        
        if not query:
            return jsonify({
                'success': False,
                'error': '查询内容不能为空'
            }), 400
        
        print(f"执行搜索查询: {query}")
        
        # 首先检查索引是否初始化成功
        if not hasattr(rag_assistant, 'initialized') or not rag_assistant.initialized:
            # 尝试重新加载索引
            print("索引未初始化，尝试重新加载...")
            rag_assistant.load_index()
            if not rag_assistant.initialized:
                return jsonify({
                    'success': False,
                    'error': '知识库索引加载失败',
                    'note': '请检查索引文件是否存在'
                })
        
        # 执行搜索
        results = rag_assistant.search(query, top_k=top_k)
        
        # 即使搜索结果为空，也继续执行生成过程
        # 这样我们的generate_rag_response方法仍然会被调用
        if not results:
            print("⚠️ 搜索结果为空，但仍会尝试生成回答")
            # 创建空的formatted_results，允许后续流程继续
            formatted_results = []
        
        # 只有在results不为空时才格式化搜索结果
        if results and 'formatted_results' not in locals():
            formatted_results = []
            for result in results:
                metadata = result.get('metadata', {})
                filename = metadata.get('filename', '未知文件')
                content = result.get('content', '内容不可用')
                score = result.get('score', 0)
                
                # 提取分类信息
                category = '通用'
                if '门票' in filename or '票务' in filename:
                    category = '门票信息'
                elif '酒店' in filename:
                    category = '酒店服务'
                elif '项目' in filename or '游玩' in filename:
                    category = '游乐项目'
                elif '餐饮' in filename:
                    category = '餐饮服务'
                elif '会员' in filename or '尊享' in filename:
                    category = '会员服务'
                elif '地图' in filename or '区域' in filename:
                    category = '园区导览'
                elif '攻略' in filename:
                    category = '游玩攻略'
                
                formatted_results.append({
                    'filename': filename,
                    'content': content,
                    'score': score,
                    'category': category
                })
        
        # 优化的简洁总结生成，更好地提炼用户问题和解决方案
        def generate_brief_summary(query, results):
            """生成不超过600字的详细总结，准确提炼用户问题并提供结构化解决方案"""
            # 文本清理函数，增强乱码处理能力
            def clean_text(text):
                if not text:
                    return ""
                
                # 第一步：移除控制字符和不可见字符
                cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
                
                # 第二步：处理可能的编码混合问题
                # 移除明显的乱码模式（连续的非中文字符且不是英文/数字）
                # 保留基本的中文、英文、数字和常见标点
                cleaned = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？.,!?:;；\s\(\)\[\]\{\}\"\']', '', cleaned)
                
                # 第三步：移除连续的空白字符
                cleaned = re.sub(r'\s+', ' ', cleaned)
                
                # 第四步：移除连续的标点符号
                cleaned = re.sub(r'(，|。|！|？|,|\.|!|\?|:|；|;)\1+', '\1', cleaned)
                
                # 第五步：处理常见的编码错误模式
                # 移除明显的乱码字符组（5个以上的连续非中文/非英文/非数字字符）
                cleaned = re.sub(r'([^\u4e00-\u9fa5a-zA-Z0-9\s]){5,}', '', cleaned)
                
                return cleaned.strip()
            
            # 分析用户问题类型，提取核心问题
            def analyze_query(query):
                # 根据常见问题类型进行分类
                if any(word in query for word in ['哪里', '位置', '地图', '路线']):
                    return '位置查询', '位置信息'
                elif any(word in query for word in ['时间', '开放', '闭园', '表演', '烟花', '巡游']):
                    return '时间查询', '时间安排'
                elif any(word in query for word in ['票价', '门票', '价格', '多少钱']):
                    return '价格查询', '票价信息'
                elif any(word in query for word in ['项目', '游玩', '必玩', '刺激', '适合']):
                    return '项目查询', '推荐项目'
                elif any(word in query for word in ['餐饮', '餐厅', '吃', '食物']):
                    return '餐饮查询', '推荐餐厅'
                elif any(word in query for word in ['酒店', '住宿', '房间']):
                    return '住宿查询', '酒店信息'
                elif any(word in query for word in ['攻略', '建议', '提示', '技巧']):
                    return '攻略咨询', '实用建议'
                else:
                    return '一般咨询', '相关信息'
            
            # 提取重要的信息片段，按问题类型和解决方案分类
            problem_type, solution_category = analyze_query(query)
            core_answers = []  # 核心解决方案
            supplementary_info = []  # 补充信息
            max_length = 550  # 预留结尾空间
            
            # 按相关性优先处理结果
            for result in results[:3]:  # 只处理前3个结果
                raw_content = result.get('content', '').strip()
                content = clean_text(raw_content)
                
                # 提取短句并清理
                sentences = re.split(r'[.!?。！？\n]+', content)
                sentences = [clean_text(s) for s in sentences if len(clean_text(s)) > 5]
                
                # 更精确的关键词分析，区分问题关键词和解决方案关键词
                query_lower = query.lower()
                # 解决方案关键词列表
                solution_keywords = {
                    '位置查询': ['位于', '在', '地址', '地图', '方向'],
                    '时间查询': ['开放', '闭园', '开始', '结束', '时间', '几点', '分钟', '小时'],
                    '价格查询': ['元', '价格', '票价', '优惠', '折扣', '免费'],
                    '项目查询': ['项目', '游玩', '设施', '体验', '身高', '年龄'],
                    '餐饮查询': ['餐厅', '食物', '套餐', '价格', '推荐'],
                    '住宿查询': ['酒店', '房间', '入住', '退房', '价格'],
                    '攻略咨询': ['建议', '提示', '技巧', '推荐', '注意']
                }
                
                # 获取当前问题类型的解决方案关键词
                current_solution_keywords = solution_keywords.get(problem_type, [])
                
                for sentence in sentences:
                    sentence_lower = sentence.lower()
                    # 检查是否包含问题核心词汇或解决方案关键词
                    contains_query_core = any(keyword in sentence_lower for keyword in query_lower.split())
                    contains_solution_keyword = any(keyword in sentence_lower for keyword in current_solution_keywords)
                    has_specific_info = any(char.isdigit() for char in sentence) or any(word in sentence for word in ['是', '位于', '提供', '开放'])
                    
                    # 优先选择解决方案类信息
                    if contains_solution_keyword and has_specific_info:
                        if len(sentence) < 250 and len(''.join(core_answers)) + len(sentence) < max_length * 0.8:
                            core_answers.append(sentence)
                    # 补充信息次之
                    elif contains_query_core and not contains_solution_keyword:
                        if len(sentence) < 200 and len(''.join(core_answers + supplementary_info)) + len(sentence) < max_length:
                            supplementary_info.append(sentence)
            
            # 构建结构化回答
            summary = f"针对'{query}'的问题，"
            
            # 优先添加核心解决方案
            if core_answers:
                summary += f"以下是{solution_category}："
                # 添加第一个核心解决方案
                summary += core_answers[0]
                # 添加其他核心解决方案
                for i, answer in enumerate(core_answers[1:], 1):
                    connector = '另外，' if i == 1 else '还有，'
                    if len(summary + connector + answer) <= max_length:
                        summary += connector + answer
                    else:
                        break
            
            # 适当添加补充信息
            if supplementary_info and len(summary) < max_length * 0.9:
                for info in supplementary_info:
                    connector = '补充说明，' if not core_answers else '需要说明的是，'
                    if len(summary + connector + info) <= max_length:
                        summary += connector + info
                    else:
                        break
            
            # 如果没有找到合适的信息，提供默认回答
            if len(summary) <= len(f"针对'{query}'的问题，") + len(f"以下是{solution_category}："):
                if results and 'content' in results[0]:
                    first_content = clean_text(results[0]['content'][:max_length - len(summary) - 5])
                    summary += f"我找到了一些相关信息：{first_content}..."
                else:
                    summary += "我找到了一些相关信息，请查看下方参考来源获取详情。"
            
            # 最终清理并确保总长度不超过600字
            summary = clean_text(summary)
            if len(summary) > 600:
                summary = summary[:597] + "..."
            
            return summary
        
        try:
            # 尝试使用DisneyRAGAssistant的generate_rag_response方法生成真正的RAG报告
            print(f"📝 使用RAG生成完整回答: {query}")
            
            # 准备文档内容列表
            docs = []
            for result in formatted_results:
                if isinstance(result, dict) and 'content' in result:
                    docs.append(result['content'])
            
            # 调用generate_rag_response方法生成完整RAG报告
            if hasattr(rag_assistant, 'generate_rag_response'):
                try:
                    print(f"尝试使用generate_rag_response方法生成回答...")
                    # 正确调用RAG生成方法，只传入query参数
                    rag_response = rag_assistant.generate_rag_response(query, top_k=5)
                    if rag_response and rag_response.get('success'):
                        print("✓ 使用generate_rag_response生成回答成功")
                        return jsonify({
                            'success': True,
                            'answer': rag_response.get('answer', ''),
                            'sources': rag_response.get('sources', []),
                            'model_used': rag_response.get('model_used', ''),
                            'fallback_results': formatted_results,
                            'note': '使用generate_rag_response方法生成',
                            'rag_type': 'full_rag'
                        })
                    else:
                        print("⚠️ RAG生成返回空结果，直接返回提示")
                        # 关停回退机制，返回固定提示语
                        return jsonify({
                            'success': True,
                            'answer': '对不起，我暂时没有办法帮您解决您的问题。',
                            'fallback_results': formatted_results,
                            'note': 'RAG生成失败，已关停回退机制',
                            'rag_type': 'direct_message'
                        })
                except Exception as rag_error:
                    print(f"✗ RAG生成错误: {str(rag_error)}")
                    # 关停回退机制，返回固定提示语
                    return jsonify({
                        'success': True,
                        'answer': '对不起，我暂时没有办法帮您解决您的问题。',
                        'fallback_results': formatted_results,
                        'note': f'RAG生成失败，已关停回退机制: {str(rag_error)}',
                        'rag_type': 'direct_message'
                    })
            else:
                print("⚠️ RAG助手没有generate_rag_response方法，直接返回提示")
                # 关停回退机制，返回固定提示语
                return jsonify({
                    'success': True,
                    'answer': '对不起，我暂时没有办法帮您解决您的问题。',
                    'fallback_results': formatted_results,
                    'note': 'RAG助手没有generate_rag_response方法，已关停回退机制',
                    'rag_type': 'direct_message'
                })
        except Exception as e:
            print(f"❌ 生成回答时出错: {str(e)}")
            import traceback
            print(f"错误堆栈: {traceback.format_exc()}")
            # 返回原始RAG结果作为备选
            # 使用generate_brief_summary函数生成基于搜索结果的摘要
            try:
                fallback_answer = generate_brief_summary(query, formatted_results)
                return jsonify({
                    'success': True,
                    'answer': fallback_answer,
                    'fallback_results': formatted_results,
                    'note': f'RAG生成出错，返回备选摘要: {str(e)}',
                    'rag_type': 'fallback_summary'
                })
            except:
                # 如果摘要生成也失败，返回基础信息
                return jsonify({
                    'success': True,
                    'answer': f"已为您找到{len(formatted_results)}条相关信息，详情请查看下方参考来源。",
                    'fallback_results': formatted_results,
                    'note': 'RAG生成和摘要生成都失败',
                    'rag_type': 'basic_info'
                })
        
    except Exception as e:
        print(f"搜索错误: {str(e)}")
        import traceback
        print(f"错误堆栈: {traceback.format_exc()}")
        # 捕获特定错误并返回友好信息
        error_msg = str(e)
        if 'API密钥' in error_msg or 'token' in error_msg.lower() or '认证' in error_msg:
            return jsonify({
                'success': False,
                'error': '系统缺少必要的API密钥，请配置环境变量DASHSCOPE_API_KEY或ALIYUN_API_KEY'
            })
        elif 'DeepSeek' in error_msg:
            return jsonify({
                'success': False,
                'error': 'DeepSeek API错误，请检查API密钥配置'
            })
        elif '请先加载索引' in error_msg:
            return jsonify({
                'success': False,
                'error': '知识库索引未加载，请检查索引文件是否存在'
            })
        else:
            return jsonify({
                'success': False,
                'error': f'处理请求时出错: {error_msg}',
                'note': '请尝试使用其他关键词或稍后再试'
            })

# 配置管理API
@app.route('/api/config', methods=['GET'])
def get_config():
    """获取系统配置信息"""
    # 检测是否有阿里云百炼API密钥环境变量
    has_api_key = ('DASHSCOPE_API_KEY' in os.environ and os.environ['DASHSCOPE_API_KEY']) or \
                 ('ALIYUN_API_KEY' in os.environ and os.environ['ALIYUN_API_KEY'])
    
    # 检查是否使用了完整版助手
    is_full_version = hasattr(rag_assistant, 'generate_rag_response') if rag_assistant else False
    
    return jsonify({
        'success': True,
        'features': {
            'vector_search': is_full_version,  # 如果是完整版则支持向量搜索
            'rag_generation': has_api_key,  # 只有有API密钥时才支持生成
            'keyword_search': True,
            'rerank': is_full_version  # 如果是完整版则支持重排序
        },
        'model_info': None,
        'environment': {
            'has_api_key': has_api_key,
            'mode': 'full' if is_full_version else 'simplified'  # 当前运行模式
        }
    })

# 统计信息API
@app.route('/api/stats', methods=['GET'])
def stats():
    """获取知识库统计信息"""
    global rag_assistant  # 全局变量声明必须在使用前
    try:
        # 检查助手是否已初始化
        if not rag_assistant:
            return jsonify({
                'success': False,
                'error': '助手尚未初始化'
            }), 503
        
        print("开始获取统计信息...")
        
        # 使用正确的属性名称
        total_chunks = len(rag_assistant.chunk_mapping)
        total_index_words = len(rag_assistant.inverted_index)
        
        print(f"统计信息: 总切片数={total_chunks}, 总索引词={total_index_words}")
        
        # 统计分类信息
        categories = {}
        
        # 遍历chunk_mapping来获取分类信息
        for chunk in rag_assistant.chunk_mapping:
            metadata = chunk.get('metadata', {})
            
            # 尝试从metadata中获取分类或文件名
            category = '通用'
            
            # 优先从metadata获取filename
            if isinstance(metadata, dict):
                filename = metadata.get('filename', '未知文件')
                
                # 基于文件名进行分类
                if '门票' in filename or '票务' in filename:
                    category = '门票信息'
                elif '酒店' in filename:
                    category = '酒店服务'
                elif '项目' in filename or '游玩' in filename:
                    category = '游乐项目'
                elif '餐饮' in filename:
                    category = '餐饮服务'
                elif '会员' in filename or '尊享' in filename:
                    category = '会员服务'
                elif '地图' in filename or '区域' in filename:
                    category = '园区导览'
                elif '攻略' in filename:
                    category = '游玩攻略'
            
            categories[category] = categories.get(category, 0) + 1
        
        print(f"分类统计结果: {categories}")
        
        # 热门搜索词
        popular_searches = [
            '迪士尼乐园门票', 
            '会员权益', 
            '商品退换政策', 
            '服务时间', 
            '儿童项目'
        ]
        
        return jsonify({
            'success': True,
            'stats': {
                'total_chunks': total_chunks,
                'total_index_words': total_index_words,
                'categories': categories
            },
            'popular_searches': popular_searches
        })
    except Exception as e:
        print(f"统计API错误: {str(e)}")
        import traceback
        print(f"错误堆栈: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': f'获取统计信息时出错: {str(e)}'
        }), 500

def main():
    """
    主函数，用于支持命令行调用（通过setup.py中的entry_points配置）
    """
    print("初始化迪士尼RAG助手...")
    
    init_success = init_assistant()
    if init_success:
        print("✅ RAG助手初始化成功")
    else:
        print("⚠️ RAG助手初始化失败，将以有限功能运行")

    print("启动Flask服务器...")
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == '__main__':
    main()
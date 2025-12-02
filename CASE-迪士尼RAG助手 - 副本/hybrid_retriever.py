from typing import Dict, List, Tuple, Optional, Any
import time
# retrieval_evaluator设为可选导入
try:
    from retrieval_evaluator import RetrievalEvaluator
    RETRIEVAL_EVALUATOR_AVAILABLE = True
except ImportError:
    print("⚠️ retrieval_evaluator模块缺失，将不使用检索评估功能")
    RETRIEVAL_EVALUATOR_AVAILABLE = False
from web_searcher import WebSearcher

class HybridRetriever:
    """
    混合检索管理器
    协调本地RAG检索与网络搜索，实现智能的混合检索策略
    """
    def __init__(self, 
                 rag_retriever=None, 
                 web_searcher: Optional[WebSearcher] = None,
                 config: Optional[Dict] = None):
        """
        初始化混合检索器
        
        Args:
            rag_retriever: 本地RAG检索器实例
            web_searcher: 网络搜索引擎实例
            config: 配置参数
        """
        self.rag_retriever = rag_retriever
        self.web_searcher = web_searcher or WebSearcher()
        # 只有在模块可用时才初始化evaluator
        self.evaluator = RetrievalEvaluator() if RETRIEVAL_EVALUATOR_AVAILABLE else None
        
        # 配置参数
        self.config = {
            'enable_web_search': True,
            'search_timeout': 10.0,  # 网络搜索超时时间（秒）
            'max_search_results': 3,  # 最大使用的网络搜索结果数
            'min_local_score': 0.7,   # 本地检索结果的最小分数
            'local_results_weight': 0.6,  # 本地结果在混合排序中的权重
            'search_results_weight': 0.4, # 网络结果在混合排序中的权重
            'timeout_fallback': True,     # 网络搜索超时时是否回退到本地结果
            **(config or {})
        }
        
        print(f"混合检索器初始化完成 - 配置: {self.config}")
    
    def _run_rag_retrieval(self, query: str, **kwargs) -> List[Dict]:
        """
        执行本地RAG检索
        
        Args:
            query: 用户查询
            **kwargs: 传递给RAG检索器的额外参数
        
        Returns:
            检索结果列表
        """
        if not self.rag_retriever:
            print("警告: 本地RAG检索器未提供")
            return []
        
        try:
            start_time = time.time()
            # 调用本地检索器
            results = self.rag_retriever.retrieve(query, **kwargs)
            end_time = time.time()
            
            print(f"本地RAG检索完成 - 查询: '{query}' - 耗时: {end_time - start_time:.2f}秒 - 结果数: {len(results)}")
            
            # 确保结果包含score字段
            for i, result in enumerate(results):
                if 'score' not in result:
                    result['score'] = 1.0 - (i * 0.1)  # 简单的位置得分
                result['source'] = 'local_rag'
            
            return results
        except Exception as e:
            print(f"本地RAG检索失败: {str(e)}")
            return []
    
    def _run_web_search(self, query: str, priority: str = 'medium') -> List[Dict]:
        """
        执行网络搜索
        
        Args:
            query: 用户查询
            priority: 搜索优先级 ('high', 'medium', 'low')
        
        Returns:
            搜索结果列表
        """
        if not self.config['enable_web_search']:
            print("网络搜索已禁用")
            return []
        
        try:
            # 根据优先级调整搜索参数
            search_params = {
                'num_results': self.config['max_search_results'],
                'timeout': self.config['search_timeout']
            }
            
            if priority == 'high':
                search_params['sources'] = ['baidu', 'disney_official']  # 高优先级使用多种源
                search_params['num_results'] = min(search_params['num_results'] + 1, 5)
            elif priority == 'low':
                search_params['sources'] = ['baidu']  # 低优先级仅使用百度
                search_params['timeout'] = max(search_params['timeout'] - 2.0, 3.0)  # 减少超时
            
            start_time = time.time()
            results = self.web_searcher.search(query, **search_params)
            end_time = time.time()
            
            print(f"网络搜索完成 - 查询: '{query}' - 优先级: {priority} - 耗时: {end_time - start_time:.2f}秒 - 结果数: {len(results)}")
            
            # 处理搜索结果
            search_results = []
            for i, result in enumerate(results[:self.config['max_search_results']]):
                # 为网络搜索结果分配分数（基于位置和来源）
                base_score = 0.8 - (i * 0.15)
                if result.get('source') == 'disney_official':
                    source_score = 1.2  # 官方来源权重更高
                else:
                    source_score = 1.0
                
                search_result = {
                    'content': result.get('snippet', ''),
                    'title': result.get('title', ''),
                    'url': result.get('url', ''),
                    'source': f'web_{result.get("source", "unknown")}',
                    'score': base_score * source_score,
                    'retrieved_at': time.time()
                }
                search_results.append(search_result)
            
            return search_results
        except Exception as e:
            print(f"网络搜索失败: {str(e)}")
            if self.config['timeout_fallback']:
                print("触发回退机制，继续使用本地检索结果")
            return []
    
    def _combine_results(self, rag_results: List[Dict], search_results: List[Dict]) -> List[Dict]:
        """
        组合本地检索和网络搜索结果
        
        Args:
            rag_results: 本地检索结果
            search_results: 网络搜索结果
        
        Returns:
            组合并排序后的结果列表
        """
        all_results = []
        
        # 处理本地结果
        for result in rag_results:
            weighted_score = result['score'] * self.config['local_results_weight']
            all_results.append({
                **result,
                'weighted_score': weighted_score,
                'original_score': result['score']
            })
        
        # 处理网络结果
        for result in search_results:
            weighted_score = result['score'] * self.config['search_results_weight']
            all_results.append({
                **result,
                'weighted_score': weighted_score,
                'original_score': result['score']
            })
        
        # 根据加权分数排序（降序）
        all_results.sort(key=lambda x: x['weighted_score'], reverse=True)
        
        print(f"结果组合完成 - 总结果数: {len(all_results)}")
        return all_results
    
    def _filter_redundant_results(self, results: List[Dict], threshold: float = 0.8) -> List[Dict]:
        """
        过滤冗余结果（简单实现）
        
        Args:
            results: 结果列表
            threshold: 相似度阈值
        
        Returns:
            去重后的结果列表
        """
        if not results:
            return []
        
        # 简单的去重策略：检查URL或内容相似性
        unique_results = []
        seen_urls = set()
        seen_contents = []
        
        for result in results:
            # 检查URL重复
            url = result.get('url', '')
            if url and url in seen_urls:
                continue
            if url:
                seen_urls.add(url)
            
            # 检查内容相似度（简单实现）
            content = result.get('content', '')
            is_redundant = False
            if content and seen_contents:
                # 计算内容长度相似度（简化版）
                for seen_content in seen_contents:
                    min_len = min(len(content), len(seen_content))
                    max_len = max(len(content), len(seen_content))
                    if min_len / max_len > threshold:
                        # 简单判断：如果长内容包含短内容，则认为冗余
                        if (len(content) > len(seen_content) and seen_content in content) or \
                           (len(seen_content) > len(content) and content in seen_content):
                            is_redundant = True
                            break
            
            if not is_redundant:
                unique_results.append(result)
                if content:
                    seen_contents.append(content)
        
        return unique_results
    
    def retrieve(self, query: str, **kwargs) -> Tuple[List[Dict], Dict]:
        """
        执行混合检索
        
        Args:
            query: 用户查询
            **kwargs: 额外参数
        
        Returns:
            (检索结果, 检索统计信息)
        """
        stats = {
            'query': query,
            'timestamp': time.time(),
            'rag_results_count': 0,
            'web_results_count': 0,
            'combined_results_count': 0,
            'used_web_search': False,
            'trigger_reason': '',
            'evaluation_metrics': {}
        }
        
        print(f"\n===== 开始混合检索 =====")
        print(f"查询: '{query}'")
        
        # 1. 执行本地RAG检索
        rag_results = self._run_rag_retrieval(query, **kwargs)
        stats['rag_results_count'] = len(rag_results)
        
        # 初始化变量
        use_web_search = False
        search_priority = 'medium'
        web_results = []
        eval_metrics = {}
        
        # 2. 评估是否需要使用网络搜索
        if self.evaluator is not None:
            use_web_search, eval_metrics = self.evaluator.should_use_web_search(query, rag_results)
            stats['evaluation_metrics'] = eval_metrics
            
            # 决定是否执行网络搜索
            if use_web_search and self.config['enable_web_search']:
                # 获取搜索优先级
                search_priority = self.evaluator.get_search_priority(query)
                # 执行网络搜索
                web_results = self._run_web_search(query, priority=search_priority)
        else:
            # 没有evaluator时的默认逻辑
            eval_metrics = {'reason': '评估器不可用，使用默认策略'}
            stats['evaluation_metrics'] = eval_metrics
            
            # 默认策略：如果本地结果分数低于阈值或结果太少，则使用网络搜索
            if self.config['enable_web_search']:
                # 检查是否有足够的高质量本地结果
                if len(rag_results) == 0 or (len(rag_results) > 0 and rag_results[0].get('score', 0) < self.config['min_local_score']):
                    use_web_search = True
                    # 执行网络搜索
                    web_results = self._run_web_search(query, priority=search_priority)
        
        # 3. 处理网络搜索结果
        if use_web_search and web_results:
            stats['web_results_count'] = len(web_results)
            stats['used_web_search'] = True
            stats['trigger_reason'] = eval_metrics.get('reason', '未知原因')
            stats['search_priority'] = search_priority
            
            # 组合结果
            combined_results = self._combine_results(rag_results, web_results)
            # 过滤冗余结果
            final_results = self._filter_redundant_results(combined_results)
        else:
            # 未执行网络搜索或网络搜索无结果，仅使用本地结果
            final_results = rag_results
        
        stats['combined_results_count'] = len(final_results)
        
        # 4. 记录统计信息
        print(f"混合检索完成 - 使用网络搜索: {use_web_search}")
        print(f"结果统计: 本地({stats['rag_results_count']}) + 网络({stats.get('web_results_count', 0)}) = 最终({stats['combined_results_count']})")
        
        return final_results, stats
    
    def update_config(self, config: Dict) -> None:
        """
        更新配置
        
        Args:
            config: 新的配置参数
        """
        self.config.update(config)
        print(f"混合检索器配置已更新: {self.config}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取当前状态
        
        Returns:
            状态信息
        """
        return {
            'config': self.config,
            'rag_retriever_available': self.rag_retriever is not None,
            'web_search_enabled': self.config['enable_web_search']
        }

# 测试代码
if __name__ == "__main__":
    # 模拟RAG检索器
    class MockRAGRetriever:
        def retrieve(self, query: str, **kwargs):
            # 简单模拟检索结果
            if '迪士尼' in query:
                return [
                    {'score': 0.85, 'content': '上海迪士尼乐园位于浦东新区，是中国大陆首座迪士尼主题乐园。', 'title': '上海迪士尼乐园简介'},
                    {'score': 0.72, 'content': '迪士尼乐园门票价格因季节不同而有所变化，建议提前在线购买。', 'title': '迪士尼门票信息'},
                    {'score': 0.65, 'content': '乐园内有多个主题园区，包括米奇大街、奇想花园、探险岛等。', 'title': '乐园园区介绍'}
                ]
            else:
                # 非迪士尼相关查询返回低分结果
                return [
                    {'score': 0.35, 'content': '迪士尼公司有多个业务板块...', 'title': '迪士尼公司概览'}
                ]
    
    # 创建混合检索器实例
    rag_retriever = MockRAGRetriever()
    hybrid_retriever = HybridRetriever(rag_retriever=rag_retriever)
    
    # 测试查询
    test_queries = [
        "上海迪士尼乐园门票价格",
        "2024年迪士尼乐园最新活动",
        "迪士尼乐园创极速光轮的身高限制",
        "迪士尼公司股价走势"
    ]
    
    print("===== 混合检索器测试 =====")
    
    for query in test_queries:
        results, stats = hybrid_retriever.retrieve(query)
        
        print(f"\n查询: {query}")
        print(f"检索统计: {stats}")
        print(f"返回结果数量: {len(results)}")
        
        # 打印前两个结果
        for i, result in enumerate(results[:2]):
            source = result.get('source', 'unknown')
            score = result.get('weighted_score', result.get('score', 0))
            title = result.get('title', 'Untitled')
            content_preview = result.get('content', '')[:100] + '...' if len(result.get('content', '')) > 100 else result.get('content', '')
            
            print(f"\n结果 {i+1} (来源: {source}, 分数: {score:.2f}):")
            print(f"标题: {title}")
            print(f"内容: {content_preview}")
            if 'url' in result and result['url']:
                print(f"URL: {result['url']}")
    
    # 测试禁用网络搜索
    print("\n===== 测试禁用网络搜索 =====")
    hybrid_retriever.update_config({'enable_web_search': False})
    results, stats = hybrid_retriever.retrieve("2024年迪士尼乐园最新活动")
    print(f"检索统计: {stats}")
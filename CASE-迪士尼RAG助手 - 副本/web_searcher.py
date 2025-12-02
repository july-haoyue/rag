import requests
import re
import json
from bs4 import BeautifulSoup
import random
import time
from typing import List, Dict, Optional, Tuple, Any

class SearchResult:
    """搜索结果数据类"""
    def __init__(self, title: str, snippet: str, url: str, source: str = "web"):
        self.title = title
        self.snippet = snippet
        self.url = url
        self.source = source
    
    def to_dict(self) -> Dict:
        return {
            "title": self.title,
            "snippet": self.snippet,
            "url": self.url,
            "source": self.source
        }

class WebSearcher:
    """
    网络搜索引擎模块
    支持多种搜索方式：百度API、通用搜索爬虫、迪士尼官网定向搜索
    """
    def __init__(self, api_key: Optional[str] = None, 
                 search_engine: str = "general",
                 timeout: int = 10,
                 retry_count: int = 3):
        """
        初始化网络搜索引擎
        
        Args:
            api_key: 搜索引擎API密钥（如有）
            search_engine: 搜索引擎类型 ("general", "baidu", "bing", "google")
            timeout: 请求超时时间（秒）
            retry_count: 请求失败重试次数
        """
        self.api_key = api_key
        self.search_engine = search_engine.lower()
        self.timeout = timeout
        self.retry_count = retry_count
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/115.0"
        ]
        
        # 初始化请求会话
        self.session = requests.Session()
        
    def _get_random_user_agent(self) -> str:
        """获取随机User-Agent"""
        return random.choice(self.user_agents)
    
    def _request_with_retry(self, url: str, **kwargs) -> Optional[requests.Response]:
        """带重试的请求方法"""
        if 'headers' not in kwargs:
            kwargs['headers'] = {}
        kwargs['headers']['User-Agent'] = self._get_random_user_agent()
        
        for i in range(self.retry_count):
            try:
                response = self.session.get(url, timeout=self.timeout, **kwargs)
                if response.status_code == 200:
                    return response
                print(f"请求失败: {response.status_code}, 重试 {i+1}/{self.retry_count}")
            except Exception as e:
                print(f"请求异常: {str(e)}, 重试 {i+1}/{self.retry_count}")
            
            # 指数退避
            time.sleep(2 ** i)
        
        return None
    
    def _general_search(self, query: str, top_k: int = 3) -> List[SearchResult]:
        """
        通用网络搜索（基于简单爬虫）
        注意：此方法仅用于演示，实际生产环境建议使用官方API
        """
        results = []
        
        # 使用百度作为默认搜索引擎
        search_url = f"https://www.baidu.com/s?wd={query}&rn={top_k}"
        response = self._request_with_retry(search_url)
        
        if response:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 解析百度搜索结果
            for result in soup.select('.result.c-container')[:top_k]:
                title_elem = result.select_one('h3')
                snippet_elem = result.select_one('.c-abstract')
                url_elem = result.select_one('a')
                
                if title_elem and snippet_elem and url_elem:
                    title = title_elem.get_text().strip()
                    snippet = snippet_elem.get_text().strip()
                    url = url_elem.get('href', '')
                    
                    results.append(SearchResult(title, snippet, url, "baidu"))
        
        return results
    
    def _baidu_api_search(self, query: str, top_k: int = 3) -> List[SearchResult]:
        """
        百度搜索API调用（需要API密钥）
        注意：此方法需要配置百度API密钥才能使用
        """
        results = []
        
        if not self.api_key:
            print("警告: 百度API密钥未配置，回退到通用搜索")
            return self._general_search(query, top_k)
        
        # 这里是百度搜索API的示例调用
        # 实际使用时需要根据百度搜索API文档进行配置
        # search_url = "https://api.baidu.com"
        # ...
        
        # 由于缺少实际API密钥，这里直接回退到通用搜索
        return self._general_search(query, top_k)
    
    def _bing_api_search(self, query: str, top_k: int = 3) -> List[SearchResult]:
        """
        必应搜索API调用
        注意：此方法需要配置必应API密钥才能使用
        """
        # 回退到通用搜索
        return self._general_search(query, top_k)
    
    def _google_api_search(self, query: str, top_k: int = 3) -> List[SearchResult]:
        """
        Google搜索API调用
        注意：此方法需要配置Google API密钥才能使用
        """
        # 回退到通用搜索
        return self._general_search(query, top_k)
    
    def _disney_official_search(self, query: str, top_k: int = 3) -> List[SearchResult]:
        """
        迪士尼官网定向搜索
        专门搜索迪士尼官网和官方渠道的信息
        """
        results = []
        
        # 迪士尼中国官网搜索
        official_sites = [
            f"https://www.shanghaidisneyresort.com/zh-cn/search/?q={query}",
            f"https://www.disney.com.cn/search/?q={query}"
        ]
        
        for site in official_sites:
            response = self._request_with_retry(site)
            if response:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 简单提取文本内容作为摘要
                text_content = soup.get_text()
                # 提取前200个字符作为摘要
                snippet = ' '.join(text_content.split())[:200]
                
                results.append(SearchResult(
                    f"迪士尼官网信息 - {query}",
                    snippet,
                    site,
                    "disney_official"
                ))
                
                if len(results) >= top_k:
                    break
        
        # 如果官网搜索结果不足，补充通用搜索结果
        if len(results) < top_k:
            remaining = top_k - len(results)
            general_results = self._general_search(query + " 迪士尼官网", remaining)
            results.extend(general_results)
        
        return results
    
    def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        执行网络搜索
        
        Args:
            query: 搜索查询
            **kwargs: 其他参数，支持num_results指定结果数量
            
        Returns:
            搜索结果列表（字典格式）
        """
        # 从kwargs中提取参数，支持num_results和top_k两种参数名
        top_k = kwargs.get('num_results', kwargs.get('top_k', 3))
        source = kwargs.get('source', "auto")
        
        print(f"执行网络搜索: {query}, 结果数量: {top_k}")
        
        # 自动选择搜索来源
        if source == "auto":
            # 检测是否为迪士尼相关查询
            disney_keywords = ["迪士尼", "disney", "乐园", "度假区", "门票", "酒店", "巡游", "烟花", "游乐项目"]
            if any(keyword in query.lower() for keyword in disney_keywords):
                source = "disney_official"
            else:
                source = self.search_engine
        
        # 根据来源选择搜索方法
        if source == "baidu":
            results = self._baidu_api_search(query, top_k)
        elif source == "bing":
            results = self._bing_api_search(query, top_k)
        elif source == "google":
            results = self._google_api_search(query, top_k)
        elif source == "disney_official":
            results = self._disney_official_search(query, top_k)
        else:
            results = self._general_search(query, top_k)
        
        # 结果过滤和后处理
        filtered_results = []
        seen_urls = set()
        
        for result in results:
            # 去重
            if result.url in seen_urls:
                continue
            seen_urls.add(result.url)
            
            # 过滤无效结果
            if not result.snippet or len(result.snippet) < 20:
                continue
            
            filtered_results.append(result.to_dict())
            
            if len(filtered_results) >= top_k:
                break
        
        print(f"搜索完成: 找到 {len(filtered_results)} 条有效结果")
        return filtered_results
    
    def format_results(self, results: List[SearchResult]) -> str:
        """
        格式化搜索结果为文本
        
        Args:
            results: 搜索结果列表
        
        Returns:
            格式化后的文本
        """
        formatted = "网络搜索结果:\n\n"
        
        for i, result in enumerate(results, 1):
            formatted += f"[结果 {i}]\n"
            formatted += f"标题: {result.title}\n"
            formatted += f"摘要: {result.snippet}\n"
            formatted += f"来源: {result.url}\n"
            formatted += f"类型: {result.source}\n"
            formatted += "-" * 50 + "\n"
        
        return formatted
    
    def close(self):
        """关闭会话"""
        self.session.close()

if __name__ == "__main__":
    # 简单测试
    searcher = WebSearcher()
    
    # 测试普通搜索
    print("测试普通搜索:")
    results1 = searcher.search("上海迪士尼乐园最新开放时间", top_k=2)
    print(searcher.format_results(results1))
    
    # 测试迪士尼官方搜索
    print("\n测试迪士尼官方搜索:")
    results2 = searcher.search("迪士尼乐园门票价格", top_k=2, source="disney_official")
    print(searcher.format_results(results2))
    
    searcher.close()
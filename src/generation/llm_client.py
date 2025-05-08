"""
LLM客户端模块，用于生成回答
"""

from typing import List, Dict, Any, Optional
import json
import requests
from langchain.docstore.document import Document

from ..utils.config import config

class LLMClient:
    """LLM客户端类，使用Qwen API生成回答"""
    
    def __init__(self, api_base: str = None, api_key: str = None):
        """
        初始化LLM客户端
        
        Args:
            api_base: API基地址
            api_key: API密钥
        """
        self.api_base = api_base if api_base else config.qwen_api_base
        self.api_key = api_key if api_key else config.qwen_api_key
        self.chat_endpoint = f"{self.api_base}/chat/completions"
    
    def generate(self, query: str, documents: List[Document], temperature: float = 0.7) -> str:
        """
        生成回答
        
        Args:
            query: 用户查询
            documents: 相关文档列表
            temperature: 温度参数
            
        Returns:
            生成的回答
        """
        # 构建提示
        system_prompt = "你是一个基于检索增强的问答助手。你的回答必须基于提供的上下文信息，如果上下文中没有相关信息，请直接说明不知道，不要编造答案。请尽量提供详细、准确、有帮助的回答。"
        
        # 格式化上下文信息
        context = self._format_documents(documents)
        
        # 构建用户消息
        user_message = f"请回答以下问题：\n{query}\n\n以下是相关的上下文信息：\n{context}"
        
        # 构建请求
        payload = {
            "model": "default",  # Qwen3-32B
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "temperature": temperature,
            "max_tokens": 2048
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        try:
            response = requests.post(
                self.chat_endpoint,
                headers=headers,
                data=json.dumps(payload)
            )
            
            if response.status_code != 200:
                raise Exception(f"API请求失败，状态码: {response.status_code}, 响应: {response.text}")
            
            result = response.json()
            answer = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            return answer
        except Exception as e:
            print(f"生成回答失败: {str(e)}")
            return f"生成回答时发生错误: {str(e)}"
    
    def generate_with_reflection(self, query: str, documents: List[Document], temperature: float = 0.7) -> str:
        """
        使用反思机制生成回答
        
        Args:
            query: 用户查询
            documents: 相关文档列表
            temperature: 温度参数
            
        Returns:
            生成的回答
        """
        # 构建系统提示
        system_prompt = """你是一个基于检索增强的问答助手。请按照以下步骤处理：
1. 分析用户问题，确定需要回答的关键信息点。
2. 仔细阅读提供的文档片段，确定哪些片段最相关。
3. 评估提供的信息是否足够回答问题。如果不够，明确说明缺少哪些信息。
4. 根据相关文档生成答案，确保答案准确、完整。
5. 不要编造不在文档中的信息。如果文档中没有相关信息，诚实地说明无法回答。"""
        
        # 格式化上下文信息
        context = self._format_documents(documents)
        
        # 构建用户消息
        user_message = f"请回答以下问题：\n{query}\n\n以下是相关的上下文信息：\n{context}"
        
        # 构建请求
        payload = {
            "model": "default",  # Qwen3-32B
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "temperature": temperature,
            "max_tokens": 2048
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        try:
            response = requests.post(
                self.chat_endpoint,
                headers=headers,
                data=json.dumps(payload)
            )
            
            if response.status_code != 200:
                raise Exception(f"API请求失败，状态码: {response.status_code}, 响应: {response.text}")
            
            result = response.json()
            answer = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            return answer
        except Exception as e:
            print(f"生成回答失败: {str(e)}")
            return f"生成回答时发生错误: {str(e)}"
    
    def _format_documents(self, documents: List[Document]) -> str:
        """
        格式化文档
        
        Args:
            documents: 文档列表
            
        Returns:
            格式化后的文档文本
        """
        if not documents:
            return "没有找到相关文档。"
        
        formatted_docs = []
        for i, doc in enumerate(documents):
            source = doc.metadata.get("source", "未知来源")
            formatted_docs.append(f"[文档 {i+1}] (来源: {source})\n{doc.page_content}")
        
        return "\n\n".join(formatted_docs)


if __name__ == "__main__":
    # 测试LLM客户端
    from langchain.docstore.document import Document
    
    # 创建测试文档
    doc1 = Document(
        page_content="这是测试文档1的内容。它包含一些关于测试主题的信息。",
        metadata={"source": "test_doc_1.txt"}
    )
    doc2 = Document(
        page_content="这是测试文档2的内容。它包含另一些关于测试主题的信息。",
        metadata={"source": "test_doc_2.txt"}
    )
    
    # 创建LLM客户端
    llm_client = LLMClient()
    
    # 生成回答
    query = "请总结这些文档的内容"
    answer = llm_client.generate(query, [doc1, doc2])
    
    print(f"查询: {query}")
    print(f"回答: {answer}") 
"""
嵌入模型模块，用于生成文本嵌入向量
"""

from typing import List, Dict, Any
import numpy as np
import requests
import json
from tqdm import tqdm

from .config import config

class EmbeddingModel:
    """嵌入模型类，使用Ollama API生成文本嵌入"""
    
    def __init__(self, api_base: str = None, model: str = None):
        """
        初始化嵌入模型
        
        Args:
            api_base: Ollama API基础URL
            model: 模型名称
        """
        self.api_base = api_base if api_base else config.ollama_api_base
        self.model = model if model else config.ollama_model
        self.embedding_endpoint = f"{self.api_base}/api/embeddings"
    
    def get_embeddings(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        获取文本列表的嵌入向量
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            
        Returns:
            嵌入向量列表
        """
        all_embeddings = []
        
        # 批处理以避免请求过大
        for i in tqdm(range(0, len(texts), batch_size), desc="生成嵌入向量"):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = []
            
            for text in batch_texts:
                try:
                    embedding = self._get_single_embedding(text)
                    batch_embeddings.append(embedding)
                except Exception as e:
                    print(f"获取嵌入向量失败: {str(e)}")
                    # 失败时使用零向量（实际应用中应改进此处理方式）
                    batch_embeddings.append([0.0] * 1024)  # 假设向量维度为1024
            
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def _get_single_embedding(self, text: str) -> List[float]:
        """
        获取单个文本的嵌入向量
        
        Args:
            text: 输入文本
            
        Returns:
            嵌入向量
        """
        payload = {
            "model": self.model,
            "prompt": text
        }
        
        response = requests.post(
            self.embedding_endpoint,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )
        
        if response.status_code != 200:
            raise Exception(f"API请求失败，状态码: {response.status_code}, 响应: {response.text}")
        
        result = response.json()
        embedding = result.get("embedding", [])
        
        if not embedding:
            raise Exception("API返回的嵌入向量为空")
        
        return embedding


if __name__ == "__main__":
    # 测试嵌入模型
    embedding_model = EmbeddingModel()
    texts = ["这是测试文本。", "这是另一个测试文本。"]
    embeddings = embedding_model.get_embeddings(texts)
    
    if embeddings:
        print(f"嵌入向量维度: {len(embeddings[0])}")
        print(f"样本嵌入向量前5个元素: {embeddings[0][:5]}") 
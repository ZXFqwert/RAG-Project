"""
Elasticsearch客户端，用于全文检索和BM25搜索
"""

from typing import List, Dict, Any, Optional
import json
from elasticsearch import Elasticsearch, helpers
from langchain.docstore.document import Document
from tqdm import tqdm

from ..utils.config import config

class ElasticsearchClient:
    """Elasticsearch客户端，用于全文检索"""
    
    def __init__(self, index_name: str = "rag_documents"):
        """
        初始化Elasticsearch客户端
        
        Args:
            index_name: 索引名称
        """
        self.index_name = index_name
        self.es_config = config.get_elasticsearch_config()
        self.client = None
        self.connected = False
    
    def connect(self) -> None:
        """连接到Elasticsearch服务器"""
        try:
            self.client = Elasticsearch(**self.es_config)
            self.connected = self.client.ping()
            if self.connected:
                print(f"已连接到Elasticsearch服务器 {self.es_config['hosts']}")
            else:
                print(f"连接到Elasticsearch服务器失败")
        except Exception as e:
            print(f"连接Elasticsearch服务器异常: {str(e)}")
            self.connected = False
    
    def _ensure_connected(self) -> None:
        """确保已连接到Elasticsearch服务器"""
        if not self.connected or not self.client:
            self.connect()
        
        if not self.connected:
            raise ConnectionError("未连接到Elasticsearch服务器")
    
    def create_index(self) -> None:
        """创建索引"""
        self._ensure_connected()
        
        # 检查索引是否存在
        if self.client.indices.exists(index=self.index_name):
            print(f"索引 {self.index_name} 已存在")
            return
        
        # 索引配置
        mapping = {
            "mappings": {
                "properties": {
                    "content": {
                        "type": "text",
                        "analyzer": "ik_smart"
                    },
                    "metadata": {
                        "type": "object",
                        "enabled": True
                    }
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "default": {
                            "type": "ik_smart"
                        }
                    }
                }
            }
        }
        
        # 创建索引
        self.client.indices.create(index=self.index_name, body=mapping)
        print(f"已创建索引 {self.index_name}")
    
    def index_documents(self, documents: List[Document]) -> None:
        """
        索引文档
        
        Args:
            documents: 文档对象列表
        """
        self._ensure_connected()
        
        if not documents:
            print("文档列表为空，无需索引")
            return
        
        # 确保索引存在
        if not self.client.indices.exists(index=self.index_name):
            self.create_index()
        
        # 准备批量索引数据
        actions = []
        for i, doc in enumerate(documents):
            action = {
                "_index": self.index_name,
                "_id": f"doc_{i}",
                "_source": {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
            }
            actions.append(action)
        
        # 批量索引
        print(f"正在索引 {len(documents)} 个文档...")
        success, failed = helpers.bulk(
            self.client,
            actions,
            stats_only=True,
            raise_on_error=False
        )
        
        print(f"成功索引 {success} 个文档，失败 {failed} 个文档")
    
    def search(self, query: str, top_k: int = None) -> List[Document]:
        """
        搜索文档
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            相关文档列表
        """
        self._ensure_connected()
        
        # 检查索引是否存在
        if not self.client.indices.exists(index=self.index_name):
            print(f"索引 {self.index_name} 不存在")
            return []
        
        top_k = top_k if top_k is not None else config.top_k_retrieval
        
        # 构建查询
        query_body = {
            "query": {
                "match": {
                    "content": query
                }
            },
            "size": top_k
        }
        
        try:
            # 执行查询
            response = self.client.search(
                index=self.index_name,
                body=query_body
            )
            
            # 解析结果
            hits = response["hits"]["hits"]
            documents = []
            
            for hit in hits:
                source = hit["_source"]
                score = hit["_score"]
                
                doc = Document(
                    page_content=source["content"],
                    metadata={**source["metadata"], "score": score}
                )
                documents.append(doc)
            
            return documents
        except Exception as e:
            print(f"搜索文档失败: {str(e)}")
            return []
    
    def delete_index(self) -> None:
        """删除索引"""
        self._ensure_connected()
        
        if self.client.indices.exists(index=self.index_name):
            self.client.indices.delete(index=self.index_name)
            print(f"已删除索引 {self.index_name}")
        else:
            print(f"索引 {self.index_name} 不存在")
    
    def get_document_count(self) -> int:
        """
        获取文档数量
        
        Returns:
            文档数量
        """
        self._ensure_connected()
        
        if not self.client.indices.exists(index=self.index_name):
            return 0
        
        response = self.client.count(index=self.index_name)
        return response["count"]


if __name__ == "__main__":
    # 测试Elasticsearch客户端
    from ..document_processor.document_loader import DocumentLoader
    from ..document_processor.text_splitter import TextSplitter
    
    # 加载文档
    loader = DocumentLoader()
    documents = loader.load_documents("../../data/documents")
    
    # 分割文档
    splitter = TextSplitter()
    chunks = splitter.split_documents(documents)
    
    # 创建Elasticsearch客户端
    client = ElasticsearchClient()
    client.connect()
    
    # 索引文档
    client.index_documents(chunks[:10])  # 仅索引前10个文档用于测试
    
    # 搜索文档
    query = "测试查询"
    results = client.search(query)
    
    print(f"搜索结果数量: {len(results)}")
    if results:
        print(f"最相关文档内容前100个字符: {results[0].page_content[:100]}") 
"""
Milvus数据库客户端，用于向量存储和检索
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from langchain.docstore.document import Document
from tqdm import tqdm

from ..utils.config import config
from ..utils.embedding_model import EmbeddingModel

class MilvusClient:
    """Milvus向量数据库客户端"""
    
    def __init__(
        self,
        collection_name: str = None,
        embedding_model: Optional[EmbeddingModel] = None
    ):
        """
        初始化Milvus客户端
        
        Args:
            collection_name: 集合名称
            embedding_model: 嵌入模型实例
        """
        self.collection_name = collection_name if collection_name else config.milvus_collection_name
        self.embedding_model = embedding_model if embedding_model else EmbeddingModel()
        self.collection = None
        self.connected = False
        
        # 连接配置
        self.host = config.milvus_host
        self.port = config.milvus_port
        self.user = config.milvus_user
        self.password = config.milvus_password
    
    def connect(self) -> None:
        """连接到Milvus服务器"""
        try:
            conn_params = {
                "host": self.host,
                "port": self.port
            }
            
            if self.user and self.password:
                conn_params["user"] = self.user
                conn_params["password"] = self.password
            
            connections.connect("default", **conn_params)
            self.connected = True
            print(f"已连接到Milvus服务器 {self.host}:{self.port}")
        except Exception as e:
            print(f"连接Milvus服务器失败: {str(e)}")
            self.connected = False
    
    def _ensure_connected(self) -> None:
        """确保已连接到Milvus服务器"""
        if not self.connected:
            self.connect()
    
    def create_collection(self, dim: int = 1024) -> None:
        """
        创建向量集合
        
        Args:
            dim: 向量维度
        """
        self._ensure_connected()
        
        if utility.has_collection(self.collection_name):
            print(f"集合 {self.collection_name} 已存在")
            self.collection = Collection(self.collection_name)
            return
        
        # 定义字段
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="metadata", dtype=DataType.JSON)
        ]
        
        # 创建集合
        schema = CollectionSchema(fields=fields, description="RAG向量存储")
        self.collection = Collection(name=self.collection_name, schema=schema)
        
        # 创建索引
        index_params = {
            "metric_type": "IP",  # 内积，等价于归一化向量的余弦相似度
            "index_type": "HNSW",
            "params": {"M": 8, "efConstruction": 64}
        }
        self.collection.create_index(field_name="embedding", index_params=index_params)
        
        print(f"已创建集合 {self.collection_name}，向量维度: {dim}")
    
    def insert_documents(self, documents: List[Document]) -> None:
        """
        插入文档
        
        Args:
            documents: 文档对象列表
        """
        self._ensure_connected()
        
        if not documents:
            print("文档列表为空，无需插入")
            return
        
        # 确保集合存在
        if not self.collection:
            # 获取测试向量以确定维度
            test_embedding = self.embedding_model.get_embeddings(["测试维度"])[0]
            dim = len(test_embedding)
            self.create_collection(dim=dim)
        
        # 加载集合
        self.collection.load()
        
        # 准备数据
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # 生成嵌入向量
        print("为文档生成嵌入向量...")
        embeddings = self.embedding_model.get_embeddings(texts)
        
        # 插入数据
        data = [
            texts,
            embeddings,
            metadatas
        ]
        
        try:
            self.collection.insert(data)
            print(f"已插入 {len(documents)} 个文档到Milvus")
        except Exception as e:
            print(f"插入文档失败: {str(e)}")
        finally:
            # 释放集合
            self.collection.release()
    
    def search(self, query: str, top_k: int = None) -> List[Document]:
        """
        搜索相似文档
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            相关文档列表
        """
        self._ensure_connected()
        
        if not self.collection:
            if not utility.has_collection(self.collection_name):
                raise ValueError(f"集合 {self.collection_name} 不存在")
            self.collection = Collection(self.collection_name)
        
        top_k = top_k if top_k is not None else config.top_k_retrieval
        
        # 加载集合
        self.collection.load()
        
        try:
            # 生成查询向量
            query_embedding = self.embedding_model.get_embeddings([query])[0]
            
            # 执行搜索
            search_params = {"metric_type": "IP", "params": {"ef": 64}}
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["text", "metadata"]
            )
            
            # 转换结果为Document对象
            documents = []
            for hits in results:
                for hit in hits:
                    doc = Document(
                        page_content=hit.entity.get("text"),
                        metadata=hit.entity.get("metadata", {})
                    )
                    documents.append(doc)
            
            return documents
        except Exception as e:
            print(f"搜索文档失败: {str(e)}")
            return []
        finally:
            # 释放集合
            self.collection.release()
    
    def drop_collection(self) -> None:
        """删除集合"""
        self._ensure_connected()
        
        if utility.has_collection(self.collection_name):
            Collection(self.collection_name).drop()
            self.collection = None
            print(f"已删除集合 {self.collection_name}")
        else:
            print(f"集合 {self.collection_name} 不存在")
    
    def get_document_count(self) -> int:
        """
        获取文档数量
        
        Returns:
            文档数量
        """
        self._ensure_connected()
        
        if not utility.has_collection(self.collection_name):
            return 0
        
        collection = Collection(self.collection_name)
        # 在获取数量前先加载集合
        collection.load()
        count = collection.num_entities
        # 使用完后释放资源
        collection.release()
        return count


if __name__ == "__main__":
    # 测试Milvus客户端
    from ..document_processor.document_loader import DocumentLoader
    from ..document_processor.text_splitter import TextSplitter
    
    # 加载文档
    loader = DocumentLoader()
    documents = loader.load_documents("../../data/documents")
    
    # 分割文档
    splitter = TextSplitter()
    chunks = splitter.split_documents(documents)
    
    # 创建Milvus客户端
    client = MilvusClient()
    client.connect()
    
    # 插入文档
    client.insert_documents(chunks[:10])  # 仅插入前10个文档用于测试
    
    # 搜索文档
    query = "测试查询"
    results = client.search(query)
    
    print(f"搜索结果数量: {len(results)}")
    if results:
        print(f"最相关文档内容前100个字符: {results[0].page_content[:100]}") 
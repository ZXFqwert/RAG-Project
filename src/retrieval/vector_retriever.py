"""
向量检索器模块，使用语义嵌入进行文本检索
"""

from typing import List, Dict, Any, Tuple, Optional
import os
import pickle
import numpy as np
from tqdm import tqdm
import faiss
from langchain.docstore.document import Document

from ..utils.config import config
from ..utils.embedding_model import EmbeddingModel

class VectorRetriever:
    """向量检索器类，使用语义嵌入进行文本检索"""
    
    def __init__(self, embedding_model: Optional[EmbeddingModel] = None):
        """
        初始化向量检索器
        
        Args:
            embedding_model: 嵌入模型实例
        """
        self.embedding_model = embedding_model if embedding_model else EmbeddingModel()
        self.documents = []
        self.embeddings = None
        self.index = None
    
    def fit(self, documents: List[Document]) -> None:
        """
        使用文档列表构建向量索引
        
        Args:
            documents: 文档对象列表
        """
        self.documents = documents
        
        # 获取文档内容
        texts = [doc.page_content for doc in documents]
        
        # 生成文档嵌入向量
        print("为文档生成嵌入向量...")
        embeddings = self.embedding_model.get_embeddings(texts)
        
        # 构建FAISS索引
        self._build_index(embeddings)
        
        print(f"向量索引已构建，共包含 {len(self.documents)} 个文档")
    
    def _build_index(self, embeddings: List[List[float]]) -> None:
        """
        构建FAISS向量索引
        
        Args:
            embeddings: 嵌入向量列表
        """
        if not embeddings:
            raise ValueError("嵌入向量列表为空")
        
        # 转换为numpy数组
        embeddings_np = np.array(embeddings).astype(np.float32)
        self.embeddings = embeddings_np
        
        # 获取向量维度
        dimension = embeddings_np.shape[1]
        
        # 创建FAISS索引
        self.index = faiss.IndexFlatIP(dimension)  # 使用内积，等价于余弦相似度
        self.index.add(embeddings_np)
        
        print(f"FAISS索引已创建，向量维度: {dimension}")
    
    def retrieve(self, query: str, top_k: int = None) -> List[Tuple[Document, float]]:
        """
        检索与查询相关的文档
        
        Args:
            query: 查询文本
            top_k: 返回的文档数量
            
        Returns:
            文档和相关性得分的元组列表
        """
        if self.index is None:
            raise ValueError("向量索引未构建，请先调用fit方法")
        
        top_k = top_k if top_k is not None else config.top_k_retrieval
        
        # 生成查询嵌入向量
        query_embedding = self.embedding_model.get_embeddings([query])[0]
        query_np = np.array([query_embedding]).astype(np.float32)
        
        # 执行查询
        scores, indices = self.index.search(query_np, min(top_k, len(self.documents)))
        
        # 返回文档和得分
        results = [(self.documents[int(idx)], float(score)) for score, idx in zip(scores[0], indices[0])]
        
        return results
    
    def save(self, file_path: str) -> None:
        """
        保存向量索引
        
        Args:
            file_path: 保存路径
        """
        if self.index is None:
            raise ValueError("向量索引未构建，无法保存")
        
        # 创建目录
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # 保存索引和文档
        data = {
            "documents": self.documents,
            "embeddings": self.embeddings
        }
        
        # 保存FAISS索引
        faiss.write_index(self.index, f"{file_path}.faiss")
        
        # 保存文档和嵌入信息
        with open(f"{file_path}.pkl", 'wb') as f:
            pickle.dump(data, f)
        
        print(f"向量索引已保存到: {file_path}")
    
    def load(self, file_path: str) -> None:
        """
        加载向量索引
        
        Args:
            file_path: 索引文件路径
        """
        if not os.path.exists(f"{file_path}.faiss") or not os.path.exists(f"{file_path}.pkl"):
            raise FileNotFoundError(f"找不到索引文件: {file_path}")
        
        # 加载文档和嵌入信息
        with open(f"{file_path}.pkl", 'rb') as f:
            data = pickle.load(f)
        
        self.documents = data["documents"]
        self.embeddings = data["embeddings"]
        
        # 加载FAISS索引
        self.index = faiss.read_index(f"{file_path}.faiss")
        
        print(f"向量索引已加载，共包含 {len(self.documents)} 个文档")


if __name__ == "__main__":
    # 测试向量检索器
    from ..document_processor.document_loader import DocumentLoader
    from ..document_processor.text_splitter import TextSplitter
    
    # 加载文档
    loader = DocumentLoader()
    documents = loader.load_documents("../../data/documents")
    
    # 分割文档
    splitter = TextSplitter()
    chunks = splitter.split_documents(documents)
    
    # 创建向量检索器
    retriever = VectorRetriever()
    retriever.fit(chunks)
    
    # 测试检索
    query = "测试查询"
    results = retriever.retrieve(query)
    
    print(f"检索结果数量: {len(results)}")
    if results:
        print(f"最相关文档得分: {results[0][1]}")
        print(f"最相关文档内容前100个字符: {results[0][0].page_content[:100]}") 
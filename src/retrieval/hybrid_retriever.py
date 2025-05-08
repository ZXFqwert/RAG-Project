"""
混合检索器模块，结合BM25和向量检索进行混合检索
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from langchain.docstore.document import Document

from ..utils.config import config
from .bm25_retriever import BM25Retriever
from .vector_retriever import VectorRetriever

class HybridRetriever:
    """混合检索器类，结合BM25和向量检索"""
    
    def __init__(
        self, 
        bm25_retriever: Optional[BM25Retriever] = None,
        vector_retriever: Optional[VectorRetriever] = None,
        alpha: float = None
    ):
        """
        初始化混合检索器
        
        Args:
            bm25_retriever: BM25检索器实例
            vector_retriever: 向量检索器实例
            alpha: BM25和向量检索的权重平衡参数，越大则BM25权重越高
        """
        self.bm25_retriever = bm25_retriever if bm25_retriever else BM25Retriever()
        self.vector_retriever = vector_retriever if vector_retriever else VectorRetriever()
        self.alpha = alpha if alpha is not None else config.hybrid_alpha
    
    def fit(self, documents: List[Document]) -> None:
        """
        使用文档列表训练检索器
        
        Args:
            documents: 文档对象列表
        """
        print("训练BM25检索器...")
        self.bm25_retriever.fit(documents)
        
        print("训练向量检索器...")
        self.vector_retriever.fit(documents)
        
        print("混合检索器训练完成")
    
    def retrieve(self, query: str, top_k: int = None) -> List[Document]:
        """
        使用混合策略检索文档
        
        Args:
            query: 查询文本
            top_k: 返回的文档数量
            
        Returns:
            相关文档列表
        """
        top_k = top_k if top_k is not None else config.top_k_retrieval
        
        # 使用两种检索器分别检索
        bm25_results = self.bm25_retriever.retrieve(query, top_k=top_k * 2)
        vector_results = self.vector_retriever.retrieve(query, top_k=top_k * 2)
        
        # 标准化得分
        bm25_scores = self._normalize_scores([score for _, score in bm25_results])
        vector_scores = self._normalize_scores([score for _, score in vector_results])
        
        # 创建文档到得分的映射
        doc_scores: Dict[str, Dict[str, Any]] = {}
        
        # 添加BM25得分
        for i, (doc, score) in enumerate(bm25_results):
            doc_id = self._get_doc_id(doc)
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {
                    "document": doc,
                    "bm25_score": bm25_scores[i],
                    "vector_score": 0.0
                }
            else:
                doc_scores[doc_id]["bm25_score"] = bm25_scores[i]
        
        # 添加向量检索得分
        for i, (doc, score) in enumerate(vector_results):
            doc_id = self._get_doc_id(doc)
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {
                    "document": doc,
                    "bm25_score": 0.0,
                    "vector_score": vector_scores[i]
                }
            else:
                doc_scores[doc_id]["vector_score"] = vector_scores[i]
        
        # 计算混合得分
        for doc_id, data in doc_scores.items():
            data["combined_score"] = self.alpha * data["bm25_score"] + (1 - self.alpha) * data["vector_score"]
        
        # 按混合得分排序
        sorted_results = sorted(
            doc_scores.values(),
            key=lambda x: x["combined_score"],
            reverse=True
        )
        
        # 返回前top_k个文档
        return [data["document"] for data in sorted_results[:top_k]]
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """
        归一化得分
        
        Args:
            scores: 得分列表
            
        Returns:
            归一化后的得分列表
        """
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        return [(score - min_score) / (max_score - min_score) for score in scores]
    
    def _get_doc_id(self, doc: Document) -> str:
        """
        获取文档唯一标识
        
        Args:
            doc: 文档对象
            
        Returns:
            文档唯一标识
        """
        # 使用文档内容的前100个字符和来源作为唯一标识
        content_hash = hash(doc.page_content[:100])
        source = doc.metadata.get("source", "")
        return f"{source}_{content_hash}"
    
    def save(self, bm25_path: str, vector_path: str) -> None:
        """
        保存检索器模型
        
        Args:
            bm25_path: BM25模型保存路径
            vector_path: 向量模型保存路径
        """
        self.bm25_retriever.save(bm25_path)
        self.vector_retriever.save(vector_path)
        print("混合检索器模型已保存")
    
    def load(self, bm25_path: str, vector_path: str) -> None:
        """
        加载检索器模型
        
        Args:
            bm25_path: BM25模型文件路径
            vector_path: 向量模型文件路径
        """
        self.bm25_retriever.load(bm25_path)
        self.vector_retriever.load(vector_path)
        print("混合检索器模型已加载")


if __name__ == "__main__":
    # 测试混合检索器
    from ..document_processor.document_loader import DocumentLoader
    from ..document_processor.text_splitter import TextSplitter
    
    # 加载文档
    loader = DocumentLoader()
    documents = loader.load_documents("../../data/documents")
    
    # 分割文档
    splitter = TextSplitter()
    chunks = splitter.split_documents(documents)
    
    # 创建混合检索器
    retriever = HybridRetriever()
    retriever.fit(chunks)
    
    # 测试检索
    query = "测试查询"
    results = retriever.retrieve(query)
    
    print(f"检索结果数量: {len(results)}")
    if results:
        print(f"最相关文档内容前100个字符: {results[0].page_content[:100]}") 
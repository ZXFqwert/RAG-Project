"""
BM25检索器模块，使用BM25算法进行文本检索
"""

from typing import List, Dict, Any, Tuple
import pickle
import os
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import nltk
import re
from langchain.docstore.document import Document

from ..utils.config import config

class BM25Retriever:
    """BM25检索器类，使用BM25算法进行文本检索"""
    
    def __init__(self, tokenizer=None):
        """
        初始化BM25检索器
        
        Args:
            tokenizer: 分词器，默认使用简单的空格分词
        """
        self.tokenizer = tokenizer if tokenizer else self._default_tokenizer
        self.bm25 = None
        self.documents = []
        self.tokenized_corpus = []
    
    def _default_tokenizer(self, text: str) -> List[str]:
        """
        默认分词器
        
        Args:
            text: 输入文本
            
        Returns:
            分词后的token列表
        """
        # 移除标点符号
        text = re.sub(r'[^\w\s]', '', text.lower())
        # 简单空格分词
        return text.split()
    
    def fit(self, documents: List[Document]) -> None:
        """
        使用文档列表训练BM25模型
        
        Args:
            documents: 文档对象列表
        """
        self.documents = documents
        
        # 分词处理
        texts = [doc.page_content for doc in documents]
        self.tokenized_corpus = [self.tokenizer(text) for text in tqdm(texts, desc="BM25分词处理")]
        
        # 训练BM25模型
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print(f"BM25模型已训练，共包含 {len(self.documents)} 个文档")
    
    def retrieve(self, query: str, top_k: int = None) -> List[Tuple[Document, float]]:
        """
        检索与查询相关的文档
        
        Args:
            query: 查询文本
            top_k: 返回的文档数量
            
        Returns:
            文档和相关性得分的元组列表
        """
        if self.bm25 is None:
            raise ValueError("BM25模型未训练，请先调用fit方法")
        
        top_k = top_k if top_k is not None else config.top_k_retrieval
        
        # 对查询进行分词
        tokenized_query = self.tokenizer(query)
        
        # 获取得分
        scores = self.bm25.get_scores(tokenized_query)
        
        # 获取得分最高的文档索引
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        # 返回文档和得分
        results = [(self.documents[i], scores[i]) for i in top_indices]
        
        return results
    
    def save(self, file_path: str) -> None:
        """
        保存BM25模型
        
        Args:
            file_path: 保存路径
        """
        if self.bm25 is None:
            raise ValueError("BM25模型未训练，无法保存")
        
        data = {
            "bm25": self.bm25,
            "documents": self.documents,
            "tokenized_corpus": self.tokenized_corpus
        }
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"BM25模型已保存到: {file_path}")
    
    def load(self, file_path: str) -> None:
        """
        加载BM25模型
        
        Args:
            file_path: 模型文件路径
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到模型文件: {file_path}")
        
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        self.bm25 = data["bm25"]
        self.documents = data["documents"]
        self.tokenized_corpus = data["tokenized_corpus"]
        
        print(f"BM25模型已加载，共包含 {len(self.documents)} 个文档")


if __name__ == "__main__":
    # 测试BM25检索器
    from ..document_processor.document_loader import DocumentLoader
    from ..document_processor.text_splitter import TextSplitter
    
    # 加载文档
    loader = DocumentLoader()
    documents = loader.load_documents("../../data/documents")
    
    # 分割文档
    splitter = TextSplitter()
    chunks = splitter.split_documents(documents)
    
    # 创建BM25检索器
    retriever = BM25Retriever()
    retriever.fit(chunks)
    
    # 测试检索
    query = "测试查询"
    results = retriever.retrieve(query)
    
    print(f"检索结果数量: {len(results)}")
    if results:
        print(f"最相关文档得分: {results[0][1]}")
        print(f"最相关文档内容前100个字符: {results[0][0].page_content[:100]}") 
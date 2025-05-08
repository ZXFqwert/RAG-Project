"""
文本分块器模块，将文档切分为合适大小的文本块
"""

from typing import List, Dict, Any
import nltk
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from tqdm import tqdm

from ..utils.config import config

class TextSplitter:
    """文本分块器，将长文本分割成较小的块"""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """
        初始化文本分块器
        
        Args:
            chunk_size: 块大小，默认使用配置值
            chunk_overlap: 块重叠大小，默认使用配置值
        """
        self.chunk_size = chunk_size if chunk_size is not None else config.chunk_size
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else config.chunk_overlap
        
        # 创建分块器
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""]
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        分割文档列表
        
        Args:
            documents: 文档对象列表
            
        Returns:
            分割后的文档对象列表
        """
        split_docs = []
        
        for doc in tqdm(documents, desc="分割文档"):
            try:
                # 保留原文档的元数据
                doc_splits = self.splitter.split_documents([doc])
                split_docs.extend(doc_splits)
            except Exception as e:
                print(f"分割文档失败: {str(e)}")
        
        print(f"将 {len(documents)} 个文档分割为 {len(split_docs)} 个文本块")
        return split_docs
    
    def split_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Document]:
        """
        分割单个文本
        
        Args:
            text: 要分割的文本
            metadata: 元数据
            
        Returns:
            分割后的文档对象列表
        """
        if metadata is None:
            metadata = {}
        
        try:
            texts = self.splitter.split_text(text)
            return [Document(page_content=t, metadata=metadata) for t in texts]
        except Exception as e:
            print(f"分割文本失败: {str(e)}")
            return [Document(page_content=text, metadata=metadata)]


if __name__ == "__main__":
    # 测试文档分割
    from document_loader import DocumentLoader
    
    loader = DocumentLoader()
    documents = loader.load_documents("../../data/documents")
    
    splitter = TextSplitter()
    chunks = splitter.split_documents(documents)
    
    print(f"分割后的文档数量: {len(chunks)}")
    if chunks:
        print(f"第一个块的大小: {len(chunks[0].page_content)} 字符") 
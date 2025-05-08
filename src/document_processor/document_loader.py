"""
文档加载器模块，支持加载不同格式的文档
"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import docx2txt
import pypdf
from langchain.docstore.document import Document
import nltk
from tqdm import tqdm

# 下载nltk资源，用于文本分块
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class DocumentLoader:
    """文档加载器，支持多种格式文档的加载"""
    
    def __init__(self):
        """初始化文档加载器"""
        pass
    
    def load_document(self, file_path: str) -> Optional[str]:
        """
        加载单个文档
        
        Args:
            file_path: 文档路径
            
        Returns:
            文档内容，如果不支持则返回None
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        
        if not file_path.exists():
            print(f"文件不存在: {file_path}")
            return None
        
        try:
            if suffix == '.txt':
                return self._load_text(file_path)
            elif suffix == '.pdf':
                return self._load_pdf(file_path)
            elif suffix in ['.docx', '.doc']:
                return self._load_docx(file_path)
            else:
                print(f"不支持的文件格式: {suffix}")
                return None
        except Exception as e:
            print(f"加载文件 {file_path} 失败: {str(e)}")
            return None
    
    def load_documents(self, dir_path: str) -> List[Document]:
        """
        加载目录中的所有文档
        
        Args:
            dir_path: 文档目录路径
            
        Returns:
            Document对象列表
        """
        dir_path = Path(dir_path)
        documents = []
        
        if not dir_path.exists() or not dir_path.is_dir():
            print(f"目录不存在: {dir_path}")
            return documents
        
        all_files = list(dir_path.glob("**/*"))
        supported_files = [f for f in all_files if f.is_file() and 
                         f.suffix.lower() in ['.txt', '.pdf', '.docx', '.doc']]
        
        for file_path in tqdm(supported_files, desc="加载文档"):
            content = self.load_document(file_path)
            if content:
                metadata = {
                    "source": str(file_path),
                    "filename": file_path.name,
                    "filetype": file_path.suffix.lower()[1:],  # 去掉前导点号
                }
                documents.append(Document(page_content=content, metadata=metadata))
        
        return documents
    
    def _load_text(self, file_path: Path) -> str:
        """加载文本文件"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    
    def _load_pdf(self, file_path: Path) -> str:
        """加载PDF文件"""
        text = ""
        with open(file_path, 'rb') as f:
            pdf = pypdf.PdfReader(f)
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text
    
    def _load_docx(self, file_path: Path) -> str:
        """加载DOCX文件"""
        return docx2txt.process(file_path)


if __name__ == "__main__":
    # 测试文档加载
    loader = DocumentLoader()
    documents = loader.load_documents("./data/documents")
    print(f"加载了 {len(documents)} 个文档") 
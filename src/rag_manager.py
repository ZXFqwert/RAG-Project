"""
RAG系统管理器，整合文档处理、检索和生成
"""

import os
from typing import List, Dict, Any, Optional, Union
from langchain.docstore.document import Document
from pathlib import Path
import time

from .utils.config import config
from .document_processor.document_loader import DocumentLoader
from .document_processor.text_splitter import TextSplitter
from .retrieval.hybrid_retriever import HybridRetriever
from .retrieval.bm25_retriever import BM25Retriever
from .retrieval.vector_retriever import VectorRetriever
from .database.milvus_client import MilvusClient
from .database.elasticsearch_client import ElasticsearchClient
from .generation.llm_client import LLMClient

class RAGManager:
    """RAG系统管理器，提供完整的检索增强生成功能"""
    
    def __init__(self, use_db: bool = True):
        """
        初始化RAG管理器
        
        Args:
            use_db: 是否使用数据库而不是内存索引
        """
        # 创建组件实例
        self.document_loader = DocumentLoader()
        self.text_splitter = TextSplitter()
        self.llm_client = LLMClient()
        
        self.use_db = use_db
        self.use_elasticsearch = config.use_elasticsearch
        self.use_milvus = config.use_milvus
        
        # 数据库客户端
        self.es_client = None
        self.milvus_client = None
        
        # 检索器
        self.retriever = None
        self.bm25_retriever = None
        self.vector_retriever = None
        
        # 数据库文件路径
        self.bm25_index_path = os.path.join(config.embeddings_dir, "bm25_index")
        self.vector_index_path = os.path.join(config.embeddings_dir, "vector_index")
        
        # 初始化检索器
        self._init_retrievers()
    
    def _init_retrievers(self) -> None:
        """初始化检索器"""
        if self.use_db:
            # 使用数据库
            if self.use_elasticsearch:
                self.es_client = ElasticsearchClient()
            
            if self.use_milvus:
                self.milvus_client = MilvusClient()
            
            # 如果都没启用，使用内存检索器作为后备
            if not self.use_elasticsearch and not self.use_milvus:
                self.bm25_retriever = BM25Retriever()
                self.vector_retriever = VectorRetriever()
                self.retriever = HybridRetriever(
                    bm25_retriever=self.bm25_retriever,
                    vector_retriever=self.vector_retriever
                )
        else:
            # 使用内存检索器
            self.bm25_retriever = BM25Retriever()
            self.vector_retriever = VectorRetriever()
            self.retriever = HybridRetriever(
                bm25_retriever=self.bm25_retriever,
                vector_retriever=self.vector_retriever
            )
    
    def create_knowledge_base(self, documents_dir: str = None) -> None:
        """
        创建知识库
        
        Args:
            documents_dir: 文档目录，默认使用配置的documents_dir
        """
        documents_dir = documents_dir if documents_dir else config.documents_dir
        
        print(f"从目录 {documents_dir} 加载文档...")
        documents = self.document_loader.load_documents(documents_dir)
        
        if not documents:
            print("没有找到文档，知识库创建失败")
            return
        
        print(f"加载了 {len(documents)} 个文档，正在处理...")
        self.create_knowledge_base_from_documents(documents)
    
    def create_knowledge_base_from_documents(self, documents: List[Document]) -> None:
        """
        从文档列表创建知识库
        
        Args:
            documents: 文档对象列表
        """
        # 分割文档
        print("将文档分割为块...")
        chunks = self.text_splitter.split_documents(documents)
        
        if not chunks:
            print("文档分割失败")
            return
        
        # 创建索引
        if self.use_db:
            self._create_db_index(chunks)
        else:
            self._create_memory_index(chunks)
    
    def _create_db_index(self, chunks: List[Document]) -> None:
        """
        创建数据库索引
        
        Args:
            chunks: 文档块列表
        """
        # 使用Elasticsearch
        if self.use_elasticsearch:
            print("创建Elasticsearch索引...")
            if not self.es_client:
                self.es_client = ElasticsearchClient()
            
            self.es_client.connect()
            self.es_client.create_index()
            self.es_client.index_documents(chunks)
        
        # 使用Milvus
        if self.use_milvus:
            print("创建Milvus索引...")
            if not self.milvus_client:
                self.milvus_client = MilvusClient()
            
            self.milvus_client.connect()
            self.milvus_client.create_collection()
            self.milvus_client.insert_documents(chunks)
    
    def _create_memory_index(self, chunks: List[Document]) -> None:
        """
        创建内存索引
        
        Args:
            chunks: 文档块列表
        """
        print("创建内存索引...")
        
        # 训练混合检索器
        self.retriever.fit(chunks)
        
        # 保存索引
        os.makedirs(os.path.dirname(self.bm25_index_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.vector_index_path), exist_ok=True)
        
        print("保存索引...")
        self.retriever.save(self.bm25_index_path, self.vector_index_path)
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        添加文档到知识库（增量更新）
        
        Args:
            documents: 文档对象列表
        """
        # 分割文档
        print("将新文档分割为块...")
        chunks = self.text_splitter.split_documents(documents)
        
        if not chunks:
            print("文档分割失败")
            return
        
        # 添加到索引
        if self.use_db:
            self._add_to_db_index(chunks)
        else:
            self._add_to_memory_index(chunks)
    
    def _add_to_db_index(self, chunks: List[Document]) -> None:
        """
        增量添加到数据库索引
        
        Args:
            chunks: 文档块列表
        """
        # 使用Elasticsearch
        if self.use_elasticsearch and self.es_client:
            print("添加到Elasticsearch索引...")
            self.es_client.index_documents(chunks)
        
        # 使用Milvus
        if self.use_milvus and self.milvus_client:
            print("添加到Milvus索引...")
            self.milvus_client.insert_documents(chunks)
    
    def _add_to_memory_index(self, chunks: List[Document]) -> None:
        """
        增量添加到内存索引（需要重建索引）
        
        Args:
            chunks: 文档块列表
        """
        print("加载现有索引...")
        
        # 尝试加载现有索引
        try:
            # 检查索引文件是否存在
            if (os.path.exists(f"{self.bm25_index_path}.pkl") and 
                os.path.exists(f"{self.vector_index_path}.faiss") and 
                os.path.exists(f"{self.vector_index_path}.pkl")):
                
                self.retriever.load(self.bm25_index_path, self.vector_index_path)
                
                # 获取现有文档
                existing_docs = self.bm25_retriever.documents
                
                # 合并文档
                all_docs = existing_docs + chunks
                
                # 重建索引
                print("重建索引...")
                self.retriever.fit(all_docs)
                
                # 保存索引
                print("保存更新后的索引...")
                self.retriever.save(self.bm25_index_path, self.vector_index_path)
            else:
                print("找不到现有索引，创建新索引...")
                self.retriever.fit(chunks)
                self.retriever.save(self.bm25_index_path, self.vector_index_path)
        except Exception as e:
            print(f"加载索引失败: {str(e)}")
            print("创建新索引...")
            self.retriever.fit(chunks)
            self.retriever.save(self.bm25_index_path, self.vector_index_path)
    
    def load_knowledge_base(self) -> bool:
        """
        加载知识库
        
        Returns:
            加载是否成功
        """
        if self.use_db:
            return self._load_db_index()
        else:
            return self._load_memory_index()
    
    def _load_db_index(self) -> bool:
        """
        加载数据库索引
        
        Returns:
            加载是否成功
        """
        success = True
        
        # 连接到Elasticsearch
        if self.use_elasticsearch:
            if not self.es_client:
                self.es_client = ElasticsearchClient()
            
            try:
                self.es_client.connect()
                doc_count = self.es_client.get_document_count()
                print(f"已连接到Elasticsearch，文档数量: {doc_count}")
            except Exception as e:
                print(f"连接Elasticsearch失败: {str(e)}")
                success = False
        
        # 连接到Milvus
        if self.use_milvus:
            if not self.milvus_client:
                self.milvus_client = MilvusClient()
            
            try:
                self.milvus_client.connect()
                doc_count = self.milvus_client.get_document_count()
                print(f"已连接到Milvus，文档数量: {doc_count}")
            except Exception as e:
                print(f"连接Milvus失败: {str(e)}")
                success = False
        
        return success
    
    def _load_memory_index(self) -> bool:
        """
        加载内存索引
        
        Returns:
            加载是否成功
        """
        try:
            # 检查索引文件是否存在
            if (os.path.exists(f"{self.bm25_index_path}.pkl") and 
                os.path.exists(f"{self.vector_index_path}.faiss") and 
                os.path.exists(f"{self.vector_index_path}.pkl")):
                
                print("加载索引...")
                self.retriever.load(self.bm25_index_path, self.vector_index_path)
                return True
            else:
                print("找不到索引文件")
                return False
        except Exception as e:
            print(f"加载索引失败: {str(e)}")
            return False
    
    def clear_knowledge_base(self) -> None:
        """
        清空知识库
        
        删除所有索引数据，包括数据库索引和内存索引
        """
        if self.use_db:
            # 清空数据库索引
            if self.use_elasticsearch and self.es_client:
                try:
                    print("正在删除Elasticsearch索引...")
                    self.es_client.connect()
                    self.es_client.delete_index()
                    print("Elasticsearch索引已清空")
                except Exception as e:
                    print(f"删除Elasticsearch索引失败: {str(e)}")
            
            if self.use_milvus and self.milvus_client:
                try:
                    print("正在删除Milvus集合...")
                    self.milvus_client.connect()
                    self.milvus_client.drop_collection()
                    print("Milvus集合已清空")
                except Exception as e:
                    print(f"删除Milvus集合失败: {str(e)}")
        else:
            # 清空内存索引
            try:
                print("正在删除内存索引文件...")
                # 删除索引文件
                if os.path.exists(f"{self.bm25_index_path}.pkl"):
                    os.remove(f"{self.bm25_index_path}.pkl")
                
                if os.path.exists(f"{self.vector_index_path}.faiss"):
                    os.remove(f"{self.vector_index_path}.faiss")
                
                if os.path.exists(f"{self.vector_index_path}.pkl"):
                    os.remove(f"{self.vector_index_path}.pkl")
                
                # 重新初始化检索器
                self.bm25_retriever = BM25Retriever()
                self.vector_retriever = VectorRetriever()
                self.retriever = HybridRetriever(
                    bm25_retriever=self.bm25_retriever,
                    vector_retriever=self.vector_retriever
                )
                
                print("内存索引已清空")
            except Exception as e:
                print(f"删除内存索引文件失败: {str(e)}")
        
        print("知识库已清空")
    
    def retrieve(self, query: str, top_k: int = None) -> List[Document]:
        """
        检索文档
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            相关文档列表
        """
        top_k = top_k if top_k is not None else config.top_k_retrieval
        
        if self.use_db:
            # 数据库检索
            results = []
            
            # 从Elasticsearch检索
            if self.use_elasticsearch and self.es_client:
                try:
                    es_results = self.es_client.search(query, top_k=top_k)
                    print(f"Elasticsearch检索成功，检索到 {len(es_results)} 个结果")
                    
                    # 为ES结果添加来源标记
                    for doc in es_results:
                        doc.metadata["source_db"] = "elasticsearch"
                    
                    results.extend(es_results)
                except Exception as e:
                    print(f"Elasticsearch检索失败: {str(e)}")
            
            # 从Milvus检索
            if self.use_milvus and self.milvus_client:
                try:
                    milvus_results = self.milvus_client.search(query, top_k=top_k)
                    print(f"Milvus检索成功，检索到 {len(milvus_results)} 个结果")
                    
                    # 为Milvus结果添加来源标记
                    for doc in milvus_results:
                        doc.metadata["source_db"] = "milvus"
                    
                    results.extend(milvus_results)
                except Exception as e:
                    print(f"Milvus检索失败: {str(e)}")
            
            # 如果使用两种数据库，需要去重
            if self.use_elasticsearch and self.use_milvus:
                # 简单去重（基于内容前100个字符）
                content_set = set()
                unique_results = []
                
                for doc in results:
                    content_hash = hash(doc.page_content[:100])
                    if content_hash not in content_set:
                        content_set.add(content_hash)
                        unique_results.append(doc)
                
                # 返回所有去重后的结果，不再限制数量
                return unique_results
            else:
                # 返回所有结果，不再限制数量
                return results
        else:
            # 内存检索
            if not self.retriever:
                raise ValueError("检索器未初始化，请先加载知识库")
            
            return self.retriever.retrieve(query, top_k=top_k)
    
    def answer(self, query: str, use_reflection: bool = False) -> str:
        """
        回答问题
        
        Args:
            query: 用户查询
            use_reflection: 是否使用反思机制
            
        Returns:
            生成的回答
        """
        print(f"正在检索相关文档: {query}")
        
        # 检索相关文档
        documents = self.retrieve(query)
        
        if not documents:
            return "抱歉，我无法找到相关的信息来回答您的问题。"
        
        print(f"找到 {len(documents)} 个相关文档，正在生成回答...")
        
        # 生成回答
        if use_reflection:
            answer = self.llm_client.generate_with_reflection(query, documents)
        else:
            answer = self.llm_client.generate(query, documents)
        
        return answer


if __name__ == "__main__":
    # 测试RAG管理器
    rag_manager = RAGManager(use_db=False)  # 使用内存索引
    
    # 创建文档
    test_doc1 = Document(
        page_content="最近的一项研究显示，人工智能在医疗领域的应用越来越广泛。从辅助诊断到药物研发，AI技术正在帮助医生更准确地诊断疾病并开发新的治疗方法。",
        metadata={"source": "test_doc_1.txt", "topic": "AI医疗"}
    )
    test_doc2 = Document(
        page_content="自然语言处理（NLP）技术的进步使得计算机能够更好地理解和生成人类语言。这种技术的应用包括机器翻译、智能客服和内容生成等领域。",
        metadata={"source": "test_doc_2.txt", "topic": "NLP"}
    )
    
    # 创建知识库
    rag_manager.create_knowledge_base_from_documents([test_doc1, test_doc2])
    
    # 添加更多文档
    test_doc3 = Document(
        page_content="大型语言模型（LLM）如GPT系列是近年来AI领域的重大突破。它们通过海量数据训练，能够生成连贯、相关和有意义的文本，实现各种自然语言处理任务。",
        metadata={"source": "test_doc_3.txt", "topic": "LLM"}
    )
    rag_manager.add_documents([test_doc3])
    
    # 提问
    query = "AI在医疗领域有哪些应用？"
    answer = rag_manager.answer(query)
    
    print(f"问题: {query}")
    print(f"回答: {answer}") 
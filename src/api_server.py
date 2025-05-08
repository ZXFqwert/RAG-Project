"""
RAG系统API服务器
"""

import os
import json
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from pathlib import Path
import shutil

from .rag_manager import RAGManager
from .utils.config import config
from .document_processor.document_loader import DocumentLoader
from langchain.docstore.document import Document

# 创建FastAPI应用
app = FastAPI(
    title="RAG API",
    description="RAG系统API服务",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局RAG管理器实例
rag_manager = None

# 请求和响应模型
class QueryRequest(BaseModel):
    query: str
    use_reflection: bool = False


class QueryResponse(BaseModel):
    answer: str
    documents: List[Dict[str, Any]]


class IndexingResponse(BaseModel):
    success: bool
    message: str


# 初始化RAG管理器
def initialize_rag_manager():
    global rag_manager
    if rag_manager is None:
        rag_manager = RAGManager()
        rag_manager.load_knowledge_base()
    return rag_manager


@app.on_event("startup")
async def startup_event():
    initialize_rag_manager()


@app.get("/")
def read_root():
    return {"message": "RAG API 服务运行中"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    查询RAG系统
    """
    manager = initialize_rag_manager()
    
    try:
        # 检索文档
        documents = manager.retrieve(request.query)
        
        # 生成回答
        answer = manager.answer(request.query, use_reflection=request.use_reflection)
        
        # 格式化文档
        formatted_docs = []
        for doc in documents:
            formatted_docs.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        
        return QueryResponse(
            answer=answer,
            documents=formatted_docs
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents/upload", response_model=IndexingResponse)
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    rebuild_index: bool = Form(False)
):
    """
    上传文档到知识库
    """
    manager = initialize_rag_manager()
    uploaded_files = []
    
    try:
        # 确保文档目录存在
        os.makedirs(config.documents_dir, exist_ok=True)
        
        # 保存上传的文件
        for file in files:
            file_path = os.path.join(config.documents_dir, file.filename)
            
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            uploaded_files.append(file_path)
        
        # 异步处理索引创建/更新
        def process_documents():
            if rebuild_index:
                # 重建整个知识库
                manager.create_knowledge_base()
            else:
                # 只处理新上传的文档
                loader = DocumentLoader()
                documents = []
                
                for file_path in uploaded_files:
                    doc_content = loader.load_document(file_path)
                    if doc_content:
                        metadata = {
                            "source": file_path,
                            "filename": os.path.basename(file_path),
                            "filetype": os.path.splitext(file_path)[1][1:]
                        }
                        documents.append(Document(page_content=doc_content, metadata=metadata))
                
                if documents:
                    manager.add_documents(documents)
        
        # 添加后台任务
        background_tasks.add_task(process_documents)
        
        return IndexingResponse(
            success=True,
            message=f"已上传 {len(files)} 个文件，正在处理..."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
def get_status():
    """
    获取系统状态
    """
    manager = initialize_rag_manager()
    
    try:
        # 尝试获取各个数据库的文档数量
        status = {
            "system": "running",
            "documents_count": 0
        }
        
        if manager.use_db:
            if manager.use_elasticsearch and manager.es_client:
                status["elasticsearch_count"] = manager.es_client.get_document_count()
            
            if manager.use_milvus and manager.milvus_client:
                status["milvus_count"] = manager.milvus_client.get_document_count()
        else:
            # 使用内存索引
            if manager.bm25_retriever and manager.bm25_retriever.documents:
                status["documents_count"] = len(manager.bm25_retriever.documents)
        
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """启动API服务器"""
    uvicorn.run("src.api_server:app", host=host, port=port, reload=True)


if __name__ == "__main__":
    run_server()
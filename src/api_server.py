"""
RAG系统API服务器
"""

import os
import json
import asyncio
from typing import List, Dict, Any, Optional, Union
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks, WebSocket, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
from pathlib import Path
import shutil

from .rag_manager import RAGManager
from .utils.config import config
from .document_processor.document_loader import DocumentLoader
from langchain.docstore.document import Document

# 创建目录
templates_dir = Path(__file__).parent / "templates"
templates_dir.mkdir(exist_ok=True)
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)

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

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# 设置模板引擎
templates = Jinja2Templates(directory=templates_dir)

# 全局RAG管理器实例
rag_manager = None

# 活跃的WebSocket连接
active_connections: Dict[str, WebSocket] = {}

# 聊天历史记录
chat_histories: Dict[str, List[Dict[str, str]]] = {}

# 请求和响应模型
class QueryRequest(BaseModel):
    query: str
    use_reflection: bool = False
    session_id: str = "default"


class StreamQueryRequest(BaseModel):
    query: str
    use_reflection: bool = False
    session_id: str = "default"


class QueryResponse(BaseModel):
    answer: str
    documents: List[Dict[str, Any]]
    es_results: List[Dict[str, Any]] = []
    milvus_results: List[Dict[str, Any]] = []


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


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/chat-history")
def get_chat_history(session_id: str = "default"):
    """获取聊天历史"""
    if session_id not in chat_histories:
        chat_histories[session_id] = []
    
    return {"history": chat_histories[session_id]}


@app.delete("/api/chat-history")
def clear_chat_history(session_id: str = "default"):
    """清空聊天历史"""
    chat_histories[session_id] = []
    return {"success": True, "message": "聊天历史已清空"}


@app.post("/api/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    查询RAG系统
    """
    manager = initialize_rag_manager()
    
    try:
        # 检索文档
        documents = manager.retrieve(request.query)
        
        # 分离ES和Milvus的结果
        es_results = []
        milvus_results = []
        
        for doc in documents:
            if "source_db" in doc.metadata:
                if doc.metadata["source_db"] == "elasticsearch":
                    es_results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    })
                elif doc.metadata["source_db"] == "milvus":
                    milvus_results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    })
        
        # 生成回答
        answer = manager.answer(request.query, use_reflection=request.use_reflection)
        
        # 保存到聊天历史
        if request.session_id not in chat_histories:
            chat_histories[request.session_id] = []
        
        chat_histories[request.session_id].append({
            "role": "user",
            "content": request.query
        })
        
        chat_histories[request.session_id].append({
            "role": "assistant",
            "content": answer
        })
        
        # 格式化文档
        formatted_docs = []
        for doc in documents:
            formatted_docs.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        
        return QueryResponse(
            answer=answer,
            documents=formatted_docs,
            es_results=es_results,
            milvus_results=milvus_results
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def stream_generator(query: str, use_reflection: bool, session_id: str):
    """生成流式回答"""
    manager = initialize_rag_manager()
    
    try:
        # 检索文档
        documents = manager.retrieve(query)
        
        # 分离ES和Milvus的结果
        es_results = []
        milvus_results = []
        
        for doc in documents:
            if "source_db" in doc.metadata:
                if doc.metadata["source_db"] == "elasticsearch":
                    es_results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    })
                elif doc.metadata["source_db"] == "milvus":
                    milvus_results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    })
        
        # 格式化文档
        formatted_docs = []
        for doc in documents:
            formatted_docs.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        
        # 先返回检索结果
        yield json.dumps({
            "type": "retrieval_results",
            "documents": formatted_docs,
            "es_results": es_results,
            "milvus_results": milvus_results
        })
        
        # 模拟流式输出（实际实现需要修改LLM客户端支持流式返回）
        answer = manager.answer(query, use_reflection=use_reflection)
        
        # 分段返回答案，模拟流式效果
        words = answer.split()
        full_answer = ""
        
        for i in range(0, len(words), 3):
            chunk = " ".join(words[i:i+3])
            full_answer += chunk + " "
            yield json.dumps({
                "type": "token",
                "token": chunk + " "
            })
            await asyncio.sleep(0.1)  # 模拟延迟
        
        # 保存到聊天历史
        if session_id not in chat_histories:
            chat_histories[session_id] = []
        
        chat_histories[session_id].append({
            "role": "user",
            "content": query
        })
        
        chat_histories[session_id].append({
            "role": "assistant",
            "content": full_answer
        })
        
        # 完成消息
        yield json.dumps({
            "type": "end",
            "answer": full_answer
        })
    
    except Exception as e:
        yield json.dumps({
            "type": "error",
            "message": str(e)
        })


@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connection_id = str(id(websocket))
    active_connections[connection_id] = websocket
    
    try:
        while True:
            # 接收消息
            data = await websocket.receive_text()
            request_data = json.loads(data)
            
            # 提取参数
            query = request_data.get("query", "")
            use_reflection = request_data.get("use_reflection", False)
            session_id = request_data.get("session_id", "default")
            
            # 流式生成回答
            async for chunk in stream_generator(query, use_reflection, session_id):
                await websocket.send_text(chunk)
    
    except Exception as e:
        print(f"WebSocket错误: {str(e)}")
    finally:
        # 连接关闭时清理
        if connection_id in active_connections:
            del active_connections[connection_id]


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
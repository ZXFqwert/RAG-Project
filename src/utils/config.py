"""
配置加载模块，处理环境变量和系统配置
"""

import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from pathlib import Path

# 加载环境变量
load_dotenv()

class Config:
    """配置管理类，负责加载和提供配置项"""
    
    def __init__(self, env_path: Optional[str] = None):
        """
        初始化配置
        
        Args:
            env_path: 环境变量文件路径（可选）
        """
        if env_path:
            load_dotenv(env_path)
        
        # Ollama配置
        self.ollama_api_base = os.getenv("OLLAMA_API_BASE", "http://175.6.27.232:11435")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "bge-m3:latest")
        
        # Qwen配置
        self.qwen_api_base = os.getenv("QWEN_API_BASE", "http://175.6.27.232:8102/v1")
        self.qwen_api_key = os.getenv("QWEN_API_KEY", "")
        
        # Elasticsearch配置
        self.elasticsearch_host = os.getenv("ELASTICSEARCH_HOST", "localhost")
        self.elasticsearch_port = int(os.getenv("ELASTICSEARCH_PORT", "9200"))
        self.elasticsearch_username = os.getenv("ELASTICSEARCH_USERNAME", "")
        self.elasticsearch_password = os.getenv("ELASTICSEARCH_PASSWORD", "")
        
        # Milvus配置
        self.milvus_host = os.getenv("MILVUS_HOST", "localhost")
        self.milvus_port = int(os.getenv("MILVUS_PORT", "19530"))
        self.milvus_user = os.getenv("MILVUS_USER", "")
        self.milvus_password = os.getenv("MILVUS_PASSWORD", "")
        self.milvus_collection_name = os.getenv("MILVUS_COLLECTION_NAME", "rag_documents")
        
        # 系统配置
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "512"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "50"))
        self.top_k_retrieval = int(os.getenv("TOP_K_RETRIEVAL", "5"))
        self.hybrid_alpha = float(os.getenv("HYBRID_ALPHA", "0.5"))
        
        # 特性开关
        self.use_elasticsearch = os.getenv("USE_ELASTICSEARCH", "true").lower() == "true"
        self.use_milvus = os.getenv("USE_MILVUS", "true").lower() == "true"
        
        # 文件路径配置
        self.base_dir = Path(__file__).parent.parent.parent  # RAG目录
        self.data_dir = self.base_dir / "data"
        self.documents_dir = self.data_dir / "documents"
        self.embeddings_dir = self.data_dir / "embeddings"
        
        # 确保必要的目录存在
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    def get_elasticsearch_config(self) -> Dict[str, Any]:
        """获取Elasticsearch配置"""
        config = {
            "hosts": [f"{self.elasticsearch_host}:{self.elasticsearch_port}"]
        }
        
        if self.elasticsearch_username and self.elasticsearch_password:
            config["basic_auth"] = (self.elasticsearch_username, self.elasticsearch_password)
            
        return config
    
    def get_milvus_config(self) -> Dict[str, Any]:
        """获取Milvus配置"""
        config = {
            "host": self.milvus_host,
            "port": self.milvus_port
        }
        
        if self.milvus_user and self.milvus_password:
            config["user"] = self.milvus_user
            config["password"] = self.milvus_password
            
        return config

# 创建全局配置实例
config = Config()

if __name__ == "__main__":
    # 测试配置加载
    print(f"Ollama API Base: {config.ollama_api_base}")
    print(f"Documents Directory: {config.documents_dir}")
    print(f"Using Elasticsearch: {config.use_elasticsearch}")
    print(f"Using Milvus: {config.use_milvus}") 
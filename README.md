# RAG系统

一个基于混合检索策略（BM25 + 语义向量检索）的检索增强生成系统。

## 功能特性

- **混合检索策略**：结合了BM25稀疏检索和语义向量稠密检索
- **多种数据库支持**：支持使用Elasticsearch和Milvus作为后端数据库
- **内存模式**：也可以使用内存索引模式，无需安装数据库
- **增量更新**：支持增量添加文档到知识库
- **多格式文档处理**：支持TXT、PDF、DOCX等多种文档格式
- **API服务**：提供RESTful API接口
- **命令行工具**：提供命令行交互方式

## 系统架构

系统由以下主要模块组成：

1. **文档处理模块**：负责文档的加载、处理和分块
2. **索引模块**：包括BM25索引和向量索引的创建和管理
3. **检索模块**：实现混合检索策略，结合BM25和向量检索结果
4. **生成模块**：将用户问题和检索结果输入大模型，生成回答
5. **数据库模块**：提供对Elasticsearch和Milvus的支持
6. **增量更新模块**：支持知识库的增量更新

## 安装依赖

1. 安装Python依赖：

```bash
pip install -r requirements.txt
```

2. 可选：安装Elasticsearch和Milvus

- Elasticsearch: https://www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html
- Milvus: https://milvus.io/docs/install_standalone-docker.md

## 使用方法

### 配置

复制`env.example`为`.env`并编辑配置：

```bash
cp env.example .env
```

### 命令行工具

#### 创建知识库

```bash
# 使用默认文档目录
python main.py create

# 使用自定义文档目录
python main.py create --documents-dir /path/to/documents

# 使用内存索引模式
python main.py create --memory
```

#### 查询知识库

```bash
# 交互式查询
python main.py query

# 单次查询
python main.py query --query "你的问题"

# 使用反思机制
python main.py query --reflection

# 使用内存索引模式
python main.py query --memory
```

#### 添加文档到知识库

```bash
# 使用默认文档目录
python main.py add

# 使用自定义文档目录
python main.py add --documents-dir /path/to/documents
```

#### 启动API服务器

```bash
python main.py server --port 8000
```

### API接口

启动API服务器后，可以使用以下接口：

- `GET /`: 检查服务状态
- `POST /query`: 查询知识库
- `POST /documents/upload`: 上传文档到知识库
- `GET /status`: 获取系统状态

## 开发者指南

### 项目结构

```
RAG/
├── data/                   # 数据目录
│   ├── documents/          # 文档目录
│   └── embeddings/         # 嵌入向量存储目录
├── src/                    # 源代码
│   ├── document_processor/ # 文档处理模块
│   ├── retrieval/          # 检索模块
│   ├── database/           # 数据库模块
│   ├── generation/         # 生成模块
│   ├── utils/              # 工具模块
│   ├── api_server.py       # API服务器
│   ├── cli.py              # 命令行工具
│   └── rag_manager.py      # RAG系统管理器
├── main.py                 # 主入口
├── requirements.txt        # 依赖列表
└── env.example             # 环境变量示例
```

### 模块说明

- **document_processor**: 提供文档的读取、解析和分块功能
- **retrieval**: 实现BM25检索、向量检索和混合检索策略
- **database**: 提供Elasticsearch和Milvus数据库接口
- **generation**: 提供LLM生成接口
- **utils**: 工具函数和配置加载
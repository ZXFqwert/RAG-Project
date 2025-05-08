"""
RAG系统命令行工具
"""

import argparse
import os
import sys
from typing import List, Optional
from pathlib import Path

from .rag_manager import RAGManager
from .utils.config import config
from .api_server import run_server


def create_knowledge_base(args):
    """创建知识库"""
    print("创建知识库...")
    
    rag_manager = RAGManager(use_db=not args.memory)
    
    documents_dir = args.documents_dir if args.documents_dir else config.documents_dir
    print(f"使用文档目录: {documents_dir}")
    
    rag_manager.create_knowledge_base(documents_dir)
    
    print("知识库创建完成")


def query(args):
    """查询知识库"""
    print("加载知识库...")
    
    rag_manager = RAGManager(use_db=not args.memory)
    if not rag_manager.load_knowledge_base():
        print("加载知识库失败，请先创建知识库")
        return
    
    if args.query:
        # 单次查询
        print(f"\n问题: {args.query}")
        answer = rag_manager.answer(args.query, use_reflection=args.reflection)
        print(f"\n回答: {answer}\n")
    else:
        # 交互式模式
        print("\n进入交互式查询模式，输入'退出'或'exit'结束\n")
        while True:
            query_text = input("\n问题: ")
            if query_text.lower() in ["退出", "exit", "quit", "q"]:
                break
            
            answer = rag_manager.answer(query_text, use_reflection=args.reflection)
            print(f"\n回答: {answer}\n")


def add_documents(args):
    """添加文档到知识库"""
    print("加载知识库...")
    
    rag_manager = RAGManager(use_db=not args.memory)
    rag_manager.load_knowledge_base()
    
    documents_dir = args.documents_dir if args.documents_dir else config.documents_dir
    print(f"从目录添加文档: {documents_dir}")
    
    rag_manager.create_knowledge_base(documents_dir)
    
    print("文档添加完成")


def run_api(args):
    """启动API服务器"""
    print(f"启动API服务器，端口: {args.port}...")
    run_server(host=args.host, port=args.port)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="RAG系统命令行工具")
    subparsers = parser.add_subparsers(dest="command", help="命令")
    
    # 创建知识库命令
    create_parser = subparsers.add_parser("create", help="创建知识库")
    create_parser.add_argument("--documents-dir", "-d", type=str, help="文档目录路径")
    create_parser.add_argument("--memory", "-m", action="store_true", help="使用内存索引而不是数据库")
    
    # 查询命令
    query_parser = subparsers.add_parser("query", help="查询知识库")
    query_parser.add_argument("--query", "-q", type=str, help="查询文本")
    query_parser.add_argument("--reflection", "-r", action="store_true", help="使用反思机制")
    query_parser.add_argument("--memory", "-m", action="store_true", help="使用内存索引而不是数据库")
    
    # 添加文档命令
    add_parser = subparsers.add_parser("add", help="添加文档到知识库")
    add_parser.add_argument("--documents-dir", "-d", type=str, help="文档目录路径")
    add_parser.add_argument("--memory", "-m", action="store_true", help="使用内存索引而不是数据库")
    
    # API服务器命令
    server_parser = subparsers.add_parser("server", help="启动API服务器")
    server_parser.add_argument("--host", type=str, default="0.0.0.0", help="主机地址")
    server_parser.add_argument("--port", "-p", type=int, default=8000, help="端口号")
    
    args = parser.parse_args()
    
    if args.command == "create":
        create_knowledge_base(args)
    elif args.command == "query":
        query(args)
    elif args.command == "add":
        add_documents(args)
    elif args.command == "server":
        run_api(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
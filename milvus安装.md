# Milvus 安装指南

Milvus是一个开源的向量数据库，专为嵌入式向量相似性搜索和人工智能应用设计。本指南将帮助您使用Docker安装和配置Milvus。

## 1. 系统要求

- Docker 19.03或更高版本
- Docker Compose 1.24.1或更高版本
- 至少2GB可用内存
- 至少10GB可用磁盘空间

## 2. 创建目录结构

```bash
# 创建Milvus数据和配置目录
mkdir -p /home/roots/docker/milvus/{data,conf,logs}
chmod -R 777 /home/roots/docker/milvus
```

## 3. 下载Docker Compose配置文件

```bash
# 下载Milvus Lite版本的docker-compose文件
cd /home/roots/docker/milvus
wget https://github.com/milvus-io/milvus/releases/download/v2.3.3/milvus-standalone-docker-compose.yml -O docker-compose.yml
```

## 4. 修改Docker Compose配置

默认的Docker Compose配置使用的是默认端口19530。如果需要修改端口或其他配置，可以编辑docker-compose.yml文件：

```bash
# 编辑配置文件
nano docker-compose.yml
```

修改端口映射部分（如果需要）：
```yaml
ports:
  - "19530:19530"
  - "9091:9091"
```

## 5. 启动Milvus服务

```bash
# 启动Milvus服务
docker-compose up -d
```

## 6. 验证Milvus服务状态

```bash
# 检查容器状态
docker-compose ps

# 查看日志
docker-compose logs -f milvus-standalone
```

## 7. 配置RAG项目使用Milvus

确保RAG项目的`.env`文件中包含以下配置：

```
# Milvus设置
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_USER=
MILVUS_PASSWORD=
MILVUS_COLLECTION_NAME=rag_documents
USE_MILVUS=true
```

## 8. 常见问题排查

### 连接问题

如果遇到连接问题，请检查：

1. Milvus容器是否正常运行：
```bash
docker ps | grep milvus
```

2. 网络连接是否正常：
```bash
telnet localhost 19530
```

3. 查看Milvus日志：
```bash
docker logs milvus-standalone
```

### 内存不足

如果遇到内存不足的问题，可以修改docker-compose.yml文件中的资源限制：

```yaml
environment:
  MALLOC_CONF: prof:true,background_thread:true,thp:never,metadata_thp:auto
  MILVUS_RESOURCE_LIMIT: "memory:4Gi,cpu:2"
```

## 9. 停止和重启Milvus

```bash
# 停止Milvus服务
cd /home/roots/docker/milvus
docker-compose down

# 重启Milvus服务
docker-compose up -d
```

## 10. 卸载Milvus

如果需要完全卸载Milvus：

```bash
# 停止并删除容器
cd /home/roots/docker/milvus
docker-compose down -v

# 删除数据目录
rm -rf /home/roots/docker/milvus
```

## 11. 升级Milvus

要升级到新版本的Milvus：

```bash
# 停止当前服务
cd /home/roots/docker/milvus
docker-compose down

# 下载新版本的docker-compose文件
wget https://github.com/milvus-io/milvus/releases/download/[NEW_VERSION]/milvus-standalone-docker-compose.yml -O docker-compose.yml

# 启动新版本
docker-compose up -d
```

请将`[NEW_VERSION]`替换为您要升级到的Milvus版本。 
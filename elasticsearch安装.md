## Elasticsearch 和 Kibana 安装指南

### 1. 系统配置准备

```bash
# 设置系统参数，避免启动错误
sudo sysctl -w vm.max_map_count=262144
```

### 2. 创建目录结构

```bash
# 创建配置、数据和插件目录
mkdir -p /home/roots/docker/elasticsearch/{config,data,plugins}

# 设置目录权限
chmod -R 777 /home/roots/docker/elasticsearch
```

### 3. 配置 Elasticsearch

```bash
# 设置允许外部访问
echo "http.host: 0.0.0.0" > /home/roots/docker/elasticsearch/config/elasticsearch.yml
```

### 4. 拉取 Elasticsearch 镜像

```bash
docker pull elasticsearch:7.17.10
```

### 5. 启动 Elasticsearch 容器

```bash
docker run --name RAG-es -p 9201:9200 -p 9301:9300 \
-e "discovery.type=single-node" \
-e ES_JAVA_OPTS="-Xms512m -Xmx512m" \
-e xpack.security.enabled=false \
-v /home/roots/docker/elasticsearch/config/elasticsearch.yml:/usr/share/elasticsearch/config/elasticsearch.yml \
-v /home/roots/docker/elasticsearch/data:/usr/share/elasticsearch/data \
-v /home/roots/docker/elasticsearch/plugins:/usr/share/elasticsearch/plugins \
-d elasticsearch:7.17.10
```

### 6. 验证 Elasticsearch 安装

```bash
# 检查是否成功启动
curl http://localhost:9201

# 设置开机自启动
docker update RAG-es --restart=always
```

### 7. 拉取 Kibana 镜像

```bash
docker pull kibana:7.17.10
```

### 8. 启动 Kibana 容器

```bash
docker run --name RAG-kibana \
-p 5601:5601 \
-e "ELASTICSEARCH_HOSTS=http://175.6.27.232:9201" \
-d kibana:7.17.10

# 设置开机自启动
docker update RAG-kibana --restart=always
```

### 9. 配置 Kibana 中文界面

```bash
# 正确添加中文设置（确保添加换行）
docker exec -it RAG-kibana bash -c 'echo -e "\ni18n.locale: \"zh-CN\"" >> /usr/share/kibana/config/kibana.yml'

# 重启 Kibana 使配置生效
docker restart RAG-kibana
```

### 10. 安装 IK 分词器（用于中文搜索）

```bash
# 进入插件目录
cd /home/roots/docker/elasticsearch/plugins

# 下载IK分词器（使用官方推荐的新链接）
wget https://get.infini.cloud/elasticsearch/analysis-ik/7.17.10

# 解压到 ik 目录
mkdir -p ik
unzip 7.17.10 -d ik/
rm 7.17.10

# 设置权限
chmod -R 777 ik

# 重启 Elasticsearch
docker restart RAG-es
```

### 11. 验证 IK 分词器安装

```bash
# 查看已安装的插件
curl -X GET "http://localhost:9201/_cat/plugins"

# 测试 IK 分词器
curl -X POST "http://localhost:9201/_analyze" -H 'Content-Type: application/json' -d'{"analyzer": "ik_smart", "text": "中华人民共和国国歌"}'

# ik_max_word模式
curl -X POST "http://localhost:9201/_analyze" -H 'Content-Type: application/json' -d'{"analyzer": "ik_max_word", "text": "中华人民共和国国歌"}' | python -m json.tool
```

### 12. 访问服务

- Elasticsearch: http://175.6.27.232:9201
- Kibana: http://175.6.27.232:5601

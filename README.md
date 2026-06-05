# 智能RAG知识库问答系统 v2

基于 **ChromaDB 向量数据库 + DeepSeek API** 的企业级 RAG (检索增强生成) 问答系统。

## 功能

- 📄 支持上传 TXT / PDF / DOCX 文档
- 🔍 **语义检索**（ChromaDB + n-gram哈希嵌入）
- 📊 检索相似度分数 + 来源追踪
- 🤖 基于 DeepSeek API 智能回答
- 💬 多轮对话记忆
- ⚡ 流式输出（打字机效果）

## 项目结构

```
rag-project/
├── backend/
│   ├── main.py          # FastAPI 后端服务
│   ├── rag_engine.py    # RAG 核心引擎（ChromaDB检索 + LLM问答）
│   └── requirements.txt
├── frontend/
│   └── app.py           # Streamlit 前端界面
├── data/                # 文档存储目录
└── README.md
```

## 运行方式

### 1. 激活环境

```bash
conda activate ai_app
```

### 2. 启动后端

```bash
cd rag-project
python backend/main.py
# 后端运行在 http://localhost:8000
# API 文档在 http://localhost:8000/docs
```

### 3. 启动前端

```bash
cd rag-project
streamlit run frontend/app.py --server.port 8501
# 界面在 http://localhost:8501
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 系统信息 |
| POST | `/upload` | 上传文档并导入向量库 |
| POST | `/ask` | 问答（返回结果+来源+分数） |
| GET | `/ask?q=问题` | 流式问答 |
| GET | `/files` | 文件列表 |
| DELETE | `/files/{name}` | 删除文件 |
| GET | `/stats` | 引擎状态（Embedding类型、chunk数等） |

## 核心技术

### RAG 流程

```
文档上传 → 智能分块(300字符+50重叠) → n-gram哈希嵌入 → 存入ChromaDB
                                                      ↓
用户提问 → 问题嵌入 → ChromaDB余弦相似度检索 → 拼Prompt → 调DeepSeek → 回答
```

### 本地Embedding引擎

- 基于字符 n-gram 哈希（n=1,2,3）的特征嵌入
- 256维向量，L2归一化
- 纯 numpy 实现，无需下载模型
- 可插拔设计，可替换为 sentence-transformers

### ChromaDB 向量数据库

- 持久化存储，重启不丢失
- ANN（近似最近邻）索引，快速检索
- 支持 metadata 过滤和来源追踪

## 面试重点

- **RAG架构**: 检索增强生成，解决AI私有知识问答
- **语义检索**: 比TF-IDF能理解同义词、近义词
- **向量数据库**: ChromaDB与传统数据库的区别
- **分块策略**: 重叠chunk防止切碎语义
- **可插拔设计**: Embedding层可无缝切换

## v1 → v2 升级

| 维度 | v1 | v2 |
|------|-----|-----|
| 检索方式 | TF-IDF词频统计 | ChromaDB语义检索 |
| 分块 | 简单按空行切 | 智能分块+重叠 |
| 结果展示 | 仅回答 | 回答+来源+相似度分数 |
| 配置 | 硬编码 | Config类管理 |

## 升级路线

- 替换 Embedding → sentence-transformers / BGE
- 加 Rerank 重排序 → 提高检索精度
- 加 Graph RAG → 处理多文档关系
- 部署到云服务器 → 公网可访问

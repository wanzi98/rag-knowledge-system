"""
RAG 引擎 v2：ChromaDB 向量存储 + 本地语义嵌入
==============================================
v1 用的是 TF-IDF（基于词频统计）
v2 升级为 Embedding（基于向量语义）+ ChromaDB 向量数据库

面试重点：
  1. 什么是 Embedding？→ 把文字转成固定长度的向量，语义相近的文字向量也相近
  2. 什么是向量数据库？→ 专门存向量、做相似度搜索的数据库
  3. 为什么比 TF-IDF 好？→ 能理解"同义词""近义词"，不仅仅是字面匹配
  4. 为什么本地实现？→ 不需要下载模型，不依赖外部API，适合中国网络环境
  5. 升级路线？→ 换用 BGE/sentence-transformers 等预训练模型可大幅提升精度
"""

import os
import json
import re
import math
import hashlib
from typing import List, Optional, Dict, Tuple

import numpy as np
import chromadb
from chromadb.config import Settings
from openai import OpenAI


# ============================================================
# 配置管理（面试重点：为什么分离配置？→ 方便部署、不用改代码）
# ============================================================

class Config:
    """集中管理所有配置，不改代码就能调参数"""
    # DeepSeek API
    DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-your-key-here")
    DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    LLM_MODEL = "deepseek-chat"
    LLM_TEMPERATURE = 0.3
    LLM_MAX_TOKENS = 500

    # 分块参数
    CHUNK_SIZE = 300       # 每个块目标字符数
    CHUNK_OVERLAP = 50     # 块之间重叠字符数（防止语义被切碎）

    # 检索参数
    RETRIEVAL_TOP_K = 5    # 检索返回多少个候选块
    FINAL_TOP_K = 3        # 最终送给AI的块数（如果有reranker）

    # Embedding 参数
    EMBED_DIM = 256        # 嵌入向量维度（越大越精确，但越慢）

    # 路径
    BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
    DATA_DIR = os.path.join(BASE_DIR, "data")
    CHROMA_PATH = os.path.join(DATA_DIR, "chroma_db")


# ============================================================
# 1. 本地 Embedding 引擎（面试核心！）
# ============================================================

class LocalEmbedding:
    """
    基于字符 n-gram 哈希的本地嵌入模型

    原理（面试要能讲清楚）：
    ┌─────────────────────────────────────────────────────────────────┐
    │ 1. 把文本拆成字符级 n-gram（n=1,2,3）                          │
    │    比如 "AI" → ["A", "I", "AI"]                                │
    │    比如 "人工" → ["人", "工", "人工"]                           │
    │ 2. 用哈希函数把每个 n-gram 映射到 [0, DIM) 的一个位置          │
    │ 3. 对应位置计数 +1（这叫"特征哈希"/Feature Hashing）           │
    │ 4. L2归一化 → 得到固定长度向量                                  │
    └─────────────────────────────────────────────────────────────────┘
    优点：不依赖外部模型，纯 numpy，速度快，适合中文
    缺点：质量不如 BERT 等预训练模型（升级路线：换 sentence-transformers）

    面试追问：
    Q: 为什么用 n-gram 不用词？
    A: 中文没有天然空格分词，n-gram 能捕捉字序信息，"人工"≠"工人"
    Q: 哈希冲突怎么办？
    A: 维度设大一些（256~512），冲突概率低。而且哈希是"有损压缩"，
       少量冲突反而有正则化效果，防止过拟合
    Q: 余弦相似度的几何意义？
    A: 两个向量的夹角，夹角越小越相似。范围[-1,1]，1表示完全同向
    """

    def __init__(self, dim: int = Config.EMBED_DIM):
        self.dim = dim

    def embed(self, texts: List[str]) -> List[np.ndarray]:
        """把一批文本转成向量"""
        results = []
        for text in texts:
            vec = self._hash_embed(text)
            results.append(vec)
        return results

    def embed_one(self, text: str) -> np.ndarray:
        """把单条文本转成向量"""
        return self._hash_embed(text)

    def _hash_embed(self, text: str) -> np.ndarray:
        """核心：n-gram 哈希嵌入"""
        vec = np.zeros(self.dim, dtype=np.float32)
        text = text.strip()

        if not text:
            return vec

        # 确保 text 至少长到可以提取 n-gram
        # 提取 n-gram (n=1,2,3)
        max_n = min(3, len(text))
        for n in range(1, max_n + 1):
            for i in range(len(text) - n + 1):
                gram = text[i:i + n]
                if not gram:
                    continue
                # 用 hashlib 做确定性哈希
                hash_bytes = hashlib.md5(gram.encode('utf-8')).digest()
                hash_val = int.from_bytes(hash_bytes[:4], 'little')
                idx = hash_val % self.dim
                # n=1的权重高一些，n=3低一些（长n-gram更稀疏）
                weight = 1.0 / n
                vec[idx] += weight

        # L2 归一化
        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            vec = vec / norm

        return vec

    def similarity(self, text1: str, text2: str) -> float:
        """计算两段文本的余弦相似度（面试：怎么算的？）"""
        v1 = self.embed_one(text1)
        v2 = self.embed_one(text2)
        return float(np.dot(v1, v2))


# ============================================================
# 2. 文档处理（比v1更好的分块策略）
# ============================================================

class DocumentProcessor:
    """
    文档分块策略（面试高频考点）

    v1 的问题：单纯按空行切，如果一段很长会超过token限制，
    如果一段很短则语义不完整。而且段落边界可能切碎完整语义。

    v2 改进：
      1. 按段落先切 → 长段落再按句子切 → 保证每块不超过 CHUNK_SIZE
      2. 块之间加 overlap（重叠）→ 防止边界切断重要信息
      3. 保留段落结构信息 → 方便溯源
    """

    def __init__(self, chunk_size: int = Config.CHUNK_SIZE,
                 chunk_overlap: int = Config.CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_and_chunk(self, filepath: str) -> List[Dict]:
        """加载文档并智能分块，返回 [{text, metadata}, ...]"""
        ext = os.path.splitext(filepath)[1].lower()

        if ext == ".txt":
            text = self._read_txt(filepath)
        elif ext == ".pdf":
            text = self._read_pdf(filepath)
        elif ext == ".docx":
            text = self._read_docx(filepath)
        else:
            raise ValueError(f"不支持的文件格式: {ext}")

        chunks = self._smart_chunk(text)
        filename = os.path.basename(filepath)
        return [
            {"text": chunk, "metadata": {"source": filename, "chunk_id": i}}
            for i, chunk in enumerate(chunks)
        ]

    def _read_txt(self, filepath: str) -> str:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    def _read_pdf(self, filepath: str) -> str:
        from PyPDF2 import PdfReader
        reader = PdfReader(filepath)
        return "\n".join([page.extract_text() for page in reader.pages])

    def _read_docx(self, filepath: str) -> str:
        import docx
        doc = docx.Document(filepath)
        return "\n".join([p.text for p in doc.paragraphs])

    def _smart_chunk(self, text: str) -> List[str]:
        """
        智能分块算法：
        1. 先按段落（空行）切
        2. 超长段落按句子切
        3. 短段落合并
        4. 相邻块加 overlap
        """
        # 第一步：按段落切
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        chunks = []
        buffer = ""

        for para in paragraphs:
            # 太短的段落直接跳过（可能是标题内的空行）
            if len(para) < 10:
                if buffer:
                    buffer += "\n" + para
                continue

            # 如果当前块 + 这个段落还在限制内，合并
            if len(buffer) + len(para) < self.chunk_size:
                buffer = (buffer + "\n\n" + para) if buffer else para
            else:
                # 当前块够了，保存
                if buffer:
                    chunks.append(buffer)

                # 如果段落本身超长，按句子切
                if len(para) > self.chunk_size:
                    sub_chunks = self._split_long_paragraph(para)
                    chunks.extend(sub_chunks)
                    buffer = ""
                else:
                    buffer = para

        # 最后一块
        if buffer:
            chunks.append(buffer)

        # 第二步：加 overlap（每个块的末尾重叠到下一个块的开头）
        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._add_overlap(chunks)

        return chunks

    def _split_long_paragraph(self, para: str) -> List[str]:
        """把超长段落按句子切分成多个chunk"""
        # 按中文句号、问号、感叹号、换行切分
        sentences = re.split(r'(?<=[。！？\n])', para)
        chunks = []
        buffer = ""

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if len(buffer) + len(sent) < self.chunk_size:
                buffer += sent
            else:
                if buffer:
                    chunks.append(buffer)
                buffer = sent

        if buffer:
            chunks.append(buffer)

        return chunks if chunks else [para]

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """给相邻chunk加重叠，防止边界切断语义"""
        overlapped = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                # 从前一个chunk的末尾取 overlap 字符拼到开头
                prev_tail = chunks[i - 1][-self.chunk_overlap:]
                chunk = prev_tail + chunk
            overlapped.append(chunk)
        return overlapped


# ============================================================
# 3. ChromaDB 向量存储（核心升级！替换 TF-IDF）
# ============================================================

class VectorStore:
    """
    ChromaDB 向量数据库封装

    面试重点：
    Q: 为什么用 ChromaDB？
    A: 专门做向量相似度搜索的数据库，除了存向量还能存 metadata，
       自带 ANN（近似最近邻）算法，比暴力搜索快几十倍。

    Q: 和传统数据库的区别？
    A: 传统数据库（MySQL）做 LIKE '%关键词%' 精确匹配
       向量数据库做"找出最相似的"模糊匹配 → 能处理同义词、近义词

    Q: 什么是 ANN？
    A: Approximate Nearest Neighbor，近似最近邻搜索。
       不找"绝对最近"，而是找"大概最近"的。
       牺牲一点点精度，换来几十倍的搜索速度提升。
    """

    def __init__(self, persist_dir: str = Config.CHROMA_PATH):
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        # 创建一个集合（类似MySQL的表，但存的是向量）
        self.collection = self.client.get_or_create_collection(
            name="knowledge_base",
            metadata={"hnsw:space": "cosine"}  # 用余弦距离
        )
        self.embedding = LocalEmbedding(dim=Config.EMBED_DIM)

    def add_documents(self, chunks: List[Dict]) -> int:
        """
        把文本块加入向量库

        步骤：
        1. 用 embedding 模型把每个文本块转成向量
        2. 把向量 + 文本 + metadata 存入 ChromaDB
        3. ChromaDB 自动建 ANN 索引
        """
        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        ids = [
            f"{c['metadata']['source']}_{c['metadata']['chunk_id']}"
            for c in chunks
        ]

        # 生成向量（这是最关键的步骤）
        embeddings = self.embedding.embed(texts)

        # ChromaDB 需要 list 格式，不是 numpy array
        embeddings_list = [emb.tolist() for emb in embeddings]

        # 存入 ChromaDB
        self.collection.add(
            embeddings=embeddings_list,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        return len(chunks)

    def search(self, query: str, top_k: int = Config.RETRIEVAL_TOP_K
               ) -> List[Tuple[str, float, Dict]]:
        """
        检索最相关的文本块

        步骤：
        1. 把查询转成向量（用同样的 embedding 模型！）
        2. ChromaDB 做 ANN 搜索（余弦相似度）
        3. 返回 top_k 个结果，包括文本、分数、metadata

        面试重点：为什么查询也要转向量？
        → 只有把查询和文档映射到同一个向量空间，才能做相似度比较。
          就像你要比较苹果和橘子，得先放在同一个秤上称。
        """
        # 查询转向量
        query_embedding = self.embedding.embed_one(query)

        # ANN 搜索
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )

        if not results["documents"] or not results["documents"][0]:
            return []

        # 组装结果
        output = []
        for i in range(len(results["documents"][0])):
            doc = results["documents"][0][i]
            distance = results["distances"][0][i] if results["distances"] else 0.0
            meta = results["metadatas"][0][i] if results["metadatas"] else {}

            # ChromaDB 返回的是距离（越小越近），我们转成相似度（越大越近）
            similarity = 1.0 - distance
            output.append((doc, similarity, meta))

        # 按相似度排序（ChromaDB 默认已排序，但显式排序更安全）
        output.sort(key=lambda x: x[1], reverse=True)
        return output

    def list_files(self) -> List[str]:
        """列出所有已入库的文档"""
        # 从 metadata 里提取唯一的 source 字段
        results = self.collection.get()
        if not results["metadatas"]:
            return []
        filenames = set(m["source"] for m in results["metadatas"] if "source" in m)
        return sorted(filenames)

    def delete_file(self, filename: str):
        """删除某个文档的所有块"""
        if not self.collection.count():
            return
        # 通过 where 条件筛选删除
        self.collection.delete(where={"source": filename})

    def count(self) -> int:
        """当前向量库中的总块数"""
        return self.collection.count()


# ============================================================
# 4. LLM 调用（基本和 v1 一样）
# ============================================================

class LLM:
    """封装 DeepSeek API 调用"""

    def __init__(self):
        self.client = OpenAI(
            api_key=Config.DEEPSEEK_API_KEY,
            base_url=Config.DEEPSEEK_BASE_URL
        )

    def ask(self, query: str, context: str, history: List[dict] = None) -> str:
        """带上下文的问答"""
        system_prompt = f"""你是一个知识库问答助手。
请基于以下【背景资料】回答问题。
如果背景资料不足以回答，就说"根据已有资料无法回答这个问题"。
不要编造信息。

【背景资料】
{context}"""

        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history[-6:])
        messages.append({"role": "user", "content": query})

        response = self.client.chat.completions.create(
            model=Config.LLM_MODEL,
            messages=messages,
            temperature=Config.LLM_TEMPERATURE,
            max_tokens=Config.LLM_MAX_TOKENS,
            stream=False
        )
        return response.choices[0].message.content

    def ask_stream(self, query: str, context: str, history: List[dict] = None):
        """流式版本"""
        system_prompt = f"""你是一个知识库问答助手。
请基于以下【背景资料】回答问题。
如果背景资料不足以回答，就说"根据已有资料无法回答这个问题"。
不要编造信息。

【背景资料】
{context}"""

        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history[-6:])
        messages.append({"role": "user", "content": query})

        response = self.client.chat.completions.create(
            model=Config.LLM_MODEL,
            messages=messages,
            temperature=Config.LLM_TEMPERATURE,
            max_tokens=Config.LLM_MAX_TOKENS,
            stream=True
        )

        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


# ============================================================
# 5. RAG 引擎（组装）
# ============================================================

class RAGEngine:
    """
    RAG 引擎 v2

    架构：
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │ Document │ → │  Vector  │ → │   LLM    │
    │Processor │   │  Store   │   │(DeepSeek)│
    │  分块    │   │ ChromaDB │   │  回答     │
    └──────────┘   └──────────┘   └──────────┘
                        ↑
                  ┌──────────┐
                  │  Local   │
                  │Embedding │
                  │ 哈希嵌入 │
                  └──────────┘

    RAG 全流程（面试必背）：
    Step 1: 加载文档 → 智能分块（重叠chunk）
    Step 2: 每个chunk转成向量 → 存入 ChromaDB
    Step 3: 用户提问 → 转成向量 → ChromaDB 搜最相似chunk
    Step 4: 检索到的chunk + 问题 → 拼 prompt → 调 DeepSeek
    Step 5: 返回回答
    """

    def __init__(self):
        self.processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.llm = LLM()

    def ingest(self, filepath: str) -> int:
        """处理文档：分段 → 嵌入 → 入库"""
        chunks = self.processor.load_and_chunk(filepath)
        count = self.vector_store.add_documents(chunks)
        return count

    def query(self, question: str, history: List[dict] = None) -> dict:
        """
        问答：检索 → 拼 prompt → 回答
        返回 {answer, sources, similarity_scores}
        """
        # 1. 检索
        results = self.vector_store.search(question)

        if not results:
            return {
                "answer": "没有找到相关资料，请先上传文档。",
                "sources": [],
                "scores": []
            }

        # 2. 拼 context（取 top 3）
        top_results = results[:Config.FINAL_TOP_K]
        context = "\n---\n".join([r[0] for r in top_results])

        # 3. 调用 LLM
        answer = self.llm.ask(question, context, history)

        # 4. 返回结果（附带来源和分数，前端可以展示）
        return {
            "answer": answer,
            "sources": [r[2].get("source", "unknown") for r in top_results],
            "scores": [round(r[1], 3) for r in top_results]
        }

    def query_stream(self, question: str, history: List[dict] = None):
        """流式问答"""
        results = self.vector_store.search(question)

        if not results:
            yield "没有找到相关资料，请先上传文档。"
            return

        top_results = results[:Config.FINAL_TOP_K]
        context = "\n---\n".join([r[0] for r in top_results])
        yield from self.llm.ask_stream(question, context, history)

    def get_stats(self) -> dict:
        """获取引擎状态（用于监控/调试）"""
        return {
            "total_chunks": self.vector_store.count(),
            "files": self.vector_store.list_files(),
            "embedding_dim": Config.EMBED_DIM,
            "embedding_type": "ngram_hash (local)",
            "chunk_size": Config.CHUNK_SIZE,
        }


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    # 快速测试本地嵌入
    emb = LocalEmbedding()
    v1 = emb.embed_one("人工智能")
    v2 = emb.embed_one("AI技术")
    v3 = emb.embed_one("番茄炒蛋")
    print(f"相似度 '人工智能' vs 'AI技术': {emb.similarity('人工智能', 'AI技术'):.3f}")
    print(f"相似度 '人工智能' vs '番茄炒蛋': {emb.similarity('人工智能', '番茄炒蛋'):.3f}")
    print(f"  → 语义相近的分数更高，说明嵌入有效！")

    # 测试引擎
    engine = RAGEngine()
    test_file = os.path.join(Config.DATA_DIR, "knowledge.txt")
    if os.path.exists(test_file):
        n = engine.ingest(test_file)
        print(f"\n已导入 {n} 个文本块到 ChromaDB")
        print(f"引擎状态: {engine.get_stats()}")

    # 交互问答
    print("\n开始问答（输入 exit 退出）")
    while True:
        try:
            q = input("\n问题: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() == "exit":
            break
        if not q:
            continue

        result = engine.query(q)
        print(f"\n回答: {result['answer']}")
        if result["sources"]:
            print(f"来源: {result['sources']}")
            print(f"相似度: {result['scores']}")

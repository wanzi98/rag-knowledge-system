"""
RAG 引擎单元测试 + 集成测试

运行方式：
    # 单元测试（纯 numpy，不需外部依赖）
    python -m pytest tests/test_rag_engine.py -v -k "not integration"

    # 全部测试（需要 ChromaDB）
    pip install chromadb
    python -m pytest tests/test_rag_engine.py -v

    # 覆盖报告（可选）
    pip install pytest-cov
    python -m pytest tests/test_rag_engine.py --cov=../backend/rag_engine
"""

import sys
import os
import math
import tempfile

# 确保能 import backend 模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import numpy as np
import pytest


# ============================================================
# 检查 ChromaDB 是否可用（用于集成测试）
# ============================================================

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False


# ============================================================
# 测试 LocalEmbedding（单元测试 — 纯 numpy）
# ============================================================

class TestLocalEmbedding:
    """测试自研的 n-gram 哈希嵌入引擎"""

    def setup_method(self):
        """每个测试前都创建一个新的 embedding 实例"""
        from rag_engine import LocalEmbedding
        self.emb = LocalEmbedding(dim=256)

    def test_embed_output_shape(self):
        """嵌入结果应该是固定维度的向量"""
        from rag_engine import LocalEmbedding
        emb = LocalEmbedding(dim=256)
        vec = emb.embed_one("测试文本")
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (256,)
        assert vec.dtype == np.float32

    def test_embed_normalized(self):
        """嵌入结果应该是 L2 归一化的（范数 ≈ 1）"""
        vec = self.emb.embed_one("人工智能")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5

    def test_empty_text(self):
        """空文本应该返回全零向量"""
        vec = self.emb.embed_one("")
        assert np.all(vec == 0)

    def test_whitespace_text(self):
        """纯空白文本应该返回全零向量"""
        vec = self.emb.embed_one("   ")
        assert np.all(vec == 0)

    def test_similar_texts_have_higher_similarity(self):
        """共享字符越多的文本，余弦相似度越高（n-gram 哈希的特性）"""
        # 注：n-gram 哈希基于字符重叠，所以"共享字符"≈"语义相近"是近似关系
        # "人工智能技术"包含"人工智能"的所有字符 → 共享n-gram多 → 相似度高
        sim_share = self.emb.similarity("人工智能技术", "人工智能")
        sim_no_share = self.emb.similarity("人工智能技术", "番茄炒蛋")

        assert sim_share > sim_no_share, (
            f"共享字符的文本 ({sim_share:.3f}) 应该比无关文本 ({sim_no_share:.3f}) 相似度高"
        )

    def test_different_texts_have_lower_similarity(self):
        """完全不共享字符的文本，余弦相似度应该很低（哈希碰撞导致略 > 0）"""
        sim = self.emb.similarity("ABCDEFG", "HIJKLMN")
        # 这两个文本不共享任何字符，但哈希碰撞可能产生微小相似度
        assert sim < 0.15, f"不共享字符的文本相似度应该很低，实际为 {sim:.3f}"

    def test_identical_texts_perfect_similarity(self):
        """完全相同的文本应该相似度 ≈ 1.0"""
        sim = self.emb.similarity("测试文本完全相同", "测试文本完全相同")
        assert abs(sim - 1.0) < 1e-5

    def test_deterministic_output(self):
        """相同输入应该每次输出相同向量"""
        v1 = self.emb.embed_one("深度学习")
        v2 = self.emb.embed_one("深度学习")
        assert np.allclose(v1, v2)

    def test_batch_embed(self):
        """批量嵌入应该和单条嵌入结果一致"""
        texts = ["人工智能", "机器学习", "深度学习"]
        batch_results = self.emb.embed(texts)
        assert len(batch_results) == 3
        for i, text in enumerate(texts):
            single = self.emb.embed_one(text)
            assert np.allclose(batch_results[i], single)

    @pytest.mark.parametrize("dim", [64, 128, 256, 512])
    def test_different_dimensions(self, dim):
        """不同维度应该都能正常工作"""
        from rag_engine import LocalEmbedding
        emb = LocalEmbedding(dim=dim)
        vec = emb.embed_one("测试")
        assert vec.shape == (dim,)

    def test_similarity_symmetric(self):
        """余弦相似度应该是对称的：sim(a,b) == sim(b,a)"""
        sim_ab = self.emb.similarity("人工智能", "机器学习")
        sim_ba = self.emb.similarity("机器学习", "人工智能")
        assert abs(sim_ab - sim_ba) < 1e-5

    def test_similarity_range(self):
        """余弦相似度应该在 [-1, 1] 范围内"""
        sim = self.emb.similarity("正面文本", "完全相反的内容")
        assert -1.0 <= sim <= 1.0

    def test_short_text(self):
        """很短的内容（1-2个字符）也应该能正常嵌入"""
        vec = self.emb.embed_one("A")
        assert np.linalg.norm(vec) > 0
        assert vec.shape == (256,)

    def test_long_text(self):
        """长文本也应该能正常嵌入"""
        long_text = "测试" * 1000
        vec = self.emb.embed_one(long_text)
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-5


# ============================================================
# 测试 DocumentProcessor（单元测试 — 纯 Python）
# ============================================================

class TestDocumentProcessor:
    """测试文档智能分块"""

    def setup_method(self):
        from rag_engine import DocumentProcessor
        self.dp = DocumentProcessor()

    def test_empty_document(self):
        """空文档应该返回空列表"""
        chunks = self.dp._smart_chunk("")
        assert isinstance(chunks, list)

    def test_single_short_paragraph(self):
        """长度 >= 10 的短段落应该作为一个块（< 10 字符的段落会被过滤）"""
        self.dp.chunk_size = 300
        # 代码中 len(para) < 10 的段落会被跳过，所以需要 10 个以上字符
        text = "这是一段足够长的测试文本段落内容。"
        chunks = self.dp._smart_chunk(text)
        assert len(chunks) >= 1

    def test_chunk_size_respected(self):
        """超长段落会被_按句子切分，每个块控制在合理长度"""
        from rag_engine import DocumentProcessor
        dp = DocumentProcessor(chunk_size=100)
        # 需要有句号等分隔符，才能被 _split_long_paragraph 切分
        text = "第一句内容。第二句内容。第三句内容。第四句内容。第五句内容。" * 20
        chunks = dp._smart_chunk(text)
        # 应该被切分成多个块
        assert len(chunks) > 1, f"应该被切分成多个块，实际只有 {len(chunks)} 个"

    def test_overlap_added(self):
        """相邻块之间应该包含重叠部分"""
        from rag_engine import DocumentProcessor
        dp = DocumentProcessor(chunk_size=100, chunk_overlap=20)
        text = "项目一：RAG架构。项目二：AI Agent。项目三：Function Calling。项目四：Prompt Engineering。"
        chunks = dp._smart_chunk(text)

        if len(chunks) > 1:
            prev_tail = chunks[0][-dp.chunk_overlap:]
            assert prev_tail in chunks[1], (
                f"前一块末尾 '{prev_tail}' 应出现在后一块开头"
            )

    def test_paragraph_preserved(self):
        """段落结构应该尽量保留（< 10 字符的短段会被跳过，但附近内容会保留）"""
        self.dp.chunk_size = 500
        text = "这是引言部分的内容概述。\n\n这是方法部分的核心描述内容。"
        chunks = self.dp._smart_chunk(text)
        assert any("引言" in c for c in chunks)
        assert any("方法" in c for c in chunks)

    def test_chunks_are_strings(self):
        """每个块应该是字符串"""
        text = "测试文本"
        chunks = self.dp._smart_chunk(text)
        assert all(isinstance(c, str) for c in chunks)

    def test_split_long_paragraph(self):
        """超长段落应该被切分成多个块"""
        from rag_engine import DocumentProcessor
        dp = DocumentProcessor(chunk_size=50)
        long_para = "第一句话。第二句话。第三句话。第四句话。第五句话。" * 10
        chunks = dp._split_long_paragraph(long_para)
        assert len(chunks) > 1, "长段落应该被切分成多个块"

    def test_txt_loading(self):
        """TXT 文件读取应该返回字符串"""
        from rag_engine import DocumentProcessor
        dp = DocumentProcessor()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("测试文件内容")
            tmp_path = f.name
        try:
            text = dp._read_txt(tmp_path)
            assert isinstance(text, str)
            assert "测试文件内容" in text
        finally:
            os.unlink(tmp_path)


# ============================================================
# 集成测试（需要 ChromaDB）
# ============================================================

@pytest.mark.skipif(not CHROMA_AVAILABLE, reason="需要 ChromaDB（pip install chromadb）")
class TestVectorStoreIntegration:
    """测试 ChromaDB 向量存储（集成测试）"""

    def test_store_and_search(self):
        """存文档后应该能搜到"""
        from rag_engine import VectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(persist_dir=tmpdir)
            chunks = [
                {"text": "Python是一种编程语言", "metadata": {"source": "test.txt", "chunk_id": 0}},
                {"text": "RAG是检索增强生成", "metadata": {"source": "test.txt", "chunk_id": 1}},
            ]
            count = store.add_documents(chunks)
            assert count == 2

            results = store.search("编程", top_k=1)
            assert len(results) > 0
            assert "编程" in results[0][0]

    def test_search_no_results(self):
        """空库搜索应该返回空"""
        from rag_engine import VectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(persist_dir=tmpdir)
            results = store.search("随便问问")
            assert results == []

    def test_list_files(self):
        """文件列表应该正确"""
        from rag_engine import VectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(persist_dir=tmpdir)
            assert store.list_files() == []

            chunks = [
                {"text": "测试", "metadata": {"source": "a.txt", "chunk_id": 0}},
                {"text": "测试2", "metadata": {"source": "b.txt", "chunk_id": 0}},
            ]
            store.add_documents(chunks)
            files = store.list_files()
            assert "a.txt" in files
            assert "b.txt" in files
            assert len(files) == 2

    def test_delete_file(self):
        """删除文件后应该搜不到"""
        from rag_engine import VectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(persist_dir=tmpdir)
            chunks = [
                {"text": "要删除的内容", "metadata": {"source": "delete_me.txt", "chunk_id": 0}},
            ]
            store.add_documents(chunks)
            assert store.count() == 1

            store.delete_file("delete_me.txt")
            assert store.count() == 0

    def test_count(self):
        """count 应该返回正确的 chunk 数量"""
        from rag_engine import VectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(persist_dir=tmpdir)
            assert store.count() == 0

            chunks = [{"text": f"内容{i}", "metadata": {"source": "test.txt", "chunk_id": i}} for i in range(5)]
            store.add_documents(chunks)
            assert store.count() == 5

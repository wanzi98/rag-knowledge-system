"""
pytest 配置：在导入 rag_engine 前 mock 外部依赖，
使单元测试可以不依赖真实 API / 数据库运行。

Mock 清单：
- chromadb（向量数据库）
- openai（LLM API 客户端）
"""
import sys
import unittest.mock


# ============================================================
# Mock chromadb
# ============================================================

class _MockCollection:
    """模拟 ChromaDB Collection"""
    def add(self, embeddings=None, documents=None, metadatas=None, ids=None):
        return None

    def query(self, query_embeddings=None, n_results=None):
        return {"documents": [], "distances": [], "metadatas": [], "ids": []}

    def get(self):
        return {"metadatas": []}

    def delete(self, where=None):
        return None

    def count(self):
        return 0


class _MockPersistentClient:
    """模拟 ChromaDB PersistentClient"""
    def __init__(self, path=None, settings=None):
        pass

    def get_or_create_collection(self, name=None, metadata=None):
        return _MockCollection()

    def get_collection(self, name=None):
        return _MockCollection()


class _MockSettings:
    def __init__(self, anonymized_telemetry=False):
        pass


mock_chromadb = unittest.mock.MagicMock()
mock_chromadb.PersistentClient = _MockPersistentClient
mock_chromadb.Settings = _MockSettings
mock_chromadb.config.Settings = _MockSettings

sys.modules['chromadb'] = mock_chromadb
sys.modules['chromadb.config'] = mock_chromadb.config


# ============================================================
# Mock openai
# ============================================================

class _MockMessage:
    def __init__(self, content="模拟回答"):
        self.content = content
        self.tool_calls = []


class _MockChoice:
    def __init__(self):
        self.message = _MockMessage()
        self.finish_reason = "stop"
        self.delta = _MockDelta()


class _MockDelta:
    def __init__(self):
        self.content = None


class _MockCompletion:
    def __init__(self):
        self.choices = [_MockChoice()]


class _MockStreamChunk:
    def __init__(self):
        self.choices = [_MockStreamChoice()]


class _MockStreamChoice:
    def __init__(self):
        self.delta = _MockDelta()
        self.finish_reason = None


class _MockChat:
    def __init__(self):
        self.completions = _MockCompletions()


class _MockCompletions:
    def create(self, *args, **kwargs):
        if kwargs.get("stream"):
            return iter([_MockStreamChunk()])
        return _MockCompletion()


class _MockOpenAI:
    def __init__(self, api_key=None, base_url=None):
        self.chat = _MockChat()
        self.api_key = api_key
        self.base_url = base_url


mock_openai = unittest.mock.MagicMock()
mock_openai.OpenAI = _MockOpenAI

sys.modules['openai'] = mock_openai


# ============================================================
# Mock 其他可能缺失的库
# ============================================================

for mod_name in ['PyPDF2', 'docx']:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = unittest.mock.MagicMock()

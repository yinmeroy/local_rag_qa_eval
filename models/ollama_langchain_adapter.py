# models/ollama_langchain_adapter.py
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from typing import List, Optional, Any
from concurrent.futures import ThreadPoolExecutor
import asyncio
from models.ollama_client import OllamaClient

# 全局共享的线程池执行器
_executor = ThreadPoolExecutor(max_workers=8)


class OllamaChatModel(BaseChatModel):
    """将 OllamaClient 适配为 LangChain BaseChatModel 接口"""
    
    client: OllamaClient
    model_name: str = "local_ollama"
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> ChatResult:
        """实现 LangChain 的 _generate 方法"""
        # 将 LangChain messages 转换为 Ollama 格式
        ollama_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                ollama_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                ollama_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                ollama_messages.append({"role": "assistant", "content": msg.content})
        
        # 调用 Ollama
        try:
            import ollama
            from config.settings import OLLAMA_HOST
            import os
            os.environ["OLLAMA_HOST"] = OLLAMA_HOST
            
            response = ollama.chat(
                model=self.client.llm_model,
                messages=ollama_messages
            )
            content = response["message"]["content"]
            
            # 返回 LangChain 格式
            generation = ChatGeneration(
                message=AIMessage(content=content),
                text=content
            )
            return ChatResult(generations=[generation])
        except Exception as e:
            raise Exception(f"Ollama 调用失败：{str(e)}")
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> ChatResult:
        """实现 LangChain 的异步 _agenerate 方法（RAGAS 需要）"""
        # 使用全局线程池执行同步调用
        loop = asyncio.get_event_loop()
        
        try:
            # 增加超时时间到 300 秒（5 分钟），适配本地 Ollama 模型的慢响应
            # 本地模型通常需要较长时间，RAGAS 默认超时太短
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    _executor,
                    lambda: self._generate(messages, stop, run_manager, **kwargs)
                ),
                timeout=300  # 5 分钟超时
            )
            return result
        except asyncio.TimeoutError:
            # 超时错误，提示用户
            error_msg = f"⏱️ LLM 调用超时（超过 300 秒）\n请检查：\n1. Ollama 服务是否正常运行\n2. 模型是否正在加载（首次调用可能较慢）\n3. 系统资源是否充足"
            print(error_msg)
            raise TimeoutError(error_msg)
        except asyncio.CancelledError:
            # 任务被取消（通常是外部超时机制触发）
            print(f"⚠️ LLM 调用被取消")
            raise
        except Exception as e:
            print(f"❌ _agenerate 调用失败：{e}")
            raise
    
    @property
    def _llm_type(self) -> str:
        return "ollama_chat"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        """兼容旧的 _call 方法"""
        messages = [HumanMessage(content=prompt)]
        result = self._generate(messages, stop=stop, **kwargs)
        return result.generations[0].text
    
    def call_as_llm(self, message: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        """RAGAS 兼容方法：直接调用 LLM 获取回复"""
        return self._call(message, stop=stop, **kwargs)


class OllamaEmbeddings(Embeddings):
    """将 OllamaClient 适配为 LangChain Embeddings 接口"""
    
    client: OllamaClient
    
    def __init__(self, client: OllamaClient):
        super().__init__()
        self.client = client
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量获取嵌入向量"""
        embeddings = []
        for i, text in enumerate(texts):
            try:
                embedding = self.client.get_embedding(text)
                if not embedding:
                    raise ValueError(f"文档 {i+1} 的嵌入向量为空")
                embeddings.append(embedding)
            except Exception as e:
                # 根据规范：关键数据获取失败必须抛出异常，不能返回占位符
                raise Exception(f"嵌入文档 {i+1}/{len(texts)} 失败：{e}")
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """获取单个查询的嵌入向量"""
        try:
            embedding = self.client.get_embedding(text)
            if not embedding:
                raise ValueError("嵌入向量为空")
            return embedding
        except Exception as e:
            # 根据规范：关键数据获取失败必须抛出异常
            raise Exception(f"嵌入查询失败：{e}")

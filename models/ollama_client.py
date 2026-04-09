import os
import requests
import ollama
from config.settings import OLLAMA_HOST

# 全局设置Ollama地址
os.environ["OLLAMA_HOST"] = OLLAMA_HOST

class OllamaClient:
    """封装Ollama调用，支持懒加载+未下载模型提示"""
    def __init__(self, llm_model: str, embedding_model: str):
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        
        # 关键修复：检测时必须使用与实际调用一致的匹配策略
        # Ollama API 需要精确匹配，不能只匹配基础名称
        self.llm_exists = self._check_model_exists(llm_model)
        self.embedding_exists = self._check_model_exists(embedding_model)

    def _check_model_exists(self, model_name: str) -> bool:
        """
        检测模型是否已下载到本地
        重要：必须与实际调用时的匹配策略一致
        Ollama API 调用需要精确匹配或能自动补全标签
        """
        try:
            response = requests.get(f"{OLLAMA_HOST}/api/tags")
            if response.status_code != 200:
                print(f"⚠️  Ollama API 返回异常状态码: {response.status_code}")
                return False
            
            models = response.json().get("models", [])
            full_names = [m["name"] for m in models]
            
            # 打印所有已下载的模型（调试信息）
            print(f"\n📦 Ollama 已下载模型: {full_names}")
            print(f"🔍 检查模型 '{model_name}' 是否存在")
            
            # 策略 1: 精确匹配（用户输入完整名称）
            if model_name in full_names:
                print(f"   ✅ 精确匹配成功")
                return True
            
            # 策略 2: 自动补全 :latest 标签
            # 如果用户输入 "all-minilm"，检查是否存在 "all-minilm:latest"
            if not model_name.endswith(":"):
                latest_name = f"{model_name}:latest"
                if latest_name in full_names:
                    print(f"   ✅ 自动补全标签匹配: {latest_name}")
                    return True
            
            # 策略 3: 如果用户只输入基础名称，检查是否有任意版本
            # 但这种情况应该提示用户明确指定版本，而不是返回 True
            input_base_name = model_name.split(":")[0].strip()
            matched_versions = [name for name in full_names if name.split(":")[0] == input_base_name]
            
            if matched_versions:
                print(f"   ⚠️  找到相关版本: {matched_versions}")
                print(f"   ❌ 但未找到精确匹配 '{model_name}'")
                print(f"   💡 请使用完整名称，例如: {matched_versions[0]}")
                return False
            
            print(f"   ❌ 未找到任何匹配模型")
            return False
        except Exception as e:
            print(f"⚠️  检查模型失败: {e}")
            return False

    def _get_download_command(self, model_name: str) -> str:
        """返回模型的下载命令"""
        return f"请在终端执行命令下载模型：\nollama pull {model_name}"

    def get_embedding(self, text: str) -> list[float]:
        """获取文本嵌入向量（懒加载，未下载则提示）"""
        if not self.embedding_exists:
            raise ValueError(f"嵌入模型未下载：{self.embedding_model}\n请执行：ollama pull {self.embedding_model}")
        
        if not text or not text.strip():
            raise ValueError("文本内容不能为空")
        
        try:
            # 使用 Ollama 官方 API 获取嵌入向量
            resp = ollama.embeddings(model=self.embedding_model, prompt=text.strip())
            embedding = resp.get("embedding", [])
            
            if not embedding:
                raise ValueError(f"未能获取嵌入向量，模型：{self.embedding_model}")
            
            return embedding
        except Exception as e:
            # 抛出异常而不是返回空列表，让上层正确处理
            raise Exception(f"获取嵌入向量失败（模型：{self.embedding_model}）：{str(e)}")

    def batch_get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """批量获取嵌入向量"""
        return [self.get_embedding(text) for text in texts]

    def chat(self, prompt: str, chat_history: list[tuple[str, str]] = None) -> str:
        """对话调用（懒加载，未下载则提示）"""
        if not self.llm_exists:
            return self._get_download_command(self.llm_model)
        
        if chat_history is None:
            chat_history = []
        
        messages = []
        for human, ai in chat_history:
            messages.append({"role": "user", "content": human})
            messages.append({"role": "assistant", "content": ai})
        messages.append({"role": "user", "content": prompt})

        try:
            resp = ollama.chat(model=self.llm_model, messages=messages)
            return resp["message"]["content"]
        except Exception as e:
            return f"模型调用失败：{str(e)}"
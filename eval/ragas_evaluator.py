# eval/ragas_evaluator.py
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy
)
from datasets import Dataset, Features, Sequence, Value
from typing import List, Dict
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


class RagasEvaluator:
    """RAGAS 评测封装（使用 OpenAI 兼容 API）"""
    
    def __init__(self, api_base_url: str = "https://4zapi.com/v1", api_key: str = None):
        """
        初始化评测器
        
        Args:
            api_base_url: API 基础 URL（默认使用 4zapi）
            api_key: API Key（如果环境变量已设置可忽略）
        """
        self.api_base_url = api_base_url
        self.api_key = api_key
        
        # 打印调试信息
        print(f"\n🔧 初始化 RAGAS 评测器...")
        print(f"📡 API 地址：{api_base_url}")
        print(f"🔑 API Key：{'已设置' if api_key else '使用占位符'}")
        
        # 初始化 LLM 和 Embeddings（使用 OpenAI 兼容接口）
        # 注意：langchain-openai 0.1.x 版本中
        # - ChatOpenAI 使用 base_url
        # - OpenAIEmbeddings 使用 openai_api_base
        self.llm = ChatOpenAI(
            base_url=api_base_url,
            api_key=api_key if api_key else "sk-placeholder",
            temperature=0.0,
            model="gpt-4o",
        )
        
        self.embeddings = OpenAIEmbeddings(
            openai_api_base=api_base_url,  # 使用 openai_api_base 而不是 base_url
            openai_api_key=api_key if api_key else "sk-placeholder",
            model="text-embedding-ada-002",
        )
        
        print(f"✅ LLM 模型：{getattr(self.llm, 'model_name', getattr(self.llm, 'model', 'unknown'))}")
        print(f"✅ 嵌入模型：{getattr(self.embeddings, 'model', 'unknown')}")
    
    def evaluate_qa_pairs(self, qa_data: List[Dict]) -> Dict:
        """
        评测问答对
        
        Args:
            qa_data: 问答数据列表，格式：
                [
                    {
                        "question": "问题",
                        "answer": "模型回答",
                        "contexts": ["检索到的上下文片段"],
                        "ground_truth": "标准答案（可选）"
                    }
                ]
        
        Returns:
            评测结果字典
        """
        if not qa_data:
            return {"error": "没有评测数据"}
        
        # 打印调试信息
        print(f"\n🔍 开始 RAGAS 评测...")
        print(f"📊 评测数据量：{len(qa_data)} 个问答对")
        print(f"🤖 使用 LLM 模型：{self.llm.model_name if hasattr(self.llm, 'model_name') else self.llm.model}")
        print(f"🔧 使用嵌入模型：{self.embeddings.model if hasattr(self.embeddings, 'model') else 'unknown'}")
        
        # 构建评测数据集
        dataset_dict = {
            "question": [item["question"] for item in qa_data],
            "answer": [item["answer"] for item in qa_data],
            "contexts": [[str(ctx) for ctx in item["contexts"]] for item in qa_data]
        }
        
        # 打印调试信息：检查数据格式
        print(f"\n📋 数据样本（第1条）：")
        print(f"   问题：{dataset_dict['question'][0][:100]}...")
        print(f"   回答：{dataset_dict['answer'][0][:100]}...")
        print(f"   上下文数量：{len(dataset_dict['contexts'][0])} 条")
        if dataset_dict['contexts'][0]:
            ctx_len = len(dataset_dict['contexts'][0][0])
            print(f"   上下文1长度：{ctx_len} 字符")
            print(f"   上下文1示例：{dataset_dict['contexts'][0][0][:100]}...")
        else:
            print(f"   ⚠️ 上下文为空！这将导致大部分指标为0！")
            print(f"   🔍 请检查：检索模块是否正常工作？")
        
        # 检查是否有标准答案（需要先检查）
        has_ground_truth = all(item.get("ground_truth") for item in qa_data)
        
        if has_ground_truth:
            dataset_dict["ground_truth"] = [item["ground_truth"] for item in qa_data]
            print(f"✅ 检测到标准答案，将计算 Context Recall")
        else:
            print(f"⚠️  未提供标准答案，跳过 Context Recall 指标")
            print(f"   标准答案：{dataset_dict['ground_truth'][0][:100]}...")
        
        # 显式指定 features，避免类型推断错误
        features = Features({
            "question": Value("string"),
            "answer": Value("string"),
            "contexts": Sequence(Value("string")),
        })
        if has_ground_truth:
            features["ground_truth"] = Value("string")
        
        dataset = Dataset.from_dict(dataset_dict, features=features)
        print(f"✅ 数据集构建成功")
        
        # 选择评测指标
        metrics = [context_precision, faithfulness, answer_relevancy]
        if has_ground_truth:
            metrics.append(context_recall)
        
        try:
            # 执行评测（使用 OpenAI 兼容 API）
            print(f"🚀 开始执行评测（使用 API 模式，速度更快）...")
            
            # 使用 API 进行评测，完全避免本地模型的超时问题
            result = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=self.llm,
                embeddings=self.embeddings,
                raise_exceptions=False,  # 忽略单个样本的错误，继续执行
            )
            
            print(f"✅ 评测完成")
            
            return {
                "context_precision": float(result["context_precision"]),
                "faithfulness": float(result["faithfulness"]),
                "answer_relevancy": float(result["answer_relevancy"]),
                "context_recall": float(result["context_recall"]) if has_ground_truth else "需要标准答案"
            }
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"❌ 评测详细错误：\n{error_detail}")
            return {"error": f"评测执行失败：{str(e)}\n\n请检查：\n1. Ollama 服务是否正常运行\n2. 模型是否已下载\n3. 问答数据是否有效"}
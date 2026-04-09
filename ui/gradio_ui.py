# ui/gradio_ui.py
import gradio as gr
import os
import shutil
from config.settings import (
    GRADIO_HOST, GRADIO_PORT, GRADIO_SHARE,
    SUPPORTED_LLM_MODELS, SUPPORTED_EMBEDDING_MODELS,
    DEFAULT_LLM_MODEL, DEFAULT_EMBEDDING_MODEL
)
from document.doc_processor import doc_processor
from document.arxiv_loader import arxiv_loader
from vector_db.faiss_manager import faiss_manager
from models.ollama_client import OllamaClient
from eval.ragas_evaluator import RagasEvaluator
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

os.environ["GRADIO_API_ENABLED"] = "False"


class GradioUI:
    """封装 Gradio 界面（新增模型状态校验 + RAGAS 评测 + Arxiv 支持）"""
    
    def __init__(self):
        self.faiss_manager = faiss_manager
        self.doc_processor = doc_processor
        self.qa_chain = None
        self.ollama_client = None  # 动态初始化 Ollama 客户端
        self.evaluator = None  # RAGAS 评测器
        self.model_available = False
        self.qa_history = []  # 记录问答历史用于评测
        
        # RAG 问答专用模型客户端（可与初始化模型不同）
        self.rag_ollama_client = None
        self.rag_model_available = False

    def init_ollama_client(self, llm_model, embedding_model):
        """初始化 Ollama 客户端（仅检测，不下载）"""
        try:
            self.ollama_client = OllamaClient(llm_model, embedding_model)
            # 校验模型是否都已下载
            self.model_available = self.ollama_client.llm_exists and self.ollama_client.embedding_exists
            
            # 初始化评测器（使用 API，不再依赖本地 Ollama）
            if self.model_available:
                from eval.ragas_evaluator import RagasEvaluator
                # 使用 API 进行评测，避免本地模型超时问题
                self.evaluator = RagasEvaluator(
                    api_base_url="https://your-api-base-url/v1", # 【必填】修改为你的官方地址或中转站地址
                    api_key="sk-your-real-api-key-here"  # 【必填】修改为你自己的 API Key
                )
                print("✅ RAGAS 评测器初始化成功（使用 API 模式）")
            
            # 拼接检测结果提示
            llm_status = "✅ 已下载" if self.ollama_client.llm_exists else "❌ 未下载"
            embed_status = "✅ 已下载" if self.ollama_client.embedding_exists else "❌ 未下载"
            
            if self.model_available:
                return f"模型检测完成：\nLLM({llm_model}) {llm_status}\n嵌入 ({embedding_model}) {embed_status}\n✅ 所有模型已就绪，可上传文件\n"
            else:
                download_cmds = []
                if not self.ollama_client.llm_exists:
                    download_cmds.append(f"ollama pull {llm_model}")
                if not self.ollama_client.embedding_exists:
                    download_cmds.append(f"ollama pull {embedding_model}")
                return f"模型检测完成：\nLLM({llm_model}) {llm_status}\n嵌入 ({embedding_model}) {embed_status}\n❌ 请先下载模型：\n{chr(10).join(download_cmds)}"
        except Exception as e:
            self.model_available = False
            self.evaluator = None
            return f"模型检测失败：{str(e)}"

    def upload_file(self, file_obj, chatbot):
        """上传文件：新增模型状态校验"""
        if not self.ollama_client:
            return "请先选择并初始化模型！", chatbot
        if not self.model_available:
            return "❌ 所选模型未全部下载，无法上传文件！请先下载模型后重新初始化", chatbot
        
        # 1. 自动清空对话 + 清除向量缓存
        chatbot = []
        self.clear_vector_db()
        self.qa_history = []  # 清空问答历史

        if not file_obj:
            return "请选择文件！", chatbot
        
        try:
            file_path = file_obj.name
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                return "文件内容为空，请上传非空文件！", chatbot
            
            # 2. 正确初始化向量库的 Embeddings 实例
            from vector_db.faiss_manager import CustomEmbeddings
            self.faiss_manager.embeddings = CustomEmbeddings(self.ollama_client.get_embedding)
            
            # 3. 构建向量库
            split_docs = self.doc_processor.process_file(file_path)
            self.faiss_manager.add_documents(split_docs)
            self._create_qa_chain()
            
            return "上传成功，可开始提问", chatbot
        except Exception as e:
            return f"上传失败：{str(e)}", chatbot

    def load_arxiv_paper(self, arxiv_id: str, chatbot):
        """加载 Arxiv 论文"""
        if not self.ollama_client:
            return "请先选择并初始化模型！", chatbot
        if not self.model_available:
            return "❌ 所选模型未全部下载，无法加载！请先下载模型后重新初始化", chatbot
        
        if not arxiv_id or not arxiv_id.strip():
            return "请输入 Arxiv ID！", chatbot
        
        try:
            # 清空旧数据
            chatbot = []
            self.clear_vector_db()
            self.qa_history = []  # 清空问答历史
            
            # 加载并处理论文
            split_docs = arxiv_loader.fetch_and_process(arxiv_id.strip())
            
            # 初始化向量库
            from vector_db.faiss_manager import CustomEmbeddings
            self.faiss_manager.embeddings = CustomEmbeddings(self.ollama_client.get_embedding)
            self.faiss_manager.add_documents(split_docs)
            self._create_qa_chain()
            
            return f"✅ Arxiv 论文加载成功：{arxiv_id.strip()}", chatbot
        except Exception as e:
            return f"❌ 加载失败：{str(e)}", chatbot

    def _create_qa_chain(self):
        """创建检索增强问答链"""
        if not self.ollama_client:
            raise ValueError("没有可用的 Ollama 客户端")
        
        active_client = self.ollama_client
        
        ctx_prompt = ChatPromptTemplate.from_messages([
            ("system", "根据对话历史和当前问题，生成独立的检索问题，仅返回问题本身"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        def llm_wrapper(prompt_value):
            messages = prompt_value.to_messages()
            current_input = messages[-1].content
            chat_history = []
            for i in range(0, len(messages)-1, 2):
                if i+1 < len(messages):
                    chat_history.append((messages[i].content, messages[i+1].content))
            return active_client.chat(current_input, chat_history)

        retriever = self.faiss_manager.db.as_retriever(search_kwargs={"k": 3})
        history_retriever = create_history_aware_retriever(llm_wrapper, retriever, ctx_prompt)

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "仅根据上下文回答问题，要求：1. 回复统一使用中文；2. 修正错别字和表述错误；3. 内容完整通顺无截断。无信息则说'未找到相关答案'。上下文：{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        qa_chain = create_stuff_documents_chain(llm_wrapper, qa_prompt)
        self.qa_chain = create_retrieval_chain(history_retriever, qa_chain)

    def chat(self, msg, chat_history):
        """处理对话：新增模型状态校验 + 问答历史记录"""
        # 校验模型是否可用
        if not self.ollama_client:
            chat_history.append((msg, "❌ 请先选择并初始化模型！"))
            return "", chat_history
        if not self.model_available:
            chat_history.append((msg, "❌ 所选模型未全部下载，无法问答！请先下载模型后重新初始化"))
            return "", chat_history
        
        if not self.qa_chain:
            chat_history.append((msg, "❌ 请先上传文件！"))
            return "", chat_history

        try:
            lc_history = []
            for h, a in chat_history:
                lc_history.append(("human", h))
                lc_history.append(("ai", a))

            resp = self.qa_chain.invoke({"input": msg, "chat_history": lc_history})
            answer = resp["answer"]
            
            # 🔍 调试：打印响应的所有字段
            print(f"\n🔍 检索链响应字段：{list(resp.keys())}")
            
            # 获取检索到的上下文（尝试多种字段名）
            contexts = []
            
            # 尝试 'source_documents' 字段
            if "source_documents" in resp and resp["source_documents"]:
                contexts = [doc.page_content for doc in resp["source_documents"] if hasattr(doc, 'page_content')]
                print(f"✅ 从 'source_documents' 获取到 {len(contexts)} 个上下文")
            
            # 尝试 'context' 字段
            if not contexts and "context" in resp:
                context_data = resp["context"]
                if isinstance(context_data, list):
                    contexts = [doc.page_content for doc in context_data if hasattr(doc, 'page_content')]
                elif hasattr(context_data, 'page_content'):
                    contexts = [context_data.page_content]
                print(f"✅ 从 'context' 获取到 {len(contexts)} 个上下文")
            
            # 尝试 'documents' 字段
            if not contexts and "documents" in resp:
                contexts = [doc.page_content for doc in resp["documents"] if hasattr(doc, 'page_content')]
                print(f"✅ 从 'documents' 获取到 {len(contexts)} 个上下文")
            
            # 如果还是没有，尝试遍历所有字段查找文档
            if not contexts:
                for key, value in resp.items():
                    if key not in ["answer", "input", "chat_history"]:
                        if isinstance(value, list) and len(value) > 0:
                            if hasattr(value[0], 'page_content'):
                                contexts = [doc.page_content for doc in value]
                                print(f"✅ 从 '{key}' 获取到 {len(contexts)} 个上下文")
                                break
            
            # 调试：打印上下文信息
            print(f"📝 最终上下文数量：{len(contexts)}")
            if contexts:
                print(f"   上下文1长度：{len(contexts[0])} 字符")
                print(f"   上下文1示例：{contexts[0][:100]}...")
            else:
                print(f"   ⚠️ 上下文为空！")
                print(f"   完整响应结构：")
                for key, value in resp.items():
                    if key != "answer":
                        print(f"     {key}: {type(value).__name__}")
            
            # 记录问答历史用于评测
            self.qa_history.append({
                "question": msg,
                "answer": answer,
                "contexts": contexts
            })

            chat_history.append((msg, answer))
            return "", chat_history
        except Exception as e:
            import traceback
            error_msg = f"问答失败：{str(e)}"
            print(f"\n❌ 问答异常：\n{traceback.format_exc()}")
            chat_history.append((msg, error_msg))
            return "", chat_history

    def run_evaluation(self, ground_truths_str: str):
        """运行 RAGAS 评测"""
        if not self.evaluator:
            return "❌ 评测器未初始化，请先初始化模型"
        if not self.qa_history:
            return "❌ 没有足够的问答数据进行评测，请先进行至少 1 轮问答"
        
        try:
            # 解析标准答案（用户输入，每行一个）
            ground_truths = [gt.strip() for gt in ground_truths_str.split("\n") if gt.strip()]
            
            # 准备评测数据
            qa_data = []
            for i, qa in enumerate(self.qa_history):
                item = {
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "contexts": qa["contexts"]
                }
                if i < len(ground_truths):
                    item["ground_truth"] = ground_truths[i]
                qa_data.append(item)
            
            # 诊断信息：检查上下文质量
            print(f"\n🔍 RAGAS 评测数据诊断：")
            print(f"   问答对数量：{len(qa_data)}")
            
            empty_contexts_count = 0
            for i, qa in enumerate(qa_data[:5]):  # 检查前5个
                contexts = qa.get("contexts", [])
                print(f"   问题 {i+1}：{qa['question'][:50]}...")
                print(f"     上下文数量：{len(contexts)}")
                if contexts:
                    print(f"     上下文1长度：{len(contexts[0])} 字符")
                    print(f"     上下文1示例：{contexts[0][:80]}...")
                else:
                    print(f"     ⚠️ 上下文为空！")
                    empty_contexts_count += 1
            
            # 执行评测
            result = self.evaluator.evaluate_qa_pairs(qa_data)
            
            # 检查是否有错误
            if "error" in result:
                return f"❌ {result['error']}"
            
            # 格式化输出
            output = "## 📊 RAGAS 评测结果\n\n"
            output += f"**评测问答对数量**: {len(qa_data)}\n\n"
            
            # 检查是否所有上下文都为空
            if empty_contexts_count > 0:
                output += f"⚠️ **诊断信息**：{empty_contexts_count}/{len(qa_data)} 个问答对的上下文为空\n\n"
                output += "### 为什么会出现0分？\n"
                output += "1. **检索模块未返回上下文** - 这是最常见的原因\n"
                output += "2. **向量库构建失败** - 文档未正确加载或分块\n"
                output += "3. **查询与文档不相关** - 问题无法匹配到相关内容\n\n"
                output += "### 解决建议：\n"
                output += "- 检查终端输出，查看检索过程的详细日志\n"
                output += "- 确认文档已正确上传并构建向量库\n"
                output += "- 尝试提出更具体的问题\n"
                output += "- 运行 `python debug_rag.py` 进行系统诊断\n\n"
            
            output += "### 指标得分\n\n"
            
            metrics_desc = {
                "context_precision": "上下文精度（检索到的相关内容在上下文中的排名，越高越好）",
                "context_recall": "上下文召回率（检索到的相关内容占所有相关内容的比例，需要标准答案）",
                "answer_relevancy": "回答相关性（回答与问题的相关程度，越高越好）",
                "faithfulness": "回答忠实度（回答是否完全基于上下文，越高越好）"
            }
            
            for metric, score in result.items():
                desc = metrics_desc.get(metric, "")
                if isinstance(score, float):
                    # 根据分数显示颜色标记
                    if score >= 0.8:
                        mark = "🟢"
                    elif score >= 0.6:
                        mark = "🟡"
                    else:
                        mark = "🔴"
                    output += f"- {mark} **{metric}**: `{score:.4f}` - {desc}\n"
                else:
                    output += f"- ⚪ **{metric}**: `{score}` - {desc}\n"
            
            output += "\n### 评分说明\n"
            output += "- 🟢 优秀 (≥0.8) | 🟡 良好 (0.6-0.8) | 🔴 需改进 (<0.6)\n"
            output += "- 建议至少进行 5 轮问答以获得更准确的评测结果"
            
            return output
        except Exception as e:
            return f"❌ 评测失败：{str(e)}"
    def clear_vector_db(self):
        """清空向量库"""
        if os.path.exists(self.faiss_manager.index_path):
            shutil.rmtree(self.faiss_manager.index_path)
        self.faiss_manager.db = None
        self.qa_chain = None
        return "文件缓存已删除"

    def clear_qa_history(self):
        """清空问答历史"""
        self.qa_history = []
        return "问答历史已清空"

    def create_ui(self):
        """整合所有功能到同一界面"""
        with gr.Blocks(theme=gr.themes.Soft()) as demo:
            gr.Markdown("# 🤖 本地知识库问答工具")
            gr.Markdown("📚 支持 PDF/TXT 文档 & Arxiv 论文 | 🔄 多模型灵活切换 | 📊 RAGAS 质量评测")

            # ═══════════════════════════════════════════════════════════════
            # 第一区块：模型配置
            # ═══════════════════════════════════════════════════════════════
            with gr.Accordion("⚙️ 模型配置", open=True):
                gr.Markdown("💡 **提示**：手动输入框优先级高于下拉选择，留空则使用预设模型")
                
                # LLM 模型配置
                with gr.Group():
                    gr.Markdown("### 🧠 大语言模型 (LLM)")
                    with gr.Row():
                        llm_model = gr.Dropdown(
                            label="📋 选择预设模型",
                            choices=list(SUPPORTED_LLM_MODELS.keys()),
                            value=DEFAULT_LLM_MODEL,
                            info=SUPPORTED_LLM_MODELS[DEFAULT_LLM_MODEL],
                            scale=2
                        )
                        llm_custom = gr.Textbox(
                            label="✏️ 或手动输入模型名称",
                            placeholder="如：qwen, mistral, phi3:mini, all-minilm:16-v2",
                            info="留空则使用上方选择，支持任意 Ollama 模型",
                            scale=3
                        )
                
                # Embedding 模型配置
                with gr.Group():
                    gr.Markdown("### 🔢 嵌入模型 (Embedding)")
                    with gr.Row():
                        embedding_model = gr.Dropdown(
                            label="📋 选择预设模型",
                            choices=list(SUPPORTED_EMBEDDING_MODELS.keys()),
                            value=DEFAULT_EMBEDDING_MODEL,
                            info=SUPPORTED_EMBEDDING_MODELS[DEFAULT_EMBEDDING_MODEL],
                            scale=2
                        )
                        embedding_custom = gr.Textbox(
                            label="✏️ 或手动输入模型名称",
                            placeholder="如：bge-large, all-minilm:33m, nomic-embed-text:latest",
                            info="留空则使用上方选择，支持任意 Ollama 模型",
                            scale=3
                        )
                
                # 初始化按钮和状态
                with gr.Row():
                    init_model_btn = gr.Button("🚀 初始化模型", variant="primary", size="lg")
                model_status = gr.Textbox(
                    label="📊 模型状态",
                    interactive=False,
                    lines=4,
                    show_copy_button=True
                )

            # ═══════════════════════════════════════════════════════════════
            # 第二区块：文件上传 + Arxiv ID 输入
            # ═══════════════════════════════════════════════════════════════
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📁 本地文档")
                    file = gr.File(label="上传 PDF/TXT", file_types=[".pdf", ".txt"])
                    upload_btn = gr.Button("📤 上传文件", variant="primary")
                with gr.Column(scale=1):
                    gr.Markdown("### 📚 Arxiv 论文")
                    arxiv_id = gr.Textbox(
                        label="Arxiv ID", 
                        placeholder="如：2604.02237",
                        info="可在 arxiv.org 搜索论文获取 ID"
                    )
                    arxiv_btn = gr.Button("🌐 加载 Arxiv 论文", variant="primary")
                upload_status = gr.Textbox(label="状态", interactive=False)

            # ═══════════════════════════════════════════════════════════════
            # 第三区块：问答区域
            # ═══════════════════════════════════════════════════════════════
            gr.Markdown("### 💬 智能问答")
            chatbot = gr.Chatbot(height=500, label="对话历史")
            msg = gr.Textbox(placeholder="输入问题，按回车发送...", label="问题输入", show_label=False)

            # ═══════════════════════════════════════════════════════════════
            # 第四区块：缓存管理
            # ═══════════════════════════════════════════════════════════════
            with gr.Row():
                clear_vdb_btn = gr.Button("🗑️ 删除文件缓存", variant="secondary")
                clear_history_btn = gr.Button("🧹 清空问答历史", variant="secondary")

            # ═══════════════════════════════════════════════════════════════
            # 第五区块：RAGAS 评测（默认折叠）
            # ═══════════════════════════════════════════════════════════════
            with gr.Accordion("📊 RAGAS 评测", open=False):
                gr.Markdown("""
                #### 评测指标说明
                | 指标 | 说明 | 是否需要标准答案 |
                |------|------|------------------|
                | Context Precision | 上下文检索精度，检索到的相关内容在上下文中的排名 | 否 |
                | Context Recall | 上下文召回率，检索到的相关内容占所有相关内容的比例 | 是 |
                | Answer Relevance | 回答相关性，回答与问题的相关程度 | 否 |
                | Faithfulness | 回答忠实度，回答是否完全基于上下文 | 否 |
                
                > 💡 建议至少进行 **5 轮问答** 以获得更准确的评测结果
                """)
                ground_truths = gr.Textbox(
                    label="标准答案（用于计算 Context Recall，每行对应一个历史问题）",
                    placeholder="第一题的标准答案\n第二题的标准答案\n第三题的标准答案\n...",
                    lines=5
                )
                eval_btn = gr.Button("🚀 开始评测", variant="primary")
                eval_result = gr.Markdown(label="评测结果")

            # 绑定事件
            # 辅助函数：合并下拉选择和手动输入
            def merge_model_selection(dropdown_val, custom_val):
                """如果手动输入有值则优先使用，否则使用下拉选择"""
                if custom_val and custom_val.strip():
                    return custom_val.strip()
                return dropdown_val
            
            # 1. 模型初始化
            init_model_btn.click(
                fn=lambda llm_drop, llm_custom, emb_drop, emb_custom: self.init_ollama_client(
                    merge_model_selection(llm_drop, llm_custom),
                    merge_model_selection(emb_drop, emb_custom)
                ),
                inputs=[llm_model, llm_custom, embedding_model, embedding_custom],
                outputs=[model_status]
            )
            
            # 2. 文件上传
            upload_btn.click(
                self.upload_file,
                inputs=[file, chatbot],
                outputs=[upload_status, chatbot]
            )
            # 3. Arxiv 加载
            arxiv_btn.click(
                self.load_arxiv_paper,
                inputs=[arxiv_id, chatbot],
                outputs=[upload_status, chatbot]
            )
            # 4. 问答
            msg.submit(self.chat, inputs=[msg, chatbot], outputs=[msg, chatbot])
            # 5. 清除缓存
            clear_vdb_btn.click(self.clear_vector_db, outputs=[upload_status])
            # 6. 清空问答历史
            clear_history_btn.click(self.clear_qa_history, outputs=[upload_status])
            # 7. RAGAS 评测
            eval_btn.click(self.run_evaluation, inputs=[ground_truths], outputs=[eval_result])

        return demo

    def run(self):
        """启动界面"""
        demo = self.create_ui()
        demo.launch(
            server_name=GRADIO_HOST,
            server_port=GRADIO_PORT,
            share=GRADIO_SHARE,
            show_api=False
        )


# 单例实例
gradio_ui = GradioUI()
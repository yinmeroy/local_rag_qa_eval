

# 本地知识库问答工具 (Local RAG Switch)

一个基于 **Ollama + LangChain + FAISS** 的轻量级本地知识库问答系统。支持上传 PDF/TXT 文档或加载 Arxiv 论文，提供自然语言问答、多模型懒加载切换及 RAGAS 质量评测功能。界面简洁，数据完全本地化，适配 Windows 系统。

---

## ✨ 核心功能

- 🔒 **纯本地部署**：文档、向量库、问答模型均在本地运行，无需联网，保障数据隐私
- 📄 **多源知识接入**：支持 PDF/TXT 本地文档上传，以及通过 ID 直接加载 Arxiv 论文摘要与元数据
- 🔄 **多模型懒加载**：支持 LLM（如 Llama3, Qwen, Phi3）和 Embedding 模型按需下载与切换，节省内存
- 📊 **RAGAS 智能评测**：内置 RAGAS 评测模块，通过外部 API 自动评估检索精度、回答忠实度及相关性
- 💬 **智能对话交互**：基于 Gradio 构建统一界面，集成模型初始化、文件管理、多轮对话及缓存清理
- ⚡ **高效检索引擎**：采用 FAISS 向量数据库，结合智能分块策略，实现快速精准的知识检索

---

## 📊 RAGAS 智能评测详解

本系统集成 RAGAS (Retrieval Augmented Generation Assessment) 框架，通过以下四个核心维度量化评估问答系统的质量：

| 指标名称 | 英文标识 | 定义与意义 |
| :--- | :--- | :--- |
| **🎯 忠实度** | `Faithfulness` | 评估答案是否完全基于检索上下文，无幻觉/编造。分值越高越可信。 |
| **🔍 答案相关性** | `Answer Relevance` | 评估答案是否直接回答了用户问题，无偏题或冗余。分值越高越精准。 |
| **📚 上下文精度** | `Context Precision` | 评估检索到的相关片段是否排在前面，排除噪音干扰。分值越高检索越准。 |
| **♻️ 上下文召回率** | `Context Recall` | 评估检索内容是否覆盖了所有关键信息（需提供标准答案）。分值越高遗漏越少。 |



## 🛠️ 环境配置

### 1. 基础软件安装

#### Ollama (核心依赖)

1. **下载安装**：访问 [Ollama 官网](https://ollama.com/) 下载 Windows 版本安装包并运行
2. **验证安装**：打开终端（CMD 或 PowerShell），输入以下命令，若显示版本号则安装成功：
   ```bash
   ollama --version
   ```
3. **启动服务**：
   - Ollama 安装后通常会在后台自动运行
   - 若未运行，请在终端执行：
     ```bash
     ollama serve
     ```

### 2. 创建 Conda 虚拟环境  （或venv）

建议使用 Conda 隔离项目依赖，避免冲突。

```bash
# 1. 创建名为 rag_env 的虚拟环境，指定 Python 版本为 3.11 (推荐 3.8-3.11)
conda create -n rag_env python=3.11

# 2. 激活环境
conda activate rag_env

# 3. 验证环境
python --version
```

### 3. 安装项目依赖

在项目根目录下执行：

```bash
pip install -r requirements.txt
```

### 4. 下载 Ollama 模型

根据需求下载一个大语言模型和一个嵌入模型。在终端执行：

```bash
# 推荐组合（轻量级，中文友好）
ollama pull phi3              # 大语言模型 (约 2GB)
ollama pull nomic-embed-text  # 嵌入模型 (约 500MB)

# 其他可选模型
# ollama pull llama3          # 均衡型
# ollama pull qwen            # 阿里通义千问
# ollama pull all-minilm      # 超轻量嵌入模型
```

---

## ⚙️ RAGAS 评测配置 (重要)

⚠️ **注意**：本项目的 RAGAS 评测功能不依赖本地 Ollama 模型，而是通过 OpenAI 兼容 API 进行云端评测。这是为了保证评测的准确性和速度（本地小模型难以胜任复杂的语义评分任务）。

⚠️ **重要提醒**：项目中不包含有效的 API Key，你需要自行配置。

### 1. 获取 API Key

你需要拥有一个支持 OpenAI 兼容接口的 API Key：

- **官方 OpenAI**：需科学上网，地址 `https://api.openai.com/v1`
- **第三方中转站**：如硅基流动、OneAPI 等，地址通常为 `https://your-provider.com/v1`

### 2. 修改代码配置

打开 `ui/gradio_ui.py`，找到 `init_ollama_client` 方法中的 `RagasEvaluator` 初始化部分（约第 53 行）：

```python
# ui/gradio_ui.py 片段
self.evaluator = RagasEvaluator(
    api_base_url="https://your-api-base-url/v1",  # 【必填】修改为你的中转站地址或官方地址
    api_key="sk-your-real-api-key-here"           # 【必填】修改为你自己的 API Key
)
```

### 3. 更换评测模型 (可选)

如果你希望更换评测时使用的 LLM 或 Embedding 模型（例如使用更便宜的模型），可以修改 `eval/ragas_evaluator.py` 中的初始化参数：

```python
# eval/ragas_evaluator.py 片段
self.llm = ChatOpenAI(
    base_url=api_base_url,
    api_key=api_key,
    temperature=0.0,
    model="gpt-4o",  # <--- 可更换为其他兼容模型，如 gpt-3.5-turbo, qwen-plus 等
)

self.embeddings = OpenAIEmbeddings(
    openai_api_base=api_base_url,
    openai_api_key=api_key,
    model="text-embedding-ada-002", # <--- 可更换为其他兼容嵌入模型
)
```

---

## 📂 项目结构

```
Local_Rag_Switch/
├── config/                 # 配置模块
│   ├── model_config.py     # 支持的模型列表定义
│   └── settings.py         # 全局参数配置 (端口、路径、分块大小等)
├── document/               # 文档处理模块
│   ├── doc_processor.py    # PDF/TXT 加载与智能分块
│   └── arxiv_loader.py     # Arxiv 论文获取与处理
├── models/                 # 模型适配模块
│   ├── ollama_client.py    # Ollama API 原生调用封装
│   └── ollama_langchain_adapter.py # LangChain 接口适配
├── vector_db/              # 向量数据库模块
│   └── faiss_manager.py    # FAISS 索引构建、保存与检索
├── eval/                   # 评测模块
│   └── ragas_evaluator.py  # RAGAS 质量评测封装 (API 模式)
├── ui/                     # 用户界面
│   └── gradio_ui.py        # Gradio 前端逻辑与事件绑定
├── main.py                 # 程序入口
└── requirements.txt        # 依赖列表
```

---

## 🚀 启动与使用

### 1. 启动应用

确保 Conda 环境已激活，且 Ollama 服务正在运行：

```bash
conda activate rag_env
python main.py
```

启动成功后，浏览器自动打开或访问：**http://127.0.0.1:7860**

### 2. 使用流程

#### ① 初始化模型
1. 在界面顶部选择或手动输入进行RAG的 LLM 和 Embedding 模型
2. 点击「🔄 初始化模型」
3. 若状态显示"✅ 所有模型已就绪"，则继续；若提示未下载，请按提示在终端执行 `ollama pull <model_name>`

#### ② 导入知识
- **本地文档**：上传 PDF 或 TXT 文件
- **Arxiv 论文**：输入论文 ID（如 `2604.02237`），点击加载（需联网）

#### ③ 智能问答
在对话框提问，系统将基于知识库回答。

#### ④ RAGAS 评测 （可选）
1. 进行若干轮问答后，展开底部「📊 RAGAS 评测」
2. 填写标准答案以计算 Recall 指标
3. 点击「🚀 开始评测」查看各项得分

---

## ❓ 常见问题

### Q: 启动时报 "Ollama 连接失败"？

**A:** 请检查 Ollama 是否在后台运行。尝试在终端执行 `ollama list`，若能列出模型则服务正常。若仍报错，检查防火墙是否拦截了 11434 端口。

### Q: RAGAS 评测报错 "API Error" 或 "Authentication Error"？

**A:** 请检查 `ui/gradio_ui.py` 中的 `api_base_url` 和 `api_key` 是否正确。确保你的 API Key 有效且有余额。如果是中转站，请确认地址格式正确（通常以 `/v1` 结尾）。

### Q: 上传文件后回答不准确？

**A:** 尝试调整 `config/settings.py` 中的 `CHUNK_SIZE`（默认 1000）。减小分块大小可保留更多细节，增大则获取更多上下文。也可尝试切换中文能力更强的模型（如 Qwen）。

### Q: 如何清理缓存？

**A:** 点击界面底部的「🗑️ 删除文件缓存」按钮，或直接删除项目根目录下的 `faiss_local_index` 文件夹。

---

## 📝 免责声明

本工具仅用于学习研究和个人自用。请确保上传文档拥有合法使用权，作者不对使用本工具产生的任何后果承担责任。
# document/arxiv_loader.py
import arxiv
from langchain_core.documents import Document
from typing import List, Optional
from document.doc_processor import doc_processor


class ArxivLoader:
    """Arxiv 论文加载器"""
    
    def __init__(self):
        self.client = arxiv.Client()
    
    def fetch_paper(self, arxiv_id: str) -> Optional[Document]:
        """
        根据 Arxiv ID 获取论文
        
        Args:
            arxiv_id: Arxiv 论文 ID，格式如 "2301.07041" 或 "arXiv:2301.07041"
        
        Returns:
            Document 对象，包含论文内容
        """
        try:
            # 清理 ID 格式
            arxiv_id = arxiv_id.replace("arXiv:", "").strip()
            
            # 搜索论文
            search = arxiv.Search(id_list=[arxiv_id], max_results=1)
            results = list(self.client.results(search))
            
            if not results:
                return None
            
            paper = results[0]
            
            # 构建论文内容（摘要 + 元数据）
            content = f"""# 论文信息

## 标题
{paper.title}

## 作者
{', '.join([a.name for a in paper.authors])}

## 摘要
{paper.summary}

## 分类
{', '.join(paper.categories)}

## 发布日期
{paper.published.strftime('%Y-%m-%d')}

## 链接
{paper.entry_id}
"""
            
            return Document(
                page_content=content,
                metadata={
                    "source": f"arxiv:{arxiv_id}",
                    "title": paper.title,
                    "arxiv_id": arxiv_id,
                    "authors": [a.name for a in paper.authors],
                    "published": paper.published.strftime('%Y-%m-%d')
                }
            )
        except Exception as e:
            raise ValueError(f"获取 Arxiv 论文失败：{str(e)}")
    
    def fetch_and_process(self, arxiv_id: str) -> List[Document]:
        """
        获取并处理论文（分块）
        
        Args:
            arxiv_id: Arxiv 论文 ID
        
        Returns:
            分块后的文档列表
        """
        doc = self.fetch_paper(arxiv_id)
        if not doc:
            raise ValueError(f"未找到 Arxiv 论文：{arxiv_id}")
        return doc_processor.split_document([doc])
    
    def fetch_multiple_papers(self, arxiv_ids: List[str]) -> List[Document]:
        """
        批量获取多篇论文
        
        Args:
            arxiv_ids: Arxiv 论文 ID 列表
        
        Returns:
            文档列表
        """
        docs = []
        for arxiv_id in arxiv_ids:
            try:
                doc = self.fetch_paper(arxiv_id)
                if doc:
                    docs.append(doc)
            except Exception as e:
                print(f"跳过论文 {arxiv_id}: {str(e)}")
        return docs


# 单例实例
arxiv_loader = ArxivLoader()
import os
import sys
import time
import json
import logging
import argparse
import numpy as np
import faiss
import torch
from typing import List, Dict

# 导入自定义模块
# 确保当前目录在系统路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embedding.build_hybird import DocumentVectorIndexBuilder

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentSearcher:
    """文档搜索器：用于在索引中搜索与查询最相近的文档块"""
    
    def __init__(self, index_folder: str, index_prefix: str = "document_index"):
        """
        初始化文档搜索器
        
        Args:
            index_folder: 索引文件夹路径
            index_prefix: 索引文件前缀
        """
        self.index_builder = DocumentVectorIndexBuilder(model_name="/mnt/workspace/models/bge-large-zh-v1.5", device="cpu")
        self.index_folder = index_folder
        self.index_prefix = index_prefix
        self.loaded = False
        
    def load_index(self) -> None:
        """加载索引文件"""
        try:
            self.index_builder.load_index(self.index_folder, self.index_prefix)
            self.loaded = True
            logger.info(f"成功从 {self.index_folder} 加载索引")
        except Exception as e:
            logger.error(f"加载索引失败: {str(e)}")
            raise
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """
        搜索与查询最相似的文档块
        
        Args:
            query: 查询文本
            k: 返回的结果数量
            
        Returns:
            包含相似文档块信息的列表
        """
        # 调用索引构建器的搜索方法
        results = self.index_builder.search(query, top_k=k)
        
        # 格式化结果
        formatted_results = []
        for i, result in enumerate(results):
            formatted_results.append({
                'rank': i + 1,
                'filename': result['filename'],
                'similarity_score': 1 / (1 + result['distance']),  # 将距离转换为相似度分数
                'chunk_idx': result['chunk_idx'],
                'total_chunks': result['total_chunks'],
                'text': result['text']
            })
        
        return formatted_results
    
    def search_and_print(self, query: str, k: int = 5) -> None:
        """
        搜索并打印结果
        
        Args:
            query: 查询文本
            k: 返回的结果数量
        """
        results = self.search(query, k)
        
        print("\n搜索结果:")
        if not results:
            print("没有找到匹配的结果")
        else:
            for result in results:
                print(f"结果 {result['rank']} (相似度: {result['similarity_score']:.4f}):")
                print(f"文件名: {result['filename']}")
                print(f"块索引: {result['chunk_idx']+1}/{result['total_chunks']}")
                print(f"内容: {result['text'][:200]}..." if len(result['text']) > 200 
                      else f"内容: {result['text']}")
                print("-" * 80)
    
    def search_to_json(self, query: str, k: int = 5) -> str:
        """
        搜索并返回JSON格式的结果
        
        Args:
            query: 查询文本
            k: 返回的结果数量
            
        Returns:
            JSON格式的搜索结果
        """
        results = self.search(query, k)
        
        query_json = {
            "query": query,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": []
        }
        
        for result in results:
            query_json["results"].append({
                "rank": result["rank"],
                "filename": result["filename"],
                "similarity_score": result["similarity_score"],
                "chunk_index": result["chunk_idx"],
                "text": result["text"]
            })
        
        return json.dumps(query_json, ensure_ascii=False, indent=2)
    
    def hybrid_search(self, query: str, k: int = 5, vector_weight: float = 1.0) -> List[Dict]:
        """
        混合搜索 - 结合向量相似度和BM25的结果
        
        Args:
            query: 查询文本
            k: 返回的结果数量
            vector_weight: 向量相似度的权重 (0.0-1.0)，剩余权重分配给BM25结果
            
        Returns:
            包含相关文档块信息的列表，按混合得分排序
        """
        # 调用索引构建器的混合搜索方法
        if not self.loaded:
            self.load_index()
        
        # 确保索引构建器支持混合搜索
        if not hasattr(self.index_builder, 'hybrid_search'):
            raise AttributeError("索引构建器不支持混合搜索功能")
        
        results = self.index_builder.hybrid_search(query, top_k=k, vector_weight=vector_weight)
        
        # 格式化结果
        formatted_results = []
        for i, result in enumerate(results):
            formatted_results.append({
                'rank': i + 1,
                'filename': result['filename'],
                'vector_score': result['vector_score'],
                'bm25_score': result['bm25_score'],
                'hybrid_score': result['hybrid_score'],
                'chunk_idx': result['chunk_idx'],
                'total_chunks': result['total_chunks'],
                'text': result['text']
            })
        
        return formatted_results


def main():
    parser = argparse.ArgumentParser(description='文档搜索工具')
    parser.add_argument('--index_folder', type=str, default="/mnt/workspace/RAG/vector_test",
                        help='索引文件夹路径')
    parser.add_argument('--index_prefix', type=str, default="test_index_300_100",
                        help='索引文件前缀')
    parser.add_argument('--k', type=int, default=5,
                        help='返回的结果数量')
    parser.add_argument('--query', type=str, default=None,
                        help='查询文本，若不提供则进入交互模式')
    parser.add_argument('--json', action='store_true',
                        help='以JSON格式输出结果')
    parser.add_argument('--output', type=str, default=None,
                        help='将结果保存到指定文件')
    
    args = parser.parse_args()
    
    # 创建搜索器
    searcher = DocumentSearcher(args.index_folder, args.index_prefix)
    
    try:
        # 加载索引
        searcher.load_index()
        
        if args.query:
            # 单次查询模式
            if args.json:
                result_json = searcher.search_to_json(args.query, args.k)
                
                if args.output:
                    with open(args.output, 'w', encoding='utf-8') as f:
                        f.write(result_json)
                    print(f"结果已保存到 {args.output}")
                else:
                    print(result_json)
            else:
                searcher.search_and_print(args.query, args.k)
        else:
            # 交互模式
            print("\n=== 文档搜索系统 ===")
            print("输入 'q' 或 'exit' 退出程序")
            
            while True:
                query = input("\n请输入搜索查询: ")
                
                # 检查是否退出
                if query.lower() in ['q', 'exit', 'quit']:
                    print("感谢使用，再见！")
                    break
                
                # 执行搜索
                searcher.search_and_print(query, args.k)
        
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
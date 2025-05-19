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
from embedding.build_document_index import DocumentVectorIndexBuilder

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
    
    def search(self, query: str, k: int = 5, method: str = "vector", alpha: float = 0.5) -> List[Dict]:
        """
        搜索与查询最相似的文档块
        
        Args:
            query: 查询文本
            k: 返回的结果数量
            method: 搜索方法 ('vector', 'bm25', 'hybrid')
            alpha: 混合搜索中向量搜索的权重 (0-1)
            
        Returns:
            包含相似文档块信息的列表
        """
        # 根据选择的方法调用相应的搜索函数
        if method == "bm25":
            if self.index_builder.bm25_index is None:
                logger.warning("BM25索引不可用，回退到向量检索")
                results = self.index_builder.search(query, top_k=k)
            else:
                results = self.index_builder.bm25_search(query, top_k=k)
        elif method == "hybrid":
            if self.index_builder.bm25_index is None:
                logger.warning("BM25索引不可用，回退到向量检索")
                results = self.index_builder.search(query, top_k=k)
            else:
                results = self.index_builder.hybrid_search(query, top_k=k, alpha=alpha)
        else:  # 默认为向量检索
            results = self.index_builder.search(query, top_k=k)
        
        # 格式化结果
        formatted_results = []
        for i, result in enumerate(results):
            formatted_result = {
                'rank': i + 1,
                'filename': result['filename'],
                'similarity_score': result.get('similarity_score', 0),
                'chunk_idx': result['chunk_idx'],
                'total_chunks': result['total_chunks'],
                'text': result['text']
            }
            
            # 添加特定搜索方法的额外信息
            if method == "bm25" and 'bm25_score' in result:
                formatted_result['bm25_score'] = result['bm25_score']
            elif method == "hybrid":
                if 'vector_score' in result:
                    formatted_result['vector_score'] = result['vector_score']
                if 'bm25_score' in result:
                    formatted_result['bm25_score'] = result['bm25_score']
                
            formatted_results.append(formatted_result)
        
        return formatted_results
    
    def search_and_print(self, query: str, k: int = 5, method: str = "vector", alpha: float = 0.5) -> None:
        """
        搜索并打印结果
        
        Args:
            query: 查询文本
            k: 返回的结果数量
            method: 搜索方法 ('vector', 'bm25', 'hybrid')
            alpha: 混合搜索中向量搜索的权重 (0-1)
        """
        results = self.search(query, k, method, alpha)
        
        print(f"\n搜索结果 (使用{method}检索法):")
        if not results:
            print("没有找到匹配的结果")
        else:
            for result in results:
                print(f"结果 {result['rank']} (相似度: {result['similarity_score']:.4f}):")
                print(f"文件名: {result['filename']}")
                print(f"块索引: {result['chunk_idx']+1}/{result['total_chunks']}")
                
                # 根据不同搜索方法显示额外信息
                if method == "bm25" and 'bm25_score' in result:
                    print(f"BM25分数: {result['bm25_score']:.4f}")
                elif method == "hybrid":
                    if 'vector_score' in result:
                        print(f"向量分数: {result['vector_score']:.4f}")
                    if 'bm25_score' in result:
                        print(f"BM25分数: {result['bm25_score']:.4f}")
                
                print(f"内容: {result['text'][:200]}..." if len(result['text']) > 200 
                      else f"内容: {result['text']}")
                print("-" * 80)
    
    def search_to_json(self, query: str, k: int = 5, method: str = "all", alpha: float = 0.5) -> str:
        """
        搜索并返回JSON格式的结果
        
        Args:
            query: 查询文本
            k: 返回的结果数量
            method: 搜索方法 ('vector', 'bm25', 'hybrid', 'all')
            alpha: 混合搜索中向量搜索的权重 (0-1)
            
        Returns:
            JSON格式的搜索结果
        """
        query_json = {
            "query": query,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": {}
        }
        
        # 如果指定了单个方法
        if method != "all":
            results = self.search(query, k, method, alpha)
            
            method_results = []
            for result in results:
                result_item = {
                    "rank": result["rank"],
                    "filename": result["filename"],
                    "similarity_score": result["similarity_score"],
                    "chunk_index": result["chunk_idx"],
                    "total_chunks": result["total_chunks"],
                    "text": result["text"]
                }
                
                # 添加特定搜索方法的额外信息
                if method == "bm25" and 'bm25_score' in result:
                    result_item["bm25_score"] = result["bm25_score"]
                elif method == "hybrid":
                    if 'vector_score' in result:
                        result_item["vector_score"] = result["vector_score"]
                    if 'bm25_score' in result:
                        result_item["bm25_score"] = result["bm25_score"]
                
                method_results.append(result_item)
            
            query_json["results"][method] = method_results
            
            # 如果是混合搜索，添加alpha值
            if method == "hybrid":
                query_json["alpha"] = alpha
        
        # 如果需要所有方法的结果
        else:
            # 执行向量检索
            vector_results = self.search(query, k, "vector", alpha)
            vector_items = []
            for result in vector_results:
                result_item = {
                    "rank": result["rank"],
                    "filename": result["filename"],
                    "similarity_score": result["similarity_score"],
                    "chunk_index": result["chunk_idx"],
                    "total_chunks": result["total_chunks"],
                    "text": result["text"]
                }
                vector_items.append(result_item)
            query_json["results"]["vector"] = vector_items
            
            # 执行BM25检索
            try:
                bm25_results = self.search(query, k, "bm25", alpha)
                bm25_items = []
                for result in bm25_results:
                    result_item = {
                        "rank": result["rank"],
                        "filename": result["filename"],
                        "similarity_score": result["similarity_score"],
                        "chunk_index": result["chunk_idx"],
                        "total_chunks": result["total_chunks"],
                        "text": result["text"]
                    }
                    if 'bm25_score' in result:
                        result_item["bm25_score"] = result["bm25_score"]
                    bm25_items.append(result_item)
                query_json["results"]["bm25"] = bm25_items
            except Exception as e:
                logger.warning(f"BM25检索失败: {str(e)}")
                query_json["results"]["bm25"] = {"error": str(e)}
            
            # 执行混合检索
            try:
                hybrid_results = self.search(query, k, "hybrid", alpha)
                hybrid_items = []
                for result in hybrid_results:
                    result_item = {
                        "rank": result["rank"],
                        "filename": result["filename"],
                        "similarity_score": result["similarity_score"],
                        "chunk_index": result["chunk_idx"],
                        "total_chunks": result["total_chunks"],
                        "text": result["text"]
                    }
                    if 'vector_score' in result:
                        result_item["vector_score"] = result["vector_score"]
                    if 'bm25_score' in result:
                        result_item["bm25_score"] = result["bm25_score"]
                    hybrid_items.append(result_item)
                query_json["results"]["hybrid"] = hybrid_items
                query_json["alpha"] = alpha
            except Exception as e:
                logger.warning(f"混合检索失败: {str(e)}")
                query_json["results"]["hybrid"] = {"error": str(e)}
        
        return json.dumps(query_json, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description='文档搜索工具')
    parser.add_argument('--index_folder', type=str, default="/mnt/workspace/RAG/vectorDB",
                        help='索引文件夹路径')
    parser.add_argument('--index_prefix', type=str, default="all_index_400_100",
                        help='索引文件前缀')
    parser.add_argument('--k', type=int, default=5,
                        help='返回的结果数量')
    parser.add_argument('--query', type=str, default=None,
                        help='查询文本，若不提供则进入交互模式')
    parser.add_argument('--json', action='store_true',
                        help='以JSON格式输出结果')
    parser.add_argument('--output', type=str, default=None,
                        help='将结果保存到指定文件')
    # 添加新的参数
    parser.add_argument('--method', type=str, choices=['vector', 'bm25', 'hybrid', 'all'], default='all',
                        help='搜索方法: vector(向量检索), bm25(BM25检索), hybrid(混合检索), all(所有方法)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='混合检索中向量搜索的权重 (0-1)')
    
    args = parser.parse_args()
    
    # 创建搜索器
    searcher = DocumentSearcher(args.index_folder, args.index_prefix)
    
    try:
        # 加载索引
        searcher.load_index()
        
        if args.query:
            # 单次查询模式
            if args.json or args.method == 'all':
                result_json = searcher.search_to_json(args.query, args.k, args.method, args.alpha)
                
                if args.output:
                    with open(args.output, 'w', encoding='utf-8') as f:
                        f.write(result_json)
                    print(f"结果已保存到 {args.output}")
                else:
                    print(result_json)
            else:
                searcher.search_and_print(args.query, args.k, args.method, args.alpha)
        else:
            # 交互模式
            print("\n=== 文档搜索系统 ===")
            print("输入 'q' 或 'exit' 退出程序")
            print("可用搜索方法: [vector] 向量检索, [bm25] BM25检索, [hybrid] 混合检索, [all] 所有方法")
            print("输入格式: 查询文本 [可选: 方法 alpha]")
            print("例如: '什么是人工智能 hybrid 0.7' 或 '什么是人工智能 all'")
            print("注意: 选择'all'方法时将输出JSON格式结果")
            
            while True:
                user_input = input("\n请输入搜索查询: ")
                
                # 检查是否退出
                if user_input.lower() in ['q', 'exit', 'quit']:
                    print("感谢使用，再见！")
                    break
                
                # 解析用户输入
                input_parts = user_input.split()
                
                # 默认值
                current_method = args.method
                current_alpha = args.alpha
                
                # 检查是否提供了方法和alpha
                if len(input_parts) >= 3 and input_parts[-2] in ['vector', 'bm25', 'hybrid', 'all']:
                    try:
                        current_alpha = float(input_parts[-1])
                        current_method = input_parts[-2]
                        query = ' '.join(input_parts[:-2])
                    except ValueError:
                        # 如果最后一部分不是有效的浮点数
                        query = user_input
                elif len(input_parts) >= 2 and input_parts[-1] in ['vector', 'bm25', 'hybrid', 'all']:
                    current_method = input_parts[-1]
                    query = ' '.join(input_parts[:-1])
                else:
                    query = user_input
                
                # 执行搜索
                if current_method == 'all':
                    # 如果是所有方法，则输出JSON
                    result_json = searcher.search_to_json(query, args.k, current_method, current_alpha)
                    print(result_json)
                    
                    # 询问是否保存到文件
                    save_option = input("\n是否保存结果到文件? (y/n): ")
                    if save_option.lower() == 'y':
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        default_filename = f"search_results_{timestamp}.json"
                        filename = input(f"请输入文件名 (默认: {default_filename}): ")
                        if not filename:
                            filename = default_filename
                        
                        with open(filename, 'w', encoding='utf-8') as f:
                            f.write(result_json)
                        print(f"结果已保存到 {filename}")
                else:
                    searcher.search_and_print(query, args.k, current_method, current_alpha)
        
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
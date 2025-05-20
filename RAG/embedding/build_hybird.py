import os
import sys
import time
import pickle
import logging
import numpy as np
import faiss
from typing import List, Dict, Tuple, Union
from tqdm import tqdm 
import argparse
# 导入BM25相关库
import jieba
import re
from rank_bm25 import BM25Okapi

# 导入文档处理模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from text_split.process_docu import process_folder, chunk_by_chars, chunk_by_sentences, chunk_by_paragraphs

# 添加embedding模块路径
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'embedding'))
try:
    from embedding.BGE_embdding import FlagModel
except ImportError:
    from FlagEmbedding import FlagModel

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentVectorIndexBuilder:
    """用于为文档分块创建和管理向量索引的类"""
    
    def __init__(self, model_name: str = "BAAI/bge-large-zh-v1.5", device: str = None):
        """
        初始化文档向量索引构建器
        
        Args:
            model_name: 用于生成嵌入向量的模型名称
            device: 运行模型的设备 ('cpu', 'cuda')
        """
        if device is None:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model_name = model_name
        self._load_model()
        self.index = None
        self.chunks = {}  # 存储文档分块
        self.vectors = None
        self.chunk_metadata = []  # 存储分块元数据
        
        # 添加BM25相关属性
        self.bm25_index = None
        self.tokenized_corpus = None
        
    def _load_model(self):
        """加载embedding模型"""
        logger.info(f"加载embedding模型: {self.model_name} 到 {self.device} 设备")
        self.model = FlagModel(
            self.model_name,
            query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
            use_fp16=False,
            devices=self.device
        )
        
    def _create_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        为文本列表创建嵌入向量
        
        Args:
            texts: 需要嵌入的文本列表
        
        Returns:
            文本的嵌入向量数组
        """
        # if self.model is None:
        #     self._load_model()
        
        logger.info(f"正在为 {len(texts)} 条文本块创建嵌入向量...")
        
        # 对大量文本进行分批处理以显示进度
        batch_size = 32  # 可根据内存情况调整批次大小
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        all_embeddings = []
        
        # 使用tqdm创建进度条
        for batch in tqdm(batches, desc="生成嵌入向量", unit="批次"):
            batch_embeddings = self.model.encode(batch)
            all_embeddings.append(batch_embeddings)
        
        # 合并所有批次的结果
        if len(all_embeddings) > 1:
            embeddings = np.vstack(all_embeddings)
        else:
            embeddings = all_embeddings[0] if all_embeddings else np.array([])
        
        return embeddings.astype(np.float32)  # FAISS 需要 float32 类型
    
    def create_index_from_folder(self, 
                                folder_path: str, 
                                chunk_method: str = "by_chars", 
                                max_chars: int = 500,
                                max_sentences: int = 5,
                                max_paragraphs: int = 3, 
                                overlap: int = 100,
                                overlap_sentences: int = 1,
                                overlap_paragraphs: int = 1) -> None:
        """
        从文件夹读取文档，进行分块并创建向量索引
        
        Args:
            folder_path: 文档所在文件夹路径
            chunk_method: 分块方法 ("by_chars", "by_sentences", "by_paragraphs")
            max_chars: 按字符分块时的最大字符数
            max_sentences: 按句子分块时的最大句子数
            max_paragraphs: 按段落分块时的最大段落数
            overlap: 按字符分块时的重叠字符数
            overlap_sentences: 按句子分块时的重叠句子数
            overlap_paragraphs: 按段落分块时的重叠段落数
        """
        logger.info(f"从 {folder_path} 读取文档并使用 {chunk_method} 方法进行分块...")
        
        # 根据选择的方法设置参数
        kwargs = {}
        if chunk_method == "by_chars":
            kwargs = {"max_chars": max_chars, "overlap": overlap}
        elif chunk_method == "by_sentences":
            kwargs = {"max_sentences": max_sentences, "overlap": overlap_sentences}
        elif chunk_method == "by_paragraphs":
            kwargs = {"max_paragraphs": max_paragraphs, "overlap": overlap_paragraphs}
        
        # 处理文件夹中的文档并分块
        self.chunks = process_folder(folder_path, chunk_method=chunk_method, **kwargs)
        
        # 将所有分块转换为一个列表，用于创建向量
        all_chunks = []
        self.chunk_metadata = []
        
        print("处理文档分块信息...")
        # 使用tqdm显示文档处理进度
        for filename in tqdm(self.chunks.keys(), desc="处理文档元数据", unit="文件"):
            file_chunks = self.chunks[filename]
            for chunk_idx, chunk_text in enumerate(file_chunks):
                all_chunks.append(chunk_text)
                self.chunk_metadata.append({
                    'filename': filename,
                    'chunk_idx': chunk_idx,
                    'total_chunks': len(file_chunks),
                    'text': chunk_text
                })

        # 创建向量
        if all_chunks:
            print(f"开始为 {len(all_chunks)} 个文档块创建向量...")
            vectors = self._create_embeddings(all_chunks)
            
            # 创建FAISS索引
            print("构建FAISS索引...")
            dimension = vectors.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(vectors)
            self.vectors = vectors
            
            # 构建BM25索引
            print("构建BM25索引...")
            self._build_bm25_index(all_chunks)
            
            logger.info(f"成功为 {len(all_chunks)} 个文档块创建向量索引和BM25索引")
        else:
            logger.warning("没有找到需要索引的文档块")
    
    def save_index(self, folder_path: str, prefix: str = "document_index") -> None:
        """
        保存索引和相关数据到指定文件夹
        
        Args:
            folder_path: 保存文件夹路径
            prefix: 文件名前缀
        """
        # 确保索引目录存在
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # 保存FAISS索引
        index_path = os.path.join(folder_path, f"{prefix}_index.faiss")
        faiss.write_index(self.index, index_path)
        
        # 保存元数据和BM25索引
        metadata = {
            "chunks": self.chunks,
            "chunk_metadata": self.chunk_metadata,
            "bm25_index": self.bm25_index,
            "tokenized_corpus": self.tokenized_corpus
        }
        metadata_path = os.path.join(folder_path, f"{prefix}_metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"索引和元数据已保存至 {folder_path}")
    
    def load_index(self, folder_path: str, prefix: str = "document_index") -> None:
        """
        从指定文件夹加载索引和相关数据
        
        Args:
            folder_path: 保存文件夹路径
            prefix: 文件名前缀
        """
        # 加载FAISS索引
        index_path = os.path.join(folder_path, f"{prefix}_index.faiss")
        self.index = faiss.read_index(index_path)
        
        # 加载元数据
        metadata_path = os.path.join(folder_path, f"{prefix}_metadata.pkl")
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.chunks = metadata["chunks"]
        self.chunk_metadata = metadata["chunk_metadata"]
        
        # 加载BM25相关数据
        if "bm25_index" in metadata and "tokenized_corpus" in metadata:
            self.bm25_index = metadata["bm25_index"]
            self.tokenized_corpus = metadata["tokenized_corpus"]
        
        logger.info(f"从 {folder_path} 成功加载索引和元数据")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        搜索与查询最相关的文档块
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            包含相关文档块信息的列表
        """
        if self.model is None:
            self._load_model()
            
        # 为查询文本创建向量
        query_vector = self._create_embeddings([query])
        
        # 使用FAISS搜索最近邻
        distances, indices = self.index.search(query_vector, top_k)
        
        # 返回结果
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunk_metadata):
                result = self.chunk_metadata[idx].copy()
                result['distance'] = float(distances[0][i])
                results.append(result)
        
        return results
    
    def _tokenize_text(self, text: str) -> List[str]:
        """
        对文本进行分词处理
        
        Args:
            text: 待分词文本
            
        Returns:
            分词后的词语列表
        """
        # 移除标点符号和特殊字符
        text = re.sub(r'[^\w\s]', '', text)
        # 使用结巴分词
        return [word for word in jieba.cut(text) if word.strip()]
    
    def _build_bm25_index(self, texts: List[str]) -> None:
        """
        为文本列表构建BM25索引
        
        Args:
            texts: 需要索引的文本列表
        """
        logger.info(f"正在为 {len(texts)} 条文本块构建BM25索引...")
        
        # 对文本进行分词处理
        self.tokenized_corpus = [self._tokenize_text(text) for text in tqdm(texts, desc="分词处理", unit="块")]
        
        # 构建BM25索引
        self.bm25_index = BM25Okapi(self.tokenized_corpus)
        
        logger.info("BM25索引构建完成")
    
    def hybrid_search(self, query: str, top_k: int = 5, vector_weight: float = 0.7) -> List[Dict]:
        """
        混合搜索 - 结合向量相似度和BM25的结果
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            vector_weight: 向量相似度的权重 (0.0-1.0)，剩余权重分配给BM25结果
            
        Returns:
            包含相关文档块信息的列表，按混合得分排序
        """
        print("******************************"+ str(vector_weight) +"***************************************")
        if self.model is None:
            self._load_model()
        
        if self.index is None or self.bm25_index is None:
            raise ValueError("索引未初始化，请先创建或加载索引")
        
        # 计算BM25权重
        bm25_weight = 1.0 - vector_weight
        
        # 检索更多的候选项进行重排序
        expanded_top_k = min(top_k * 3, len(self.chunk_metadata))
        
        # 1. 向量搜索部分
        query_vector = self._create_embeddings([query])
        distances, indices = self.index.search(query_vector, expanded_top_k)
        
        # 向量结果得分归一化（较小距离表示更高相关性）
        max_dist = np.max(distances[0]) if distances[0].size > 0 else 1.0
        min_dist = np.min(distances[0]) if distances[0].size > 0 else 0.0
        dist_range = max(max_dist - min_dist, 1e-8)  # 避免除零错误
        
        # 向量搜索结果集
        vector_results = {}
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunk_metadata):
                # 将距离转换为相似性分数 (距离越小，分数越高)
                score = 1.0 - (distances[0][i] - min_dist) / dist_range
                vector_results[idx] = {
                    'idx': idx,
                    'vector_score': score,
                    'metadata': self.chunk_metadata[idx]
                }
        
        # 2. BM25搜索部分
        tokenized_query = self._tokenize_text(query)
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        
        # BM25分数归一化
        max_bm25 = np.max(bm25_scores) if bm25_scores.size > 0 else 1.0
        min_bm25 = np.min(bm25_scores) if bm25_scores.size > 0 else 0.0
        bm25_range = max(max_bm25 - min_bm25, 1e-8)  # 避免除零错误
        
        # 获取BM25的前K个结果
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:expanded_top_k]
        
        # BM25搜索结果集
        bm25_results = {}
        for idx in bm25_top_indices:
            normalized_score = (bm25_scores[idx] - min_bm25) / bm25_range
            if idx in vector_results:
                # 如果向量搜索中已经有这个结果，更新BM25分数
                vector_results[idx]['bm25_score'] = normalized_score
            else:
                # 添加BM25独有的结果
                bm25_results[idx] = {
                    'idx': idx,
                    'bm25_score': normalized_score,
                    'metadata': self.chunk_metadata[idx],
                    'vector_score': 0.0  # 向量部分给0分
                }
        
        # 3. 合并和排序结果
        combined_results = list(vector_results.values())
        
        # 为vector_results中没有BM25分数的项添加默认值
        for result in combined_results:
            if 'bm25_score' not in result:
                result['bm25_score'] = 0.0
        
        # 添加BM25独有的结果
        combined_results.extend(bm25_results.values())
        
        # 计算混合得分
        for result in combined_results:
            result['hybrid_score'] = (
                vector_weight * result['vector_score'] + 
                bm25_weight * result['bm25_score']
            )
        
        # 按混合得分排序
        sorted_results = sorted(combined_results, key=lambda x: x['hybrid_score'], reverse=True)
        
        # 取top_k个结果，并构建返回格式
        final_results = []
        for i, result in enumerate(sorted_results[:top_k]):
            metadata = result['metadata'].copy()
            metadata['vector_score'] = float(result['vector_score'])
            metadata['bm25_score'] = float(result['bm25_score'])
            metadata['hybrid_score'] = float(result['hybrid_score'])
            metadata['rank'] = i + 1
            final_results.append(metadata)
        
        return final_results


def test_embedding_functionality(model_name: str = "BAAI/bge-large-zh-v1.5", save_results: bool = True):
    """
    测试向量化功能是否正常
    
    Args:
        model_name: 用于测试的模型名称或路径
        save_results: 是否保存测试结果为JSON文件
    """
    print("开始测试向量化功能...")
    test_results = {
        "test_type": "vector_embedding",
        "model_name": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "success": False,
        "results": []
    }
    
    # 1. 创建索引构建器
    index_builder = DocumentVectorIndexBuilder(model_name=model_name, device = 'cpu')
    
    # 2. 准备测试文本
    test_texts = [
        "人工智能技术正在快速发展",
        "深度学习模型在自然语言处理领域取得了巨大成功",
        "向量数据库可以有效地存储和检索高维向量",
        "向量相似度搜索是检索相关文档的有效方法",
        "人工智能技术使得计算机能够理解和处理人类语言"
    ]
    
    try:
        # 3. 获取文本的向量表示
        print("为测试文本生成向量...")
        embeddings = index_builder._create_embeddings(test_texts)
        
        # 4. 验证向量维度和数量
        print(f"向量维度: {embeddings.shape[1]}")
        print(f"向量数量: {embeddings.shape[0]}")
        test_results["vector_info"] = {
            "dimension": int(embeddings.shape[1]),
            "count": int(embeddings.shape[0])
        }
        
        # 5. 创建临时FAISS索引
        print("创建临时FAISS索引...")
        dimension = embeddings.shape[1]
        temp_index = faiss.IndexFlatL2(dimension)
        temp_index.add(embeddings)
        
        # 6. 测试搜索功能
        print("测试搜索功能...")
        test_query = "深度学习技术在NLP领域的应用"
        query_vector = index_builder._create_embeddings([test_query])
        distances, indices = temp_index.search(query_vector, 3)
        
        # 7. 显示搜索结果
        print("\n搜索测试结果:")
        print(f"查询: '{test_query}'")
        test_results["query"] = test_query
        print("匹配的前3个文本:")
        
        search_results = []
        for i, idx in enumerate(indices[0]):
            print(f"{i+1}. 文本: '{test_texts[idx]}'")
            print(f"   距离: {distances[0][i]:.4f}")
            search_results.append({
                "rank": i + 1,
                "text": test_texts[idx],
                "distance": float(distances[0][i])
            })
        
        test_results["results"] = search_results
        test_results["success"] = True
        print("\n向量化功能测试成功!")
        
        # 保存测试结果到JSON文件
        if save_results:
            result_path = os.path.join(os.getcwd(), "vector_test_results.json")
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(test_results, f, ensure_ascii=False, indent=2)
            print(f"测试结果已保存至: {result_path}")
            
        return True
        
    except Exception as e:
        print(f"测试失败: {str(e)}")
        test_results["error"] = str(e)
        
        # 即使失败也保存结果
        if save_results:
            result_path = os.path.join(os.getcwd(), "vector_test_results_failed.json")
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(test_results, f, ensure_ascii=False, indent=2)
            print(f"失败信息已保存至: {result_path}")
            
        import traceback
        traceback.print_exc()
        return False


def test_hybrid_search(model_name: str = "BAAI/bge-large-zh-v1.5", save_results: bool = True):
    """
    测试混合搜索功能
    
    Args:
        save_results: 是否保存测试结果为JSON文件
    """
    print("\n开始测试混合搜索功能...")
    test_results = {
        "test_type": "hybrid_search",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "success": False,
        "text_corpus": [],
        "results": []
    }
    
    # 创建索引构建器
    index_builder = DocumentVectorIndexBuilder(model_name=model_name, device='cpu')
    
    # 准备测试文本
    test_texts = [
        "人工智能技术正在快速发展，深度学习是其中重要的分支。",
        "深度学习模型在自然语言处理领域取得了巨大成功，例如GPT和BERT。",
        "向量数据库可以有效地存储和检索高维向量，适合相似度搜索。",
        "全文检索引擎通过倒排索引实现关键词搜索，BM25是经典的相关性算法。",
        "混合检索结合了语义向量检索和关键词匹配的优势，提高了检索质量。"
    ]
    test_results["text_corpus"] = test_texts
    
    try:
        # 手动构建测试索引
        print("为测试文本生成向量...")
        embeddings = index_builder._create_embeddings(test_texts)
        
        # 创建临时FAISS索引
        dimension = embeddings.shape[1]
        temp_index = faiss.IndexFlatL2(dimension)
        temp_index.add(embeddings)
        index_builder.index = temp_index
        
        # 构建测试用的BM25索引
        index_builder._build_bm25_index(test_texts)
        
        # 创建测试用的元数据
        index_builder.chunk_metadata = []
        for i, text in enumerate(test_texts):
            index_builder.chunk_metadata.append({
                'filename': f'test_file_{i}.txt',
                'chunk_idx': 0,
                'total_chunks': 1,
                'text': text
            })
        
        # 测试不同权重组合的混合搜索
        test_queries = [
            "深度学习模型和自然语言处理",
            "向量检索和关键词搜索的区别",
            "人工智能在搜索引擎中的应用"
        ]
        weight_options = [0.3, 0.5, 0.7, 0.9]  # 不同的向量权重
        
        all_query_results = []
        
        for query in test_queries:
            query_result = {"query": query, "weight_tests": []}
            print(f"\n测试查询: '{query}'")
            
            for vector_weight in weight_options:
                print(f"\n使用向量权重: {vector_weight}")
                results = index_builder.hybrid_search(query, top_k=3, vector_weight=vector_weight)
                
                weight_test = {
                    "vector_weight": vector_weight,
                    "bm25_weight": 1.0 - vector_weight,
                    "search_results": []
                }
                
                print(f"查询: '{query}' (向量权重: {vector_weight})")
                print("\n匹配的前3个文本:")
                
                for i, result in enumerate(results):
                    print(f"{i+1}. 文本: '{result['text']}'")
                    print(f"   向量得分: {result['vector_score']:.4f}")
                    print(f"   BM25得分: {result['bm25_score']:.4f}")
                    print(f"   混合得分: {result['hybrid_score']:.4f}")
                    print()
                    
                    # 添加到结果
                    result_copy = {
                        "rank": i + 1,
                        "text": result['text'],
                        "filename": result['filename'],
                        "vector_score": float(result['vector_score']),
                        "bm25_score": float(result['bm25_score']),
                        "hybrid_score": float(result['hybrid_score'])
                    }
                    weight_test["search_results"].append(result_copy)
                
                query_result["weight_tests"].append(weight_test)
            
            all_query_results.append(query_result)
        
        test_results["results"] = all_query_results
        test_results["success"] = True
        print("\n混合搜索功能测试成功!")
        
        # 保存测试结果到JSON文件
        if save_results:
            result_path = os.path.join(os.getcwd(), "hybrid_search_test_results.json")
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(test_results, f, ensure_ascii=False, indent=2)
            print(f"测试结果已保存至: {result_path}")
            
        return True
        
    except Exception as e:
        print(f"测试失败: {str(e)}")
        test_results["error"] = str(e)
        
        # 即使失败也保存结果
        if save_results:
            result_path = os.path.join(os.getcwd(), "hybrid_search_test_results_failed.json")
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(test_results, f, ensure_ascii=False, indent=2)
            print(f"失败信息已保存至: {result_path}")
            
        import traceback
        traceback.print_exc()
        return False


import json

def test_vector_search(index_folder: str, 
                      prefix: str = "document_index", 
                      model_name: str = "BAAI/bge-large-zh-v1.5", 
                      top_k: int = 5,
                      save_results: bool = True,
                      output_path: str = None) -> None:
    """
    测试向量搜索功能 - 使用已有索引并允许交互式查询
    
    Args:
        index_folder: 存储索引的文件夹路径
        prefix: 索引文件前缀
        model_name: 使用的模型名称或路径
        top_k: 返回的搜索结果数量
        save_results: 是否保存测试结果
        output_path: 测试结果保存路径，为None时使用默认路径
    """
    print(f"\n开始测试向量搜索功能...")
    print(f"加载索引: {index_folder}/{prefix}")
    
    test_results = {
        "test_type": "vector_search",
        "model_name": model_name,
        "index_folder": index_folder,
        "index_prefix": prefix,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "success": False,
        "queries": []
    }
    
    try:
        # 加载索引
        index_builder = DocumentVectorIndexBuilder(model_name=model_name)
        index_builder.load_index(folder_path=index_folder, prefix=prefix)
        
        print(f"索引加载成功，包含 {len(index_builder.chunk_metadata)} 个文档块")
        
        # 交互式查询
        print("\n" + "="*50)
        print("交互式向量搜索测试 (输入'exit'或'quit'退出)")
        print("="*50)
        
        while True:
            query = input("\n请输入查询文本: ").strip()
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("测试结束")
                break
                
            if not query:
                print("查询不能为空，请重新输入")
                continue
                
            print(f"\n执行查询: '{query}'")
            
            # 执行搜索
            start_time = time.time()
            results = index_builder.search(query, top_k=top_k)
            search_time = time.time() - start_time
            
            print(f"查询耗时: {search_time:.4f} 秒")
            print(f"\n找到 {len(results)} 个匹配结果:")
            
            # 记录查询结果
            query_result = {
                "query_text": query,
                "search_time": search_time,
                "top_k": top_k,
                "results": []
            }
            
            # 显示搜索结果
            for i, result in enumerate(results):
                print(f"\n{i+1}. 相似度得分: {1 - result['distance']:.4f}")
                print(f"   文件: {result['filename']}")
                print(f"   文本片段 ({result['chunk_idx']+1}/{result['total_chunks']}): {result['text']}...")
                
                result_copy = {
                    "rank": i + 1,
                    "similarity": float(1 - result['distance']),
                    "distance": float(result['distance']),
                    "filename": result['filename'],
                    "chunk_idx": result['chunk_idx'],
                    "total_chunks": result['total_chunks'],
                    "text": result['text']
                }
                query_result["results"].append(result_copy)
            
            test_results["queries"].append(query_result)
            
            # 询问是否继续
            continue_choice = input("\n继续测试下一个查询? (y/n): ").strip().lower()
            if continue_choice != 'y':
                print("测试结束")
                break
        
        test_results["success"] = True
        
        # 保存测试结果
        if save_results and test_results["queries"]:
            if output_path is None:
                output_path = os.path.join(os.getcwd(), "vector_search_test_results.json")
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(test_results, f, ensure_ascii=False, indent=2)
            print(f"\n测试结果已保存至: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"测试失败: {str(e)}")
        test_results["error"] = str(e)
        
        # 即使失败也保存结果
        if save_results:
            if output_path is None:
                output_path = os.path.join(os.getcwd(), "vector_search_test_results_failed.json")
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(test_results, f, ensure_ascii=False, indent=2)
            print(f"\n失败信息已保存至: {output_path}")
            
        import traceback
        traceback.print_exc()
        return False

def test_hybrid_search_interactive(index_folder: str, 
                                 prefix: str = "document_index", 
                                 model_name: str = "BAAI/bge-large-zh-v1.5", 
                                 top_k: int = 5,
                                 default_vector_weight: float = 0.7,
                                 save_results: bool = True,
                                 output_folder: str = None) -> None:
    """
    交互式测试混合搜索功能 - 使用已有索引并允许交互式查询
    
    Args:
        index_folder: 存储索引的文件夹路径
        prefix: 索引文件前缀
        model_name: 使用的模型名称或路径
        top_k: 返回的搜索结果数量
        default_vector_weight: 默认的向量搜索权重 (0.0-1.0)
        save_results: 是否保存测试结果
        output_path: 测试结果保存路径，为None时使用默认路径
    """
    print(f"\n开始交互式混合搜索测试...")
    print(f"加载索引: {index_folder}/{prefix}")
    
    test_results = {
        "test_type": "hybrid_search_interactive",
        "model_name": model_name,
        "index_folder": index_folder,
        "index_prefix": prefix,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "success": False,
        "queries": []
    }
    
    try:
        # 加载索引
        index_builder = DocumentVectorIndexBuilder(model_name=model_name)
        index_builder.load_index(folder_path=index_folder, prefix=prefix)
        
        print(f"索引加载成功，包含 {len(index_builder.chunk_metadata)} 个文档块")
        
        # 交互式查询
        print("\n" + "="*50)
        print("交互式混合搜索测试 (输入'exit'或'quit'退出)")
        print("混合搜索结合了向量相似度和BM25关键词匹配")
        print("="*50)
        
        while True:
            # 获取查询文本
            query = input("\n请输入查询文本: ").strip()
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("测试结束")
                break
                
            if not query:
                print("查询不能为空，请重新输入")
                continue
            
            # 获取向量权重    
            weight_input = input(f"\n请输入向量搜索权重 (0.0-1.0，默认{default_vector_weight}): ").strip()
            vector_weight = default_vector_weight
            if weight_input:
                try:
                    vector_weight = float(weight_input)
                    if not 0 <= vector_weight <= 1:
                        print(f"权重必须在0-1之间，使用默认值: {default_vector_weight}")
                        vector_weight = default_vector_weight
                except ValueError:
                    print(f"输入格式错误，使用默认值: {default_vector_weight}")
            
            print(f"\n执行查询: '{query}' (向量权重: {vector_weight:.2f}, BM25权重: {1-vector_weight:.2f})")
            
            # 执行混合搜索
            start_time = time.time()
            results = index_builder.hybrid_search(query, top_k=top_k, vector_weight=vector_weight)
            search_time = time.time() - start_time
            
            print(f"查询耗时: {search_time:.4f} 秒")
            print(f"\n找到 {len(results)} 个匹配结果:")
            
            # 记录查询结果
            query_result = {
                "query_text": query,
                "vector_weight": vector_weight,
                "bm25_weight": 1.0 - vector_weight,
                "search_time": search_time,
                "top_k": top_k,
                "results": []
            }
            
            # 显示搜索结果
            for i, result in enumerate(results):
                print(f"\n{i+1}. 混合得分: {result['hybrid_score']:.4f}")
                print(f"   向量得分: {result['vector_score']:.4f}  BM25得分: {result['bm25_score']:.4f}")
                print(f"   文件: {result['filename']}")
                print(f"   文本片段 ({result['chunk_idx']+1}/{result['total_chunks']}): {result['text']}...")
                
                result_copy = {
                    "rank": i + 1,
                    "hybrid_score": float(result['hybrid_score']),
                    "vector_score": float(result['vector_score']),
                    "bm25_score": float(result['bm25_score']),
                    "filename": result['filename'],
                    "chunk_idx": result['chunk_idx'],
                    "total_chunks": result['total_chunks'],
                    "text": result['text']
                }
                query_result["results"].append(result_copy)
            
            # 是否尝试不同权重
            try_weight = input("\n是否尝试其他权重值进行比较? (y/n): ").strip().lower()
            if try_weight == 'y':
                while True:
                    weight_input = input("\n请输入新的向量权重 (0.0-1.0，输入空值结束): ").strip()
                    if not weight_input:
                        break
                        
                    try:
                        new_weight = float(weight_input)
                        if not 0 <= new_weight <= 1:
                            print("权重必须在0-1之间")
                            continue
                            
                        print(f"\n使用向量权重: {new_weight:.2f}, BM25权重: {1-new_weight:.2f}")
                        
                        # 执行新权重下的搜索
                        start_time = time.time()
                        new_results = index_builder.hybrid_search(query, top_k=top_k, vector_weight=new_weight)
                        new_search_time = time.time() - start_time
                        
                        print(f"查询耗时: {new_search_time:.4f} 秒")
                        print(f"\n找到 {len(new_results)} 个匹配结果:")
                        
                        # 添加到测试结果
                        new_query_result = {
                            "query_text": query,
                            "vector_weight": new_weight,
                            "bm25_weight": 1.0 - new_weight,
                            "search_time": new_search_time,
                            "top_k": top_k,
                            "results": []
                        }
                        
                        # 显示搜索结果
                        for i, result in enumerate(new_results):
                            print(f"\n{i+1}. 混合得分: {result['hybrid_score']:.4f}")
                            print(f"   向量得分: {result['vector_score']:.4f}  BM25得分: {result['bm25_score']:.4f}")
                            print(f"   文件: {result['filename']}")
                            print(f"   文本片段 ({result['chunk_idx']+1}/{result['total_chunks']}): {result['text']}...")
                            
                            result_copy = {
                                "rank": i + 1,
                                "hybrid_score": float(result['hybrid_score']),
                                "vector_score": float(result['vector_score']),
                                "bm25_score": float(result['bm25_score']),
                                "filename": result['filename'],
                                "chunk_idx": result['chunk_idx'],
                                "total_chunks": result['total_chunks'],
                                "text": result['text']
                            }
                            new_query_result["results"].append(result_copy)
                        
                        test_results["queries"].append(new_query_result)
                        
                    except ValueError:
                        print("输入格式错误，请输入0-1之间的小数")
            
            # 添加原查询结果到测试结果
            test_results["queries"].append(query_result)
            
            # 询问是否继续新的查询
            continue_choice = input("\n继续测试下一个查询? (y/n): ").strip().lower()
            if continue_choice != 'y':
                print("测试结束")
                break
        
        test_results["success"] = True
        
        # 保存测试结果
        if save_results and test_results["queries"]:
            if output_folder is not None:
                output_path = os.path.join(output_folder, "hybrid_search_interactive_results.json")
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(test_results, f, ensure_ascii=False, indent=2)
            print(f"\n测试结果已保存至: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"测试失败: {str(e)}")
        test_results["error"] = str(e)
        
        # 保存测试结果
        if save_results and test_results["queries"]:

            if output_folder is not None:
                output_path = os.path.join(output_folder, "hybrid_search_interactive_results.json")

            # 确保输出目录存在
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(test_results, f, ensure_ascii=False, indent=2)
            print(f"\n测试结果已保存至: {output_path}")
            
        import traceback
        traceback.print_exc()
        return False

# 修改main函数，添加测试选项
def main():
    parser = argparse.ArgumentParser(description='文档向量索引构建与测试工具')
    # 原有参数
    parser.add_argument('--folder', type=str, 
                        help='文档所在文件夹路径', default="/mnt/workspace/RAG/documents")
    parser.add_argument('--index_folder', type=str, default="/mnt/workspace/RAG/vectorDB",
                        help='索引保存路径')
    parser.add_argument('--index_name', type=str, default="all_index_300_100",
                        help='索引名称前缀')
    parser.add_argument('--chunk_method', type=str, default='by_chars', 
                        choices=['by_chars', 'by_sentences', 'by_paragraphs'],
                        help='分块方法')
    parser.add_argument('--max_chars', type=int, default=300,
                        help='按字符分块时的最大字符数')
    parser.add_argument('--max_sentences', type=int, default=5,
                        help='按句子分块时的最大句子数')
    parser.add_argument('--max_paragraphs', type=int, default=3,
                        help='按段落分块时的最大段落数')
    parser.add_argument('--overlap', type=int, default=100,
                        help='按字符分块时的重叠字符数')
    parser.add_argument('--overlap_sentences', type=int, default=1,
                        help='按句子分块时的重叠句子数')
    parser.add_argument('--overlap_paragraphs', type=int, default=1,
                        help='按段落分块时的重叠段落数')
    parser.add_argument('--model', type=str, default="/mnt/workspace/models/bge-large-zh-v1.5",
                        help='Embedding模型名称或路径')
    # 测试相关参数
    parser.add_argument('--test', type=str, default="hybrid_search", 
                        choices=['vector', 'hybrid', 'vector_search', 'hybrid_search'],
                        help='运行测试类型: vector (向量化功能), hybrid (混合搜索), vector_search (交互式向量搜索), hybrid_search (交互式混合搜索)')
    parser.add_argument('--top_k', type=int, default=5,
                        help='搜索返回的最大结果数量')
    parser.add_argument('--vector_weight', type=float, default=0.7,
                        help='混合搜索中向量搜索的权重 (0.0-1.0)')
    parser.add_argument('--save_results', default="true", 
                        help='是否保存测试结果为JSON文件')
    parser.add_argument('--output_path', type=str, default="/mnt/workspace/RAG/vector_test",
                        help='测试结果保存的路径')
    args = parser.parse_args()
    
    # 运行测试
    # if args.test:
    #     if args.test == 'vector':
    #         test_embedding_functionality(args.model, args.save_results)
    #     elif args.test == 'hybrid':
    #         test_hybrid_search(args.model, args.save_results)
    #     elif args.test == 'vector_search':
    #         test_vector_search(
    #             index_folder=args.index_folder,
    #             prefix=args.index_name,
    #             model_name=args.model,
    #             top_k=args.top_k,
    #             save_results=args.save_results,
    #             output_path=args.output_path
    #         )
    #     elif args.test == 'hybrid_search':
    #         test_hybrid_search_interactive(
    #             index_folder=args.index_folder,
    #             prefix=args.index_name,
    #             model_name=args.model,
    #             top_k=args.top_k,
    #             default_vector_weight=args.vector_weight,
    #             save_results=args.save_results,
    #             output_folder=args.output_path
    #         )
    #     return
    
    # 检查必要参数
    if not args.folder:
        print("错误: 必须提供文档所在文件夹路径 (--folder)")
        parser.print_help()
        return
    
    # 创建索引构建器
    index_builder = DocumentVectorIndexBuilder(model_name=args.model)
    
    try:
        # 构建索引
        start_time = time.time()
        index_builder.create_index_from_folder(
            folder_path=args.folder,
            chunk_method=args.chunk_method,
            max_chars=args.max_chars,
            max_sentences=args.max_sentences,
            max_paragraphs=args.max_paragraphs,
            overlap=args.overlap,
            overlap_sentences=args.overlap_sentences,
            overlap_paragraphs=args.overlap_paragraphs
        )
        
        # 保存索引
        index_builder.save_index(folder_path=args.index_folder, prefix=args.index_name)
        
        end_time = time.time()
        print(f"索引构建完成，耗时: {end_time - start_time:.2f} 秒")
        
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


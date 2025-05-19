import os
import sys
import time
import pickle
import logging
import numpy as np
import faiss
from typing import List, Dict
from tqdm import tqdm 
import argparse
# 添加BM25相关依赖
from rank_bm25 import BM25Okapi
import jieba
import json
from datetime import datetime

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
    
    def _tokenize_text(self, text: str) -> List[str]:
        """
        对文本进行分词，用于BM25索引
        
        Args:
            text: 需要分词的文本
            
        Returns:
            分词后的词语列表
        """
        # 对中文文本使用jieba分词
        return list(jieba.cut(text))
    
    def _tokenize_query(self, query: str) -> List[str]:
        """
        对查询文本进行分词
        
        Args:
            query: 查询文本
            
        Returns:
            分词后的词语列表
        """
        return self._tokenize_text(query)
    
    def _build_bm25_index(self, texts: List[str]) -> BM25Okapi:
        """
        构建BM25索引
        
        Args:
            texts: 文本列表
            
        Returns:
            BM25索引对象
        """
        logger.info(f"正在为 {len(texts)} 条文本构建BM25索引...")
        tokenized_corpus = [self._tokenize_text(text) for text in tqdm(texts, desc="文本分词")]
        self.tokenized_corpus = tokenized_corpus
        bm25_index = BM25Okapi(tokenized_corpus)
        return bm25_index
    
    def create_index_from_folder(self, 
                                folder_path: str, 
                                chunk_method: str = "by_chars", 
                                max_chars: int = 500,
                                max_sentences: int = 5,
                                max_paragraphs: int = 3, 
                                overlap: int = 100,
                                overlap_sentences: int = 1,
                                overlap_paragraphs: int = 1,
                                build_bm25: bool = True) -> None:
        """
        从文件夹读取文档，进行分块并创建向量索引和BM25索引
        
        Args:
            folder_path: 文档所在文件夹路径
            chunk_method: 分块方法 ("by_chars", "by_sentences", "by_paragraphs")
            max_chars: 按字符分块时的最大字符数
            max_sentences: 按句子分块时的最大句子数
            max_paragraphs: 按段落分块时的最大段落数
            overlap: 按字符分块时的重叠字符数
            overlap_sentences: 按句子分块时的重叠句子数
            overlap_paragraphs: 按段落分块时的重叠段落数
            build_bm25: 是否同时构建BM25索引
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
            
            logger.info(f"成功为 {len(all_chunks)} 个文档块创建向量索引")
            
            # 在处理完所有文本后，添加BM25索引构建
            if build_bm25:
                logger.info(f"正在为 {len(all_chunks)} 个文档块构建BM25索引...")
                self.bm25_index = self._build_bm25_index(all_chunks)
                logger.info(f"成功构建BM25索引")
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
        
        # 保存BM25索引
        if self.bm25_index is not None:
            bm25_path = os.path.join(folder_path, f"{prefix}_bm25.pkl")
            bm25_data = {
                "tokenized_corpus": self.tokenized_corpus,
                "idf": self.bm25_index.idf,
                "doc_len": self.bm25_index.doc_len,
                "avgdl": self.bm25_index.avgdl,
                "corpus_size": self.bm25_index.corpus_size,
                "epsilon": self.bm25_index.epsilon,
                "k1": self.bm25_index.k1,
                "b": self.bm25_index.b,
                "metadata": self.chunk_metadata  # 保存文本元数据映射
            }
            with open(bm25_path, 'wb') as f:
                pickle.dump(bm25_data, f)
            logger.info(f"BM25索引已保存至 {bm25_path}")
        
        # 保存元数据
        metadata = {
            "chunks": self.chunks,
            "chunk_metadata": self.chunk_metadata,
            "has_bm25_index": self.bm25_index is not None
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
        
        # 加载 BM25 索引
        bm25_path = os.path.join(folder_path, f"{prefix}_bm25.pkl")
        if os.path.exists(bm25_path):
            with open(bm25_path, 'rb') as f:
                bm25_data = pickle.load(f)
            
            self.tokenized_corpus = bm25_data["tokenized_corpus"]
            self.bm25_index = BM25Okapi(
                self.tokenized_corpus,
                idf=bm25_data["idf"],
                doc_len=bm25_data["doc_len"],
                avgdl=bm25_data["avgdl"],
                k1=bm25_data["k1"],
                b=bm25_data["b"]
            )
            logger.info(f"已加载BM25索引")
        else:
            self.bm25_index = None
            self.tokenized_corpus = None
            logger.info(f"未找到BM25索引文件，跳过加载")
        
        logger.info(f"从 {folder_path} 成功加载索引和元数据")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        使用向量搜索查询最相似的文档块
        
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
                result['similarity_score'] = float(1 / (1 + distances[0][i]))  # 将距离转换为相似度分数
                result['rank'] = i + 1
                results.append(result)
        
        return results
    
    def bm25_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        使用BM25算法搜索与查询最相似的文档块
        
        Args:
            query: 查询文本
            top_k: 返回的最相似结果数量
        
        Returns:
            包含相关文档块信息的列表
        """
        if self.bm25_index is None:
            raise ValueError("BM25索引未加载，请先加载包含BM25索引的数据或构建BM25索引")
        
         # 对查询进行分词
        tokenized_query = self._tokenize_query(query)
        logger.info(f"查询分词结果: {tokenized_query}")
        
        # 使用BM25搜索
        scores = self.bm25_index.get_scores(tokenized_query)
        
        # 获取最高分的文档块索引
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # 构建结果
        results = []
        for i, idx in enumerate(top_indices):
            if idx >= 0 and idx < len(self.chunk_metadata):
                result = self.chunk_metadata[idx].copy()
                result['bm25_score'] = float(scores[idx])
                result['similarity_score'] = float(scores[idx] / (1 + np.max(scores)))  # 归一化分数
                result['rank'] = i + 1
                results.append(result)
        
        return results
    
    def hybrid_search(self, query: str, top_k: int = 5, alpha: float = 0.5) -> List[Dict]:
        """
        混合搜索：结合向量搜索和BM25搜索结果
        
        Args:
            query: 查询文本
            top_k: 返回的最相似结果数量
            alpha: 向量搜索结果的权重 (0-1)，BM25结果权重为 (1-alpha)
        
        Returns:
            混合排序后的最相似文档块列表
        """
        if self.index is None:
            raise ValueError("向量索引未加载，请先加载索引")
        
        # 检查是否有BM25索引，如果没有，仅使用向量搜索
        if self.bm25_index is None:
            logger.warning("BM25索引未加载，将仅使用向量搜索")
            return self.search(query, top_k)
        
        # 设置较大的k值以获取更多候选结果
        candidate_k = top_k * 3
        
        # 执行向量搜索
        vector_results = self.search(query, top_k=candidate_k)
        
        # 执行BM25搜索
        bm25_results = self.bm25_search(query, top_k=candidate_k)
        
        # 合并结果并进行混合排序
        all_results = {}
        
        # 添加向量搜索结果
        for result in vector_results:
            key = (result['filename'], result['chunk_idx'])
            all_results[key] = {
                'filename': result['filename'],
                'chunk_idx': result['chunk_idx'],
                'total_chunks': result['total_chunks'],
                'text': result['text'],
                'vector_score': result['similarity_score'],
                'bm25_score': 0.0
            }
        
        # 添加BM25搜索结果
        for result in bm25_results:
            key = (result['filename'], result['chunk_idx'])
            if key in all_results:
                all_results[key]['bm25_score'] = result['similarity_score']
            else:
                all_results[key] = {
                    'filename': result['filename'],
                    'chunk_idx': result['chunk_idx'],
                    'total_chunks': result['total_chunks'],
                    'text': result['text'],
                    'vector_score': 0.0,
                    'bm25_score': result['similarity_score']
                }
        
        # 计算混合分数并排序
        results_list = list(all_results.values())
        
        # 标准化分数
        vector_scores = np.array([r['vector_score'] for r in results_list])
        bm25_scores = np.array([r['bm25_score'] for r in results_list])
        
        # 避免除零错误
        if np.max(vector_scores) > 0:
            vector_scores = vector_scores / np.max(vector_scores)
        if np.max(bm25_scores) > 0:
            bm25_scores = bm25_scores / np.max(bm25_scores)
        
        # 计算混合分数
        for i, result in enumerate(results_list):
            result['hybrid_score'] = alpha * vector_scores[i] + (1 - alpha) * bm25_scores[i]
        
        # 基于混合分数排序
        results_list.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        # 格式化最终结果
        final_results = []
        for i, result in enumerate(results_list[:top_k]):
            final_result = {
                'filename': result['filename'],
                'chunk_idx': result['chunk_idx'],
                'total_chunks': result['total_chunks'],
                'text': result['text'],
                'similarity_score': float(result['hybrid_score']),
                'vector_score': float(result['vector_score']),
                'bm25_score': float(result['bm25_score']),
                'rank': i + 1,
                'distance': 1.0 - float(result['hybrid_score'])  # 为了兼容性计算距离
            }
            final_results.append(final_result)
        
        return final_results
    
    def build_bm25_index(self) -> None:
        """
        为已加载的文档块构建BM25索引，用于在已有文档的情况下添加BM25功能
        """
        if not self.chunk_metadata:
            raise ValueError("没有已加载的文档块，请先加载文档或创建索引")
        
        # 获取所有文本块
        all_texts = [metadata['text'] for metadata in self.chunk_metadata]
        
        logger.info(f"正在为 {len(all_texts)} 个已存在的文档块构建BM25索引...")
        self.bm25_index = self._build_bm25_index(all_texts)
        logger.info("BM25索引构建完成")
    

# 修改main函数，添加BM25相关选项
def main():
    parser = argparse.ArgumentParser(description='文档向量索引构建工具')
    parser.add_argument('--folder', type=str, 
                        help='文档所在文件夹路径', default="/mnt/workspace/RAG/base_docu")
    parser.add_argument('--index_folder', type=str, default="/mnt/workspace/RAG/vector_test",
                        help='索引保存路径')
    parser.add_argument('--index_name', type=str, default="test_index_300_100",
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
    parser.add_argument('--test', action='store_true', default=False,
                        help='运行向量化功能测试')
    # 添加BM25相关参数
    parser.add_argument('--build_bm25', action='store_true', default=True,
                        help='同时构建BM25索引')
    parser.add_argument('--only_bm25', action='store_true', default=False,
                        help='仅为已有索引构建BM25索引')
    # 添加测试搜索方法的参数
    parser.add_argument('--test_search', action='store_true', default=True,
                    help='测试三种检索方法并保存结果')
    parser.add_argument('--test_queries', type=str, nargs='+', default="密码连续输入错误 授信工作人员",
                    help='测试查询列表，多个查询以空格分隔')
    parser.add_argument('--test_top_k', type=int, default=5,
                    help='测试时每个查询返回的结果数量')
    parser.add_argument('--test_alpha', type=float, default=0.5,
                    help='混合检索中向量搜索的权重 (0-1)')
    
    args = parser.parse_args()
    
    # 运行测试
    # if args.test:
    #     test_embedding_functionality(args.model)
    #     return
    
    # 测试搜索方法
    if args.test_search:
        if not args.test_queries:
            print("错误: 测试搜索方法时必须提供测试查询 (--test_queries)")
            parser.print_help()
            return
        
        test_search_methods(
            index_folder=args.index_folder,
            index_name=args.index_name,
            queries=args.test_queries,
            top_k=args.test_top_k,
            alpha=args.test_alpha
        )
        return
    
    # 创建索引构建器
    # index_builder = DocumentVectorIndexBuilder(model_name=args.model)
    
    # try:
    #     # 如果是仅构建BM25索引模式
    #     if args.only_bm25:
    #         logger.info("仅为已有索引构建BM25索引模式")
    #         # 加载现有索引
    #         index_builder.load_index(folder_path=args.index_folder, prefix=args.index_name)
    #         # 构建BM25索引
    #         index_builder.build_bm25_index()
    #         # 保存更新的索引
    #         index_builder.save_index(folder_path=args.index_folder, prefix=args.index_name)
    #         logger.info("BM25索引构建完成并保存")
    #         return
        
    #     # 检查必要参数
    #     if not args.folder:
    #         print("错误: 必须提供文档所在文件夹路径 (--folder)")
    #         parser.print_help()
    #         return
        
    #     # 构建索引
    #     start_time = time.time()
    #     index_builder.create_index_from_folder(
    #         folder_path=args.folder,
    #         chunk_method=args.chunk_method,
    #         max_chars=args.max_chars,
    #         max_sentences=args.max_sentences,
    #         max_paragraphs=args.max_paragraphs,
    #         overlap=args.overlap,
    #         overlap_sentences=args.overlap_sentences,
    #         overlap_paragraphs=args.overlap_paragraphs,
    #         build_bm25=args.build_bm25  # 传递BM25构建参数
    #     )
        
    #     # 保存索引
    #     index_builder.save_index(folder_path=args.index_folder, prefix=args.index_name)
        
    #     end_time = time.time()
    #     print(f"索引构建完成，耗时: {end_time - start_time:.2f} 秒")
        
    # except Exception as e:
    #     print(f"发生错误: {e}")
    #     import traceback
    #     traceback.print_exc()


def test_search_methods(index_folder: str, index_name: str, queries: List[str], top_k: int = 5, alpha: float = 0.5) -> None:
    """
    测试三种检索方法的效果并保存为JSON文件
    
    Args:
        index_folder: 索引文件夹路径
        index_name: 索引名称前缀
        queries: 测试查询列表
        top_k: 每个查询返回的结果数量
        alpha: 混合检索中向量搜索的权重
    """
    logger.info(f"开始测试三种检索方法: 向量检索、BM25检索和混合检索")
    
    # 加载索引
    index_builder = DocumentVectorIndexBuilder()
    index_builder.load_index(folder_path=index_folder, prefix=index_name)
    
    # 准备保存结果的数据结构
    results = {
        "metadata": {
            "test_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "index_folder": index_folder,
            "index_name": index_name,
            "top_k": top_k,
            "alpha": alpha
        },
        "queries": []
    }
    
    # 测试每个查询
    for query_idx, query in enumerate(queries):
        logger.info(f"测试查询 {query_idx+1}/{len(queries)}: {query}")
        
        query_result = {
            "query": query,
            "vector_search": [],
            "bm25_search": [],
            "hybrid_search": []
        }
        
        # 测试向量检索
        try:
            vector_results = index_builder.search(query, top_k=top_k)
            for result in vector_results:
                # 只保存必要的信息以减小文件大小
                query_result["vector_search"].append({
                    "filename": result["filename"],
                    "chunk_idx": result["chunk_idx"],
                    "rank": result["rank"],
                    "distance": result["distance"],
                    "similarity_score": result["similarity_score"],
                    "text": result["text"][:200] + "..." if len(result["text"]) > 200 else result["text"]
                })
            logger.info(f"向量检索完成，找到 {len(vector_results)} 个结果")
        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            query_result["vector_search_error"] = str(e)
        
        # 测试BM25检索
        try:
            if index_builder.bm25_index is not None:
                bm25_results = index_builder.bm25_search(query, top_k=top_k)
                for result in bm25_results:
                    query_result["bm25_search"].append({
                        "filename": result["filename"],
                        "chunk_idx": result["chunk_idx"],
                        "rank": result["rank"],
                        "bm25_score": result["bm25_score"],
                        "similarity_score": result["similarity_score"],
                        "text": result["text"][:200] + "..." if len(result["text"]) > 200 else result["text"]
                    })
                logger.info(f"BM25检索完成，找到 {len(bm25_results)} 个结果")
            else:
                logger.warning("BM25索引不可用，跳过BM25检索测试")
                query_result["bm25_search_error"] = "BM25索引不可用"
        except Exception as e:
            logger.error(f"BM25检索失败: {e}")
            query_result["bm25_search_error"] = str(e)
        
        # 测试混合检索
        try:
            if index_builder.bm25_index is not None:
                hybrid_results = index_builder.hybrid_search(query, top_k=top_k, alpha=alpha)
                for result in hybrid_results:
                    query_result["hybrid_search"].append({
                        "filename": result["filename"],
                        "chunk_idx": result["chunk_idx"],
                        "rank": result["rank"],
                        "hybrid_score": result["similarity_score"],
                        "vector_score": result["vector_score"],
                        "bm25_score": result["bm25_score"],
                        "text": result["text"][:200] + "..." if len(result["text"]) > 200 else result["text"]
                    })
                logger.info(f"混合检索完成，找到 {len(hybrid_results)} 个结果")
            else:
                logger.warning("BM25索引不可用，跳过混合检索测试")
                query_result["hybrid_search_error"] = "BM25索引不可用"
        except Exception as e:
            logger.error(f"混合检索失败: {e}")
            query_result["hybrid_search_error"] = str(e)
        
        # 添加查询结果到总结果中
        results["queries"].append(query_result)
    
    # 保存结果到JSON文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(index_folder, f"{index_name}_search_test_{timestamp}.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"测试结果已保存至: {output_file}")


if __name__ == "__main__":
    main()
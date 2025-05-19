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
            
            logger.info(f"成功为 {len(all_chunks)} 个文档块创建向量索引")
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
        
        # 保存元数据
        metadata = {
            "chunks": self.chunks,
            "chunk_metadata": self.chunk_metadata
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


def test_embedding_functionality(model_name: str = "BAAI/bge-large-zh-v1.5"):
    """
    测试向量化功能是否正常
    
    Args:
        model_name: 用于测试的模型名称或路径
    """
    print("开始测试向量化功能...")
    
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
        print("匹配的前3个文本:")
        for i, idx in enumerate(indices[0]):
            print(f"{i+1}. 文本: '{test_texts[idx]}'")
            print(f"   距离: {distances[0][i]:.4f}")
        
        print("\n向量化功能测试成功!")
        return True
        
    except Exception as e:
        print(f"测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# 修改main函数，添加测试选项
def main():
    parser = argparse.ArgumentParser(description='文档向量索引构建工具')
    parser.add_argument('--folder', type=str, 
                        help='文档所在文件夹路径', default="/mnt/workspace/RAG/documents")
    parser.add_argument('--index_folder', type=str, default="/mnt/workspace/RAG/vectorDB",
                        help='索引保存路径')
    parser.add_argument('--index_name', type=str, default="part_index",
                        help='索引名称前缀')
    parser.add_argument('--chunk_method', type=str, default='by_chars', 
                        choices=['by_chars', 'by_sentences', 'by_paragraphs'],
                        help='分块方法')
    parser.add_argument('--max_chars', type=int, default=500,
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
    
    
    args = parser.parse_args()
    
    # 运行测试
    if args.test:
        test_embedding_functionality(args.model)
        return
    
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
import os
from docx import Document
import json
# from nltk.tokenize import sent_tokenize

def read_docx(file_path):
    """读取 Word 文档文本"""
    doc = Document(file_path)
    return '\n'.join([para.text.strip() for para in doc.paragraphs if para.text.strip()])

def chunk_by_chars(text, max_chars=500, overlap=100):
    """按固定字数分块"""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        start += max_chars - overlap
    return chunks

def chunk_by_sentences(text, max_sentences=5, overlap=1):
    """按句子数分块"""
    sentences = sent_tokenize(text)
    chunks = []
    i = 0
    while i < len(sentences):
        chunk = sentences[i:i + max_sentences]
        chunks.append(' '.join(chunk))
        i += max_sentences - overlap
    return chunks

def chunk_by_paragraphs(text, max_paragraphs=3, overlap=1):
    """按段落数分块"""
    paragraphs = text.split('\n')
    paragraphs = [p for p in paragraphs if p.strip()]
    chunks = []
    i = 0
    while i < len(paragraphs):
        chunk = paragraphs[i:i + max_paragraphs]
        chunks.append('\n'.join(chunk))
        i += max_paragraphs - overlap
    return chunks

def process_folder(folder_path, chunk_method="by_chars", **kwargs):
    """处理文件夹中的所有 Word 文档，并使用指定方式进行分块"""
    chunk_fn = {
        "by_chars": chunk_by_chars,
        "by_sentences": chunk_by_sentences,
        "by_paragraphs": chunk_by_paragraphs
    }.get(chunk_method)

    if chunk_fn is None:
        raise ValueError(f"不支持的分块方法: {chunk_method}")

    all_chunks = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".docx"):
            file_path = os.path.join(folder_path, filename)
            text = read_docx(file_path)
            chunks = chunk_fn(text, **kwargs)
            all_chunks[filename] = chunks

    return all_chunks

def save_chunks_to_json(all_chunks, output_path):
    """将分块结果保存为JSON文件
    
    Args:
        all_chunks (dict): 文件名到分块列表的映射
        output_path (str): 输出JSON文件的路径
    """
    # 创建一个更适合查看的结构化输出
    result = {
        "metadata": {
            "total_files": len(all_chunks),
            "total_chunks": sum(len(chunks) for chunks in all_chunks.values()),
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "documents": {}
    }
    
    # 为每个文档添加分块信息
    # for filename, chunks in all_chunks.items():
    #     result["documents"][filename] = {
    #         "total_chunks": len(chunks),
    #         "chunks": [{"id": i, "content": chunk} for i, chunk in enumerate(chunks)]
    #     }
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存为JSON文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    return output_path


if __name__ == "__main__":
    folder_path = "/mnt/workspace/RAG/documents"
    chunk_method = "by_chars"  # 可选: "by_sentences", "by_paragraphs"
    max_chars = 400
    
    max_sentences = 5
    max_paragraphs = 3
    overlap = 100
    overlap_sentences = 1
    overlap_paragraphs = 1
    all_chunks = process_folder(folder_path, chunk_method=chunk_method, max_chars=max_chars)
    # all_chunks = process_folder(folder_path, chunk_method="by_sentences", max_sentences=max_sentences, overlap=overlap_sentences)
    # all_chunks = process_folder(folder_path, chunk_method="by_paragraphs", max_paragraphs=max_paragraphs, overlap=overlap_paragraphs)

    import datetime
    output_dir = "/mnt/workspace/RAG/text_split"
    output_filename = f"chunks_{chunk_method}_max{max_chars}_overlap{overlap}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path = os.path.join(output_dir, output_filename)
    
    # 保存为JSON文件
    json_path = save_chunks_to_json(all_chunks, output_path)

    for filename, chunks in all_chunks.items():
        print(f"文件: {filename}")
        print(f"块数: {len(chunks)}")
        for i, chunk in enumerate(chunks[:2]):
            print(f"++++++块 {i + 1}: {chunk}...")  # 仅打印前50个字符
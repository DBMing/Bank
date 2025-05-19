import json
import os
import sys
import re
from tqdm import tqdm
import asyncio
from concurrent.futures import ThreadPoolExecutor

# 设置路径和导入
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "utils"))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "RAG", "embedding"))
from vllm_server import generate_chat_completion  # 保持原接口不变（同步）
from search_document_index import DocumentSearcher  # 导入文档搜索器

# 初始化文档搜索器
base_path = os.path.dirname(os.path.dirname(__file__))
document_searcher = DocumentSearcher(
    index_folder="/mnt/workspace/RAG/vectorDB",
    index_prefix="all_index_400_100"
)

# ========== 同步工具函数 ==========
def read_questions(file_path):
    questions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                question_data = json.loads(line)
                questions.append(question_data)
    return questions

def extract_answer(response, question_type):
    if question_type == "选择题":
        match = re.search(r'最终答案[是为]*：?\s*\[([A-D](?:,\s*[A-D])*)\]', response)
        if match:
            return [opt.strip() for opt in match.group(1).split(',')]
        else:
            return []
    else:
        match = re.search(r'最终答案[是为]*：\s*(.+)', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            return ""

def save_results(results, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

# ========== 检索相关文档段落 ==========
def retrieve_relevant_passages(query, top_k=5):
    """
    检索与查询相关的文档段落
    
    Args:
        query: 查询文本
        top_k: 返回的结果数量
        
    Returns:
        格式化的检索段落文本
    """
    try:
        results = document_searcher.search(query, k=top_k)
        
        # 格式化检索结果
        formatted_passages = []
        for i, result in enumerate(results):
            passage = f"【段落{i+1}】(来源:{result['filename']})\n{result['text']}"
            formatted_passages.append(passage)

        # print(query)
        # x = "\n\n".join(formatted_passages)
        # print(x)
        
        return "\n\n".join(formatted_passages)
    except Exception as e:
        print(f"检索文档段落时发生错误: {str(e)}")
        return "【检索失败】未能获取相关文档段落"

# ========== 处理单个问题 ==========
def sync_process_question(question_data):
    question_id = question_data['id']
    question_category = question_data['category']
    question_text = question_data['question']
    content = question_data.get('content', '')
    
    # 构建查询文本
    query_text = question_text
    if content:
        query_text += " " + content
    
    # 检索相关段落
    retrieved_passages = retrieve_relevant_passages(query_text, top_k=5)
    # print(retrieve_relevant_passages)

    if content:
        option_prompt = """你现在是一位逻辑严谨、具备文献检索能力的答题助手。请根据下列不定项选择题，在**理解题干和选项的基础上，结合提供的多个检索段落**，找出所有可能的正确选项。

* 所提供的检索文本可能包含多个信息段落，其中一些可能无关，请先**识别与问题高度相关的段落**用于后续分析。
* 请对每个选项逐一分析，并说明判断依据来自哪个检索段落（可用编号或简要引文标注）。
* 所有推理**必须明确基于检索段落中的内容**，不可凭空引入外部知识。
* 如某个选项的信息在检索段落中缺失、模糊或相互矛盾，请标注为"不确定"，并说明原因。
* 最终答案按照选项之间用英文逗号分隔的格式输出，例如 `[A, C]`。

题目：{题干内容}  
选项：{选项内容}
检索段落：{检索信息}

回答格式示例：
A. 【正确/错误】, 理由是…
B. 【正确/错误】, 理由是…
C. 【正确/错误】, 理由是…  
D. 【正确/错误】, 理由是…

最终答案是：[正确选项1, 正确选项2]
"""
        prompt = option_prompt.format(题干内容=question_text, 选项内容=content, 检索信息=retrieved_passages)
    else:
        write_prompt = """你是一位**严谨的知识型助手**，请认真解答以下简答题。在作答过程中，**请充分利用所提供的多个检索段落**，筛选出与题目最相关的信息，并据此进行推理与作答。

* 请首先对题目进行深入分析，明确题干涉及的概念、背景知识与逻辑路径。
* 在分析过程中，**逐段审视检索内容，挑选与问题直接相关的信息加以引用与整合**。
* 请注意：检索段落中可能包含无关信息，你必须**基于题意判断哪些段落真正提供了有效依据**。
* 如遇信息不足或矛盾，请说明不确定性来源，并结合已有信息做出合理推理。
* 请使用"分析过程："开头详细描述你的推理过程，并注明所参考的检索段落（例如："根据段落2指出………"）。
* 最后使用"最终答案是："明确总结你得出的结论。

题目：{题干内容}
检索段落：{检索信息}


回答格式示例：
分析过程：【分析过程的详细描述】
最终答案是：最终答案
"""
        prompt = write_prompt.format(题干内容=question_text, 检索信息=retrieved_passages)

    responses = generate_chat_completion(prompt)
    if not responses:
        answer = "ERROR"
    else:
        answer = extract_answer(responses[0], question_category)

    # 返回两个结构
    question_all = question_text + (content if content is not None else "")
    result_summary = {"id": question_id, "answer": answer}
    result_full = {"id": question_id, "question": question_all, "response": responses[0], "answer": answer}
    return result_summary, result_full

# ========== 异步封装 ==========
async def async_process_questions(questions, concurrency=20):
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=concurrency)

    tasks = [loop.run_in_executor(executor, sync_process_question, q) for q in questions]

    summary_results = []
    full_results = []
    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        summary, full = await future
        summary_results.append(summary)
        full_results.append(full)
    return summary_results, full_results

# ========== 准备索引 ==========
def prepare():
    try:
        if not document_searcher.loaded:
            document_searcher.load_index()
    except Exception as e:
        print(f"加载索引失败: {str(e)}")
        raise

# ========== 异步主函数 ==========
async def async_main():
    base_path = os.path.dirname(os.path.dirname(__file__))
    input_file = os.path.join(base_path, "data", "testA.json")
    output_file = os.path.join(base_path, "data", "predictA005.json")
    full_output_file = os.path.join(base_path, "data", "response_A005.json")  # 新增

    print("读取问题中...")
    questions = read_questions(input_file)
    print(f"共读取到 {len(questions)} 个问题")
    
    prepare()  # 准备索引

    print("并发处理问题中...")
    summary_results, full_results = await async_process_questions(questions, concurrency=30)

    print("保存结果...")
    save_results(summary_results, output_file)
    save_results(full_results, full_output_file)  # 新增保存完整response

    print(f"完成！结果已保存到：\n- 答案文件：{output_file}\n- 完整回答文件：{full_output_file}")


# ========== 启动入口 ==========
if __name__ == "__main__":
    asyncio.run(async_main())

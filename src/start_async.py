import json
import os
import sys
import re
from tqdm import tqdm
import asyncio
from concurrent.futures import ThreadPoolExecutor

# 设置路径和导入
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "utils"))
from vllm_server import generate_chat_completion  # 保持原接口不变（同步）

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

# ========== 处理单个问题 ==========
def sync_process_question(question_data):
    question_id = question_data['id']
    question_category = question_data['category']
    question_text = question_data['question']
    content = question_data.get('content', '')

    if content:
        option_prompt = """你现在是一位逻辑严谨的答题助手，请根据以下不定项选择题，在理解题干和选项的基础上，找出所有可能的正确选项。  
- 多个选项可能正确，请逐一分析每个选项是否符合题干要求。  
- 回答时请首先逐项给出你的判断结果。  
- 如果某个选项无法判断，也请说明理由，并指出你的不确定性来源。
- 最终答案按照选项之间用逗号分隔的格式进行输出，例如"[A, C]"。

题目：{题干内容}  
选项：{选项内容}

回答格式示例：
A. 【正确/错误】, 理由是…
B. 【正确/错误】, 理由是…
C. 【正确/错误】, 理由是…  
D. 【正确/错误】, 理由是…

最终答案是：[正确选项1, 正确选项2]
"""
        prompt = option_prompt.format(题干内容=question_text, 选项内容=content)
    else:
        write_prompt = """你是一位严谨的知识型助手，请认真解答以下简答题。

- 请先对题目进行深入分析，理清相关概念、背景知识和推理逻辑。
- 在推理分析完成后，再明确给出最终答案。
- 请使用“分析过程：”开头详细说明你的思考过程。
- 最后使用“最终答案：”明确总结你得出的答案。

题目：
{题干内容}


回答格式示例：
分析过程：【分析过程的详细描述】
最终答案是：【最终答案】
"""
        prompt = write_prompt.format(题干内容=question_text)

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

# ========== 异步主函数 ==========
async def async_main():
    base_path = os.path.dirname(os.path.dirname(__file__))
    input_file = os.path.join(base_path, "data", "testA.json")
    output_file = os.path.join(base_path, "data", "predictA001.json")
    full_output_file = os.path.join(base_path, "data", "full_response_A001.json")  # 新增

    print("读取问题中...")
    questions = read_questions(input_file)
    print(f"共读取到 {len(questions)} 个问题")

    print("并发处理问题中...")
    summary_results, full_results = await async_process_questions(questions, concurrency=30)

    print("保存结果...")
    save_results(summary_results, output_file)
    save_results(full_results, full_output_file)  # 新增保存完整response

    print(f"完成！结果已保存到：\n- 答案文件：{output_file}\n- 完整回答文件：{full_output_file}")


# ========== 启动入口 ==========
if __name__ == "__main__":
    asyncio.run(async_main())

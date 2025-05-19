import json
import os
import sys
import re
import ast
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "utils"))
from vllm_server import generate_chat_completion

def read_questions(file_path):
    """读取JSON文件中的问题"""
    questions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                question_data = json.loads(line)
                questions.append(question_data)
    return questions

def extract_answer(response, question_type):
    """从生成的回答中提取答案"""
    if question_type == "选择题":
        # 匹配 “最终答案是：[A, B, C]” 的部分
        match = re.search(r'最终答案[是为]*：?\s*\[([A-D](?:,\s*[A-D])*)\]', response)
        if match:
            # 返回 ['A', 'B', 'C'] 形式的列表
            return [opt.strip() for opt in match.group(1).split(',')]
        else:
            return []
    else:
        # 匹配 “最终答案：XXXX” 的部分，捕获冒号后的内容
        match = re.search(r'最终答案[是为]*：\s*(.+)', response)
        if match:
            return match.group(1).strip()
        else:
            return ""

def process_questions(questions):
    """处理问题并生成答案"""
    results = []
    
    for i, question_data in tqdm(enumerate(questions[:5])):
        question_id = question_data['id']
        question_category = question_data['category']
        question_text = question_data['question']
        content = question_data.get('content', '')
        
        # 构建提示
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
        
        print("============\n"+prompt)
        
        # 调用生成函数
        responses = generate_chat_completion(prompt)
        print("============\n")
        print(responses)
        print("============\n")
        # # 提取答案
        answer = extract_answer(responses[0], question_category)
        results.append({"id": question_id, "category":question_category, "answer": answer})
        
        # # 打印进度
        # if (i + 1) % 10 == 0:
        #     print(f"已处理 {i + 1}/{len(questions)} 个问题")
    
    return results

def save_results(results, output_file):
    """保存结果到JSON文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

def main():
    input_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "testA.json")
    output_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "predictA001.json")
    
    print("读取问题中...")
    questions = read_questions(input_file)
    print(f"共读取到 {len(questions)} 个问题")
    # print(questions[:3])
    
    print("处理问题并生成答案...")
    results = process_questions(questions)
    
    print("保存结果到文件...")
    save_results(results, output_file)
    
    print(f"完成！结果已保存到 {output_file}")

if __name__ == "__main__":
    main()
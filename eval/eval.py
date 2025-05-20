import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jieba
import re

def load_json_file(file_path):
    """加载并解析JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # 跳过空行
                data.append(json.loads(line))
    return data

def evaluate_choice_question(user_answer, standard_answer):
    """评估选择题
    Args:
        user_answer: 用户提交的答案列表，如 ["A", "C"]
        standard_answer: 标准答案列表，如 ["A", "C", "D"]
    Returns:
        分数: 1 (完全正确) 或 0 (有错误)
    """
    # 确保答案格式一致且为集合，以便进行比较
    if not isinstance(user_answer, list) or not isinstance(standard_answer, list):
        return 0
    
    # 转换为集合进行比较
    user_set = set(user_answer)
    standard_set = set(standard_answer)
    
    # 完全匹配才给分
    return 1 if user_set == standard_set else 0

def preprocess_text(text):
    """对文本进行预处理"""
    if not isinstance(text, str):
        return ""
    
    # 移除数字标号和标点
    text = re.sub(r'[0-9①②③④⑤⑥⑦⑧⑨⑩]+[、：\.。，,；\s]', '', text)
    # 移除常见的标点符号
    text = re.sub(r'[，。；：""''！？、．,.:;!?()（）【】\[\]{}《》<>]', '', text)
    # 分词
    words = jieba.cut(text)
    return ' '.join(words)

def evaluate_qa_question(user_answer, standard_answer):
    """评估问答题的相似度
    Args:
        user_answer: 用户提交的答案文本
        standard_answer: 标准答案文本
    Returns:
        相似度分数: 0-1之间的浮点数
    """
    if not isinstance(user_answer, str) or not isinstance(standard_answer, str):
        return 0
    
    # 预处理文本
    user_text = preprocess_text(user_answer)
    standard_text = preprocess_text(standard_answer)
    
    if not user_text or not standard_text:
        return 0
    
    # 使用TF-IDF向量化文本
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform([user_text, standard_text])
        # 计算余弦相似度
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity
    except Exception as e:
        print(f"计算相似度时出错: {e}")
        return 0

def main():
    """主函数"""
    # 加载标准答案和待评估答案
    
    train_answer_path = '/mnt/workspace/data/train_answer.json'
    part_train_answer_path = '/mnt/workspace/data/part_train_answer3.json'
    
    train_answers = load_json_file(train_answer_path)
    part_train_answers = load_json_file(part_train_answer_path)
    
    # 创建ID到标准答案的映射
    standard_answer_map = {item['id']: item for item in train_answers}
    
    # 评估结果
    choice_scores = []
    total_choice_questions = 0
    correct_choice_questions = 0
    qa_similarities = []
    qa_total_score = 0  # 问答题总分
    
    for user_item in part_train_answers:
        question_id = user_item['id']
        
        if question_id not in standard_answer_map:
            print(f"警告: 标准答案中找不到ID={question_id}的题目")
            continue
        
        standard_item = standard_answer_map[question_id]
        user_answer = user_item.get('answer', [])
        standard_answer = standard_item.get('answer', [])
        
        # 根据题型评估
        if standard_item.get('category') == '选择题':
            total_choice_questions += 1
            score = evaluate_choice_question(user_answer, standard_answer)
            correct_choice_questions += score
            choice_scores.append(score)
        else:  # 问答题
            similarity = evaluate_qa_question(user_answer, standard_answer)
            qa_similarities.append(similarity)
            qa_total_score += similarity  # 累加问答题相似度作为总分
    
    # 计算最终得分
    choice_accuracy = correct_choice_questions / total_choice_questions if total_choice_questions > 0 else 0
    avg_qa_similarity = np.mean(qa_similarities) if qa_similarities else 0
    
    # 输出结果
    print(f"选择题总数: {total_choice_questions}")
    print(f"选择题正确数: {correct_choice_questions}")
    print(f"选择题准确率: {choice_accuracy:.4f}")
    print(f"问答题数量: {len(qa_similarities)}")
    print(f"问答题平均相似度: {avg_qa_similarity:.4f}")
    print(f"问答题总分: {qa_total_score:.4f}")
    
    # # 输出每道题的评估结果
    # print("\n详细评估结果:")
    # for user_item in part_train_answers:
    #     question_id = user_item['id']
    #     if question_id in standard_answer_map:
    #         standard_item = standard_answer_map[question_id]
    #         category = standard_item.get('category', '未知类型')
    #         user_answer = user_item.get('answer', [])
    #         standard_answer = standard_item.get('answer', [])
            
    #         if category == '选择题':
    #             score = evaluate_choice_question(user_answer, standard_answer)
    #             print(f"ID={question_id}, 类型=选择题, 用户答案={user_answer}, 标准答案={standard_answer}, 得分={score}")
    #         else:
    #             similarity = evaluate_qa_question(user_answer, standard_answer)
    #             print(f"ID={question_id}, 类型=问答题, 相似度={similarity:.4f}")

if __name__ == "__main__":
    main()
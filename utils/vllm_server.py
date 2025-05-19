from openai import OpenAI

model_path = "/mnt/workspace/models/Qwen3-8B/"

def generate_chat_completion(message, model=model_path, temperature=0.6, top_p=0.95, max_tokens=8192, repetition_penalty=1.05, n=1, stop_tokens=[]):
    client = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8080/v1",
    )
    
    messages = [
        {"role": "user", "content": message}
    ]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            n=n
            # extra_body={
            #     "top_k": 20, 
            #     "chat_template_kwargs": {"enable_thinking": False},
            # }
        )
        
        texts = [choice.message.content for choice in response.choices]
        return texts
    except Exception as e:
        return []

def generate_completion(prompt, model=model_path, temperature=0.7, top_p=0.8, max_tokens=8192, repetition_penalty=1.05, n=1, stop_tokens=[]):
    client = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8080/v1",
    )
    
    try:
        response = client.completions.create(
            model=model,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            enable_thinking=False,
            extra_body={
                "repetition_penalty": repetition_penalty,
            },
            n=n,
            stop = stop_tokens
        )
        
        texts = [choice.text for choice in response.choices]
        
        stop_reason = [choice.stop_reason for choice in response.choices]
        
        # print(stop_reason)
        return texts, stop_reason
    except Exception as e:
        print(e)
        return [], []

# 示例调用
if __name__ == "__main__":
    # import json
    # with open("/root/autodl-fs/CoA/exp/exp2_coa/exp2_coa_5.json", "r") as f:
    #     data = json.load(f)
    
    # x = ""
    # for i in data[:4]:
    #     for r in i["reasoning"]:
    #         x += r
    # yy = generate_completion(x)
    # print(yy)
    completion_results = generate_chat_completion('who are you?')
    for text in completion_results:
        print(text)
        print("---------------------------------------------------")
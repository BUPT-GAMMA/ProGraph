# import package
import os
import random
import time
import csv
import requests
from tqdm import tqdm
import openai
from tenacity import retry, wait_exponential, stop_after_attempt
import threading
import json
import pandas as pd

# set openai api key
openai.api_key = ''
openai.base_url = ''


model_name_gpt4_0409 = 'gpt-4-turbo-2024-04-09'
model_name_gpt_4omini = 'gpt-4o-mini'
model_name_gpt4_1106 = 'gpt-4-1106-preview'
model_name_gpt3_5_0125 = "gpt-3.5-turbo"
model_name_claude_opus = 'claude-3-opus-20240229'
model_name_claude_sonnet = 'claude-3-sonnet-20240229'
model_name_claude_haiku = 'claude-3-haiku-20240307'

model_use = model_name_gpt_4omini

@retry(wait=wait_exponential(multiplier=1, max=360), stop=stop_after_attempt(50))
def gpt_check_answer(question, model_name=model_use):
    # 调用 GPT-3.5-turbo-1106 API 生成回答

    # 3.5 turbo: gpt-3.5-turbo-1106
    # 4 turbo: gpt-4-1106-preview
    completion = openai.chat.completions.create(
        model=model_name,  # 使用 GPT-3.5-turbo-1106 模型
        messages=[
            {"role": "system", "content": "You are a teacher who is marking test papers. I will give you a question, reference answer(s) and the student’s answer. You need to understand the question, and based on the reference answer(s), decide if the student’s answer is correct. Please note that there may be more than one reference answer, and the student’s answer only needs to be completely consistent with any one of them to be considered correct."},
            {"role": "system", "content": "You need to give me 1 or 0 based on the student's answer.\
                                           You need to response in json format.\
                                           Your check_result_score should belong to [0, 1]. The max check_result_score is 1.\
                                           reason(str): your reason for the check_result.\
                                           check_result_score(float): 1 or 0"},
            {"role": "user", "content": question},
        ],
        
        temperature = 0.0,
        response_format={"type": "json_object"},
    )
    return completion.choices[0].message.content.strip()

@retry(wait=wait_exponential(multiplier=1, max=360), stop=stop_after_attempt(50))
def gpt_check_code(question, model_name=model_use):
    # 调用 GPT-3.5-turbo-1106 API 生成回答

    # 3.5 turbo: gpt-3.5-turbo-1106
    # 4 turbo: gpt-4-1106-preview
    completion = openai.chat.completions.create(
        model=model_name,  # 使用 GPT-3.5-turbo-1106 模型
        messages=[
            {"role": "system", "content": "You are a teacher who is marking test papers. I will give you a question, reference code and the student’s code. \
                                           You need to understand the question, and based on the reference code, decide if the student’s code is correct. \
                                           You just need to check the key api calls. If the key api calls are correct, then the code is correct, the score is 1.\
                                           If all the key api calls are wrong, then the code is wrong, the score is 0.\
                                           If the key api number is n, the student's code contains m key api calls, and m < n, then the score is m/n."},
            {"role": "system", "content": "You need to give me 1( right), m/n(when student api number m < request api number n) or 0(wrong) based on the student's answer.\
                                           Your check_result_score should belong to [0, 1]. The max check_result_score is 1.\
                                           You need to response in json format.\
                                           reason(str): your reason for the check_result.\
                                           check_result_score(float): [0, 1] 1 or 0 or m/n (when student api number m < request api number n, float)\
                                           "},
            {"role": "user", "content": question},
        ],

        temperature = 0.0,
        response_format={"type": "json_object"},
    )
    return completion.choices[0].message.content.strip()

def check_answer(i, exec_result_col=12):
    question = df.iloc[i, 1]
    reference_answer = df.iloc[i, 2]
    student_answer = df.iloc[i, exec_result_col]
    
    # 如果exec_result是字符串且字符数量超过10000，则只取前1000个字符
    if isinstance(student_answer, str):
        if len(student_answer) > 10000:
            student_answer = student_answer[:1000]
    
    if 'error' in str(student_answer) or 'Error' in str(student_answer):
        return json.dumps({'reason': 'The student answer contains error message.','check_result_score': 0.0})
    else:
        prompt = f"""
        Question:
        {question}

        Reference Answer:
        {reference_answer}

        Student Answer:
        {student_answer}
        """
        
        response = gpt_check_answer(prompt, model_use)
        return response

def check_code(i, student_code_col=11, exec_result_col=12):
    question = df.iloc[i, 1]
    reference_code = df.iloc[i, 2]
    student_code = df.iloc[i, student_code_col]
    key_api_calls = df.iloc[i, 5]
    exec_result = df.iloc[i, exec_result_col]
    
    # 如果exec_result是字符串且字符数量超过10000，则只取前1000个字符
    if isinstance(exec_result, str):
        if len(exec_result) > 10000:
            exec_result = exec_result[:1000]
    
    if 'error' in str(exec_result).lower():
        return json.dumps({'reason': 'The student answer contains error message.','check_result_score': 0.0})
    else:
        prompt = f"""
        Question:
        {question}

        Reference Code:
        {reference_code}

        Student Code:
        {student_code}
        
        Key API Calls:
        {key_api_calls}
        """
        
        response = gpt_check_code(prompt, model_use)
        return response

def extract_result(output):
    return [json.loads(res)['check_result_score'] for res in output]

def extract_reason(output):
    return [json.loads(res)['reason'] for res in output]
    
def check_threaded(df, i, results, progress_bar):
    if df.iloc[i, 6] == 'check_code':
        results[i] = check_code(i)
    else:
        results[i] = check_answer(i)
    progress_bar.update(1)
    
def run_threaded_check(df, start_idx, end_idx, results, progress_bar):
    for i in range(start_idx, end_idx):
        check_threaded(df, i, results, progress_bar)

def check_answers_by_gpt4(df):
    num_threads = 20
    results = [None] * len(df)
    threads = []

    chunk_size = (len(df) + num_threads - 1) // num_threads

    global_progress_bar = tqdm(total=len(df), desc="Total Progress")

    for i in range(num_threads):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(df))

        thread = threading.Thread(target=run_threaded_check, args=(df, start_idx, end_idx, results, global_progress_bar))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    global_progress_bar.close()

    return results

# 第一个目录
directory = 'graph_tool_agent\\benchmark\\results\\gpt-4o-mini'
dataframes = {}

for filename in os.listdir(directory):
    if filename.startswith('exec') and filename.endswith('.csv'):
        # 获取 exec_ 后的第一个字符
        name_key = filename.split('_')[1]
        file_path = os.path.join(directory, filename)
        
        # 读取 CSV 文件
        df = pd.read_csv(file_path)
        
        # 存储到字典中，以 'df_' + 第一个字符 为键
        dataframe_name = f'df_{name_key}'
        dataframes[dataframe_name] = df
        
total_question_number = 0
total_right_number = 0

# 创建一个文件gpt_check_results_gpt4_only.txt
# 设置文件名
# 第二个存储结果的文件名
file_path = 'graph_tool_agent\\benchmark\\evaluation\\code\\gpt_check_results_gpt-4o-mini.txt'

# 检查文件是否存在
if not os.path.exists(file_path):
    # 如果文件不存在，则创建它
    with open(file_path, 'w') as file:
        pass
    print(f"File {file_path} was created.")
else:
    # 如果文件已存在
    print(f"File {file_path} already exists.")

# 打印结果，展示所有加载的 DataFrame 及其对应的变量名
for name, df in dataframes.items():
    print(f"DataFrame name: {name}, Number of rows: {len(df)}")
    
    total_question_number += len(df)
    
    gpt_check_answers = check_answers_by_gpt4(df)
    
    check_results = extract_result(gpt_check_answers)
    
    # print(check_results)

    true_num = sum(check_results)
    
    total_right_number += true_num

    # 计算占比
    total_results = len(check_results)
    true_proportion = true_num / total_results if total_results > 0 else 0

    check_reasons = extract_reason(gpt_check_answers)
    
    df['gpt_check_result'] = check_results
    df['gpt_check_reason'] = check_reasons
    
    # 提取name中_后的字符串
    new_file_name = f"gpt_check_{name.split('_')[1]}.csv"
    
    # 输出结果
    print(f"{name.split('_')[1]} Number of 'True' in all forms:", true_num)
    print(f"{name.split('_')[1]} Proportion of 'True':", true_proportion)
    
    with open(file_path, 'a') as f:
        f.write(f"{name.split('_')[1]} Number of 'True' in all forms: {true_num}\n")
        f.write(f"{name.split('_')[1]} Proportion of 'True': {true_proportion}\n")
    
    df.to_csv(os.path.join(directory, new_file_name), index=False, encoding='utf-8')


print(f"Total Number of Questions: {total_question_number}")
print(f"Total Number of Right Questions: {total_right_number}")
print(f"Total Proportion of Right Questions: {total_right_number / total_question_number}")

with open(file_path, 'a') as f:
    f.write(f"Total Number of Questions: {total_question_number}\n")
    f.write(f"Total Number of Right Questions: {total_right_number}\n")
    f.write(f"Total Proportion of Right Questions: {total_right_number / total_question_number}\n")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
import pandas as pd
import numpy as np
import re
import math
import os
import os.path
import csv
from tqdm import tqdm
import threading
import glob
from tenacity import retry, wait_exponential, stop_after_attempt
import argparse

os.environ['OPENAI_API_KEY'] = ''
os.environ['OPENAI_API_BASE'] = ''
os.environ['OPENAI_BASE_URL'] = ''

os.environ['INDEX_BUILD_TYPE'] = 'json'
os.environ['RAG_DATA'] = 'data'
os.environ['RAG_LLM_TEMPERATURE'] = '0.3'

from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)

PERSIST_DIR = "./storageGPT"

if PERSIST_DIR is not None:
        if not os.path.exists(PERSIST_DIR):
            from llama_index.core.node_parser import SentenceSplitter

            # Local settings
            INDEX_BUILD_TYPE = os.environ.get("INDEX_BUILD_TYPE")
            documents = SimpleDirectoryReader(os.environ.get("RAG_DATA"), recursive=True).load_data()
            if INDEX_BUILD_TYPE == "documents":
                index = VectorStoreIndex.from_documents(
                    documents,
                    llm=OpenAI(
                        temperature=float(os.environ.get("RAG_LLM_TEMPERATURE")),
                        # embed_model=global_config["tool_agent"]["rag"]["llm"][
                        #     "model"
                        # ],
                    ),  # default 0.1 gpt-3.5-turbo
                )
                index = VectorStoreIndex.from_documents(documents)
            elif INDEX_BUILD_TYPE == "json":
                from llama_index.core.node_parser import JSONNodeParser

                parser = JSONNodeParser()

                nodes = parser.get_nodes_from_documents(documents)
                index = VectorStoreIndex(nodes)
            index.storage_context.persist(persist_dir=PERSIST_DIR)
        else:
            storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            index = load_index_from_storage(storage_context)

@retry(wait=wait_exponential(multiplier=1, max=120), stop=stop_after_attempt(50))
def rag_query(question, similarity_top_k):
    retriever = index.as_retriever(
        retriever_mode="llm",
        similarity_top_k=similarity_top_k
    )
    nodes = retriever.retrieve(question)
    ret = [item.get_text() for item in nodes]
    ret = "\n".join(ret)
    
    return (
            "according to the query and documents, the relative infos retrieved by rag are \n\n"
            + ret
            + "."
        )

device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

model_name_or_path_all = {
    'llama3': '/data/private/chenweize/checkpoints/Meta-Llama-3-8B-Instruct',
    'deepseek': '/data/private/lixin/deepseek-7B'
}

model_settings = {
    'doc and code',
    'code only',
    'no fine-tuning'
}

model_settings_llama3 = {
    'doc and code': '/home/lixin/graph-agent/data/zephyr-7b-sft-qlora-llama3-new-code-data-pro-new-0524',
    'code only': '/home/lixin/graph-agent/data/zephyr-7b-sft-qlora-llama-old-code-data-0521'
}

model_settings_deepseek = {
    'doc and code': '/home/lixin/graph-agent/data/zephyr-7b-sft-qlora-deepseek-new-code-data-pro-new-0525',
    'code only': '/home/lixin/graph-agent/data/zephyr-7b-sft-qlora-deepseek-old-code-data-0521'
}

lib_names = [
    'networkx',
    'igraph',
    'cdlib',
    'graspologic',
    'karateclub',
    'littleballoffur'
]

answer_difficulties = [
    'easy',
    'hard'
]

catogories = [
    'basic graph theory',
    'graph statistic learning',
    'graph embedding'
]

question_types = [
    'true/false',
    'calculations',
    'draw',
    'multi'
]

model = None
tokenizer = None
peft_model = None

class StopOnNewline(StoppingCriteria):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_token_id = input_ids[0, -1].item()
        last_token = self.tokenizer.decode(last_token_id)
        return last_token == '\n'

def contains_keywords(text, keywords):
    start_index = text.find('assistant\n\n')
    if start_index != -1:
        text = text[start_index:]
    for keyword in keywords:
        if keyword in text:
            return True
    return False

def extract_name_before_keyword(text, keywords):
    start_index = text.find('assistant\n\n')
    if start_index != -1:
        text = text[start_index:]
    names = []
    lines = text.split('\n')
    for line in lines:
        for keyword in keywords:
            pattern = r"\s*(\S+)\s+" + re.escape(keyword)
            matches = re.findall(pattern, line)
            if matches:
                names.extend([re.sub(r'^\W+|\W+$', '', match) for match in matches])
                return list(set(names)) if names else None
    return None

def generate_response(input_text, cuda_number=0):
    keywords = ['algorithm', 'function', 'method']
    message = [{"role": "user", "content": f"{input_text}"}]
    
    input_ids = tokenizer.apply_chat_template(conversation=message,
                                              tokenize=True,
                                              add_generation_prompt=True,
                                              return_tensors='pt')
    input_ids = input_ids.to(f"cuda:6")
    
    stop_on_newline = StopOnNewline(tokenizer)
    stopping_criteria = StoppingCriteriaList([stop_on_newline])
    
    response = ""
    while True:
        with torch.inference_mode():
            output_ids = model.generate(input_ids=input_ids, max_new_tokens=4096, 
                                        do_sample=False, pad_token_id=2, 
                                        stopping_criteria=stopping_criteria)
            
        new_tokens = output_ids[:, input_ids.shape[1]:]
        new_response = tokenizer.batch_decode(new_tokens.detach().cpu().numpy(), skip_special_tokens=True)[0]
        
        response = new_response
        
        if contains_keywords(response, keywords):
            names = extract_name_before_keyword(response, keywords)
            if names:
                return names
            else:
                return f"Found keyword but could not extract name. Response: {response}"
        else:
            input_ids = torch.cat((input_ids, new_tokens), dim=1)

def find_api_info(api_name):
    df = pd.read_csv("./api_info/all_api_template_v4.csv")
    values = df[df['api'] == api_name]['template'].values
    if values.size > 0:
        return values[0]
    else:
        return ""

def random_sentence():
    df = pd.read_csv("./api_info/sentence.csv")
    return df.sample()['text'].values[0]


def generate_response_new(input_text, assistant_info):
    message = [
        {"role": "user", "content": f"{input_text}"},
        {"role": "assistant", "content": f"{assistant_info}"},
    ]
    
    input_ids = tokenizer.apply_chat_template(conversation=message,
                                            tokenize=True,
                                            add_generation_prompt=False,
                                            return_tensors='pt')
    
    input_ids = input_ids.to("cuda:6")
    with torch.inference_mode():
        output_ids = model.generate(input_ids=input_ids[:, :-3], max_new_tokens=4096, do_sample=False, pad_token_id=2)
    response = tokenizer.batch_decode(output_ids.detach().cpu().numpy(), skip_special_tokens = True)
    
    return response

def extract_response_deepseek(response):
    answer = response[0]
    match = re.search(r"Response:(.*)", answer, re.DOTALL)
    true_response = match.group(1)
    return true_response

def extract_response_llama3(response):
    answer = response[0]
    match = re.search(r"assistant\n\n(.*)", answer, re.DOTALL)
    true_response = match.group(1)
    return true_response

def rag_questions(df_input, top_k = 0):
    
    doc_list = []
    rag_list = []
    questions = []
    
    prompt_text = 'You need to use python to solve the problem.'
    
    lock = threading.Lock()

    total_rows = len(df_input)

    def process_rows(df_subset, doc_list, rag_list, progress_bar, lock):
        for _, row in df_subset.iterrows():
            question_text = row['new_question']
            rag_result = rag_query(question_text, top_k) if top_k > 0 else None
            with lock:
                doc_list.append(question_text)
                if rag_result is not None:
                    rag_list.append(rag_result)
                progress_bar.update(1)

    progress_bar = tqdm(total=total_rows, desc='RAG query', unit='row')

    num_threads = 32
    
    chunk_size = math.ceil(total_rows / num_threads)

    threads = []
    for i in range(num_threads):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, total_rows)
        df_subset = df_input.iloc[start_idx:end_idx]
        thread = threading.Thread(target=process_rows, args=(df_subset, doc_list, rag_list, progress_bar, lock))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    progress_bar.close()
    
    q_num = len(doc_list)
    
    if top_k == 0:
        for i in range(q_num):
            question = str(doc_list[i]) + '\n' + prompt_text
            questions.append(question)
    else:
        for i in range(q_num):
            question = str(rag_list[i]) + '\n' + str(doc_list[i]) + '\n' + prompt_text
            questions.append(question)
    
    return questions


def process_data(data, model_name, top_k):
    if top_k == 0:
        first_response = generate_response(str(data))
        
        total_api_info = ""
        for fuc in first_response:
            api_info = find_api_info(fuc)
            total_api_info += api_info + "\n\n"
        last_sentence = 'By using the info above, we can write a python code to solve this problem.\nHere is the python code.'
        total_api_info += last_sentence
        total_api_info = total_api_info.strip()
        
        second_response = generate_response_new(str(data), total_api_info)
        
        if model_name == 'deepseek':
            true_response = extract_response_deepseek(second_response)
        else:
            true_response = extract_response_llama3(second_response)
        return true_response
    else:
        response = generate_response_new(str(data), '')
        
        if model_name == 'deepseek':
            true_response = extract_response_deepseek(response)
        else:
            true_response = extract_response_llama3(response)
        return true_response
        
def process_file(df_input, output_dir, model_name, top_k=0):
    
    new_csv_path = os.path.join(output_dir, 'inference_by_' + str(model_name) + '.csv')

    df = df_input.copy()
    
    questions = rag_questions(df, top_k)
    
    results = []
        
    for question in tqdm(questions, desc="Inference", unit="question"):
        processed = process_data(question, model_name, top_k)
        results.append(processed)
        
    df[model_name] = results
    
    df.to_csv(new_csv_path, index=False)

def main():
    
    print('Inference started.')
    
    global tokenizer, model, peft_model
    
    file_path = '../benchmark/merged/prograph.csv'
    
    df_input = pd.read_csv(file_path)
    
    parser = argparse.ArgumentParser(description="Select the parameters.")

    parser.add_argument(
        '--model_name',
        type=str,
        choices=model_name_or_path_all.keys(),
        required=True,
        help="Specify the model name. Available options: {}".format(', '.join(model_name_or_path_all.keys()))
    )
    
    parser.add_argument(
        '--model_setting',
        type=str,
        choices=model_settings,
        required=True,
        help="Specify the model setting. Available options: {}".format(', '.join(model_settings))
    )
    
    parser.add_argument(
        '--top_k',
        type=int,
        choices=range(0, 10),
        required=False,
        default=0,
        help="Specify the top_k value. Must be an integer between 1 and 10. 0 means no rag."
    )
    
    parser.add_argument(
        '--lib_name',
        type=str,
        choices=lib_names,
        required=False,
        help="Specify the python library name. Available options: {}".format(', '.join(lib_names))
    )
    
    parser.add_argument(
        '--answer_difficulty',
        type=str,
        choices=answer_difficulties,
        required=False,
        help="Specify the difficulty of questions. Available options: {}".format(', '.join(answer_difficulties))
    )
    
    parser.add_argument(
        '--category',
        type=str,
        choices=catogories,
        required=False,
        help="Specify the category of questions. Available options: {}".format(', '.join(answer_difficulties))
    )
    
    parser.add_argument(
        '--question_type',
        type=str,
        choices=question_types,
        required=False,
        help="Specify the difficulty of questions. Available options: {}".format(', '.join(question_types))
    )
    
    args = parser.parse_args()

    model_name = args.model_name
    model_setting = args.model_setting
    top_k = args.top_k
    lib_name = args.lib_name
    answer_difficulty = args.answer_difficulty
    category = args.category
    question_type = args.question_type
    
    if lib_name is not None:
        df_input = df_input[df_input['library'] == lib_name]
    
    if answer_difficulty is not None:
        if answer_difficulty == 'easy':
            df_input = df_input[df_input['api_num'] == 'single']
        else:
            df_input = df_input[df_input['api_num'] == 'multi']

    if category is not None:
        df_input = df_input[df_input['category'] == category]
    
    if question_type is not None:
        if question_type == 'true/false':
            df_input = df_input[df_input['note'] == 'True/False']
        elif question_type == 'calculations':
            df_input = df_input[df_input['note'] == 'calculations']
        elif question_type == 'draw':
            df_input = df_input[df_input['note'] == 'draw']
        else:
            df_input = df_input[~df_input['note'].isin(['True/False', 'calculations', 'draw'])]        
        
    if model_setting == 'doc and code' and top_k != 0:
        raise ValueError('top_k must be 0 when model_setting is doc and code.')
    
    model_name_or_path = model_name_or_path_all[model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)
    
    if model_setting != 'no fine-tuning':
        if model_name == 'llama3' and model_setting in model_settings_llama3.keys():
            peft_model_path = model_settings_llama3[model_setting]
        elif model_name == 'deepseek' and model_setting in model_settings_deepseek.keys():
            peft_model_path = model_settings_deepseek[model_setting]
            
        peft_model = PeftModel.from_pretrained(model, peft_model_path).to(device)
        
    output_dir = '../benchmark/evaluation/' + model_name + '/' + model_setting + '_top_k_' + str(top_k)
    os.makedirs(output_dir, exist_ok=True)

    process_file(df_input, output_dir, model_name, top_k)

    print('Inference finished.')

if __name__ == "__main__":
    main()
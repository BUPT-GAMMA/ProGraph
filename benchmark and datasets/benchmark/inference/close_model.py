import os
import os.path
import google.generativeai as genai
from pathlib import Path
import glob
import csv
import math
import time
import openai
import argparse
import pandas as pd
from tqdm import tqdm
import threading
from tenacity import retry, wait_exponential, stop_after_attempt

openai.api_key = os.environ.get('OPENAI_API_KEY')
openai.base_url = os.environ.get('OPENAI_BASE_URL')

os.environ['INDEX_BUILD_TYPE'] = 'json'
os.environ['RAG_DATA'] = 'data'
os.environ['RAG_LLM_TEMPERATURE'] = '0.3'

model_names = [
    'gpt-3.5-turbo',
    'gpt-3.5-turbo-0125',
    'gpt-3.5-turbo-1106',
    'gpt-4-turbo',
    'gpt-4-turbo-2024-04-09',
    'gpt-4-1106-preview',
    'gpt-4o-mini',
    'gpt-4o-mini-2024-07-18',
    'gpt-4o',
    'gpt-4o-2024-08-06',
    'gpt-4o-2024-05-13',
    'chatgpt-4o-latest',
    'claude-3-opus-20240229',
    'claude-3-sonnet-20240229',
    'claude-3-haiku-20240307',
    'gemini-1.0-pro',
    'gemini-1.5-pro'
]

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

os.environ["GEMINI_API_KEY"] = ""

def top_k():
    PERSIST_DIR = "./storageGPT"

    if PERSIST_DIR is not None:
            if not os.path.exists(PERSIST_DIR):
                from llama_index.core.node_parser import SentenceSplitter

                # local settings
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
                        ),
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

    return index

@retry(wait=wait_exponential(multiplier=1, max=120), stop=stop_after_attempt(50))
def rag_query(question, similarity_top_k, index=top_k()):
    retriever = index.as_retriever(
        retriever_mode="llm",
        similarity_top_k=similarity_top_k
    )
    nodes = retriever.retrieve(question)
    ret = [item.get_text() for item in nodes]
    ret = "\n".join(ret)
    
    return (
            "according to the query and documents, the answers retrieved by rag are \n\n"
            + ret
            + "."
        )

@retry(wait=wait_exponential(multiplier=1, max=180), stop=stop_after_attempt(30))
def gemini_generate_response(question, model_name='gemini-1.0-pro'):
    
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    if os.environ["GEMINI_API_KEY"] == "":
        raise Exception("Please set the GEMINI_API_KEY environment variable.")
    
    # random_number = random.randrange(5, 11)
    # time.sleep(random_number)
    generation_config = {
        "temperature": 0.0,
        "top_p": 0.95,
        "top_k": 0,
        "max_output_tokens": 2048,
    }

    model = genai.GenerativeModel(model_name=model_name,
                                   generation_config=generation_config,
                                   )

    convo = model.start_chat(history=[
    ])

    convo.send_message(question)
    return convo.last.text

@retry(wait=wait_exponential(multiplier=1, max=180), stop=stop_after_attempt(30))
def generate_response(question, model_name = 'gpt-3.5-turbo'):

    completion = openai.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
        ],
        # min_tokens=80,
        # max_tokens=2000,
        # response_format={"type": "json_object"}
        temperature=0.0,
    )
    return completion.choices[0].message.content.strip()

rag_list = []
doc_list = []
questions = []

prompt_text = """
You need to use python to solve the problem.
"""

def ensure_directory_exists(file_path):

    path = Path(file_path)
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Directory created: {path.parent}")
    else:
        print(f"Dir already exists: {path.parent}")

def rag_questions(df_input, top_k = 0):
    
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

    progress_bar = tqdm(total=total_rows, desc='Processing rows', unit='row')

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

def process_data(data, model_name):
    
    if 'gemini' in model_name:
        return gemini_generate_response(str(data), model_name)
    else:
        return generate_response(str(data), model_name)

def process_data_threaded(input_chunk, results, idx, selected_model_name):

    processed_chunk = []
    
    for data in input_chunk:
        
        processed_result = process_data(data, selected_model_name)
        processed_chunk.append(processed_result)

    results[idx] = processed_chunk

def answer_questions(input_list, num_threads, selected_model_name):
    
    results = [[] for _ in range(num_threads)]

    threads = []
    total_items = len(input_list)
    chunk_size = (total_items + num_threads - 1) // num_threads

    progress_bar = tqdm(total=total_items, desc="Processing", unit='row')

    def process_data_threaded_with_progress(input_chunk, results, idx, selected_model_name):
        thread_results = process_data_threaded(input_chunk, results, idx, selected_model_name)
        for _ in input_chunk:
            progress_bar.update(1)
        return thread_results

    for i in range(num_threads):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_items)
        thread = threading.Thread(target=process_data_threaded_with_progress,
                                  args=(input_list[start_idx:end_idx], results, i, selected_model_name))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    progress_bar.close()

    merged_data = [item for sublist in results for item in sublist]
    
    return merged_data

def main():
        
    file_path = '../prograph.csv'
    
    df_input = pd.read_csv(file_path)
    
    parser = argparse.ArgumentParser(description="Select the parameters.")
    parser.add_argument(
        '--model_name',
        type=str,
        choices=model_names,
        required=True,
        help="Specify the model name. Available options: {}".format(', '.join(model_names))
    )
    
    parser.add_argument(
        '--top_k',
        type=int,
        choices=range(0, 10),
        required=False,
        help="Specify the top_k value. Must be an integer between 1 and 10. 0 means no rag."
    )
    
    parser.add_argument(
        '--num_threads',
        type=int,
        choices=range(1, 33),
        help="Specify the num_threads value."
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
    selected_model_name = args.model_name
    selected_top_k = args.top_k if args.top_k is not None else 0
    selected_lib_name = args.lib_name
    num_threads = args.num_threads if args.num_threads is not None else 1
    lib_name = args.lib_name
    answer_difficulty = args.answer_difficulty
    category = args.category
    question_type = args.question_type
    
    if selected_top_k != 0:
        top_k()
    
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
        
    processed_questions = rag_questions(df_input, selected_top_k)
        
    merged_data = answer_questions(processed_questions, num_threads, selected_model_name)

    df_new = df_input
    
    df_new[selected_model_name] = merged_data
    result_file_name = '../results/' + selected_model_name + '/' + 'rag_' + str(selected_top_k) + '_' + selected_lib_name
    
    ensure_directory_exists(result_file_name)
    
    total_result_file_name = result_file_name + '.csv'
    
    df_new.to_csv(total_result_file_name, index=False)
    
    print(f"File has been updated and saved to: {result_file_name}\n")


if __name__ == "__main__":
    
    import pdb
    pdb.set_trace()
    
    main()
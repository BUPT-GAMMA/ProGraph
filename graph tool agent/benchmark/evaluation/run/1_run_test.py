import pandas as pd
import re
import io
import sys
import traceback
from tqdm import tqdm
import os
import shutil
from multiprocessing import Pool, TimeoutError, cpu_count
import signal
import argparse


DEFAULT_TIMEOUT = 120  # 设置超时时间为120秒
TIMEOUT_MSG = 'Error: timeout'
TMP_DIR = 'tmp_execution_output_lixin'  # 临时文件夹名
DATA_FOLDER = 'data'  # 目标文件夹

# 0525 new
def extract_code(text):
    text = str(text)
    matches = re.findall(r"```[ \t]*[Pp]ython[ \t]*(.*?)```", text, re.DOTALL)
    filtered_matches = [match for match in matches if not re.search(r"\bpip install\b", match)]
    code = '\n'.join(filtered_matches)
    return code if code else text

def replace_path_with_filename(code):
    # 捕捉文件路径并替换前缀为 'data/'
    pattern = r"'[^']*\/([^\/']+\.[^\/']+)'|'\b([^']+\.[^\/']+)\b'"
    modified_code = re.sub(pattern, lambda match: f"'{DATA_FOLDER}/{match.group(1) or match.group(2)}'", code)
    return modified_code

def execute_code(code):
    try:
        exec_globals = {}
        buffer = io.StringIO()
        sys.stdout = buffer

        exec(code, exec_globals)

        sys.stdout = sys.__stdout__
        return buffer.getvalue(), "success"
    except Exception as e:
        sys.stdout = sys.__stdout__
        tb_str = traceback.format_exc()
        last_line = tb_str.strip().splitlines()[-1]
        error_type = last_line.split(":")[0]
        return str(error_type), "error"

def process_row(row_data, tmp_dir):
    index, row = row_data
    new_code = replace_path_with_filename(row['code_answer'])

    try:
        with Pool(1) as pool:
            result = pool.apply_async(execute_code, (new_code,))
            output, status = result.get(timeout=DEFAULT_TIMEOUT)
        return output, status
    except TimeoutError:
        return TIMEOUT_MSG, "timeout"
    except Exception as e:
        tb_str = traceback.format_exc()
        return str(tb_str), "error"

def process_file(file_path, save_file_path):
    global total_successes, total_timeouts, total_errors

    if not os.path.exists(TMP_DIR):
        os.makedirs(TMP_DIR)

    network_analysis_df = pd.read_csv(file_path, encoding='utf-8')
    last_column_name = network_analysis_df.columns[-1]
    network_analysis_df['code_answer'] = network_analysis_df[last_column_name].apply(extract_code)

    results = []
    for row in tqdm(network_analysis_df.iterrows(), total=network_analysis_df.shape[0], desc="Processing rows"):
        result = process_row(row, TMP_DIR)
        results.append(result)

    # 清理临时文件夹
    # shutil.rmtree(TMP_DIR)

    successes = [res for res, status in results if status == "success"]
    timeouts = [res for res, status in results if status == "timeout"]
    errors = [res for res, status in results if status == "error"]

    network_analysis_df['exec_result'] = [res for res, status in results]
    # network_analysis_df['exec_output'] = [status for res, status in results]

    dir_name, file_name = os.path.split(file_path)
    new_file_name = "exec_" + file_name
    output_file_path = os.path.join(dir_name, new_file_name)
    network_analysis_df.to_csv(output_file_path, encoding='utf-8', index=False)

    print(f"Processed {output_file_path} with {len(successes)} successes, {len(timeouts)} timeouts, and {len(errors)} other errors.")

    with open(save_file_path, 'a') as file:
        file.write(f"{output_file_path}: {len(successes)} successes, {len(timeouts)} timeouts, and {len(errors)} other errors.\n")

    total_processed = len(successes) + len(timeouts) + len(errors)
    success_ratio = len(successes) / total_processed if total_processed > 0 else 0

    with open(save_file_path, 'a') as file:
        file.write(f"Total: {total_processed} processed, {len(successes)} successes, {len(timeouts)} timeouts, {len(errors)} errors.\n")
        file.write(f"Success ratio: {success_ratio:.2%}\n")

    print(f"Total processed: {total_processed}")
    print(f"Total successes: {len(successes)}")
    print(f"Total timeouts: {len(timeouts)}")
    print(f"Total errors: {len(errors)}")
    print(f"Success ratio: {success_ratio:.2%}")
    
    total_successes += len(successes)
    total_timeouts += len(timeouts)
    total_errors += len(errors)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default='../run_result/')
    parser.add_argument("--dir", type=str, default='/Users/lixin/downthesky/coding/Python/ProGraph/graph tool agent/benchmark/benchmark/evaluation/gpt-4o-mini')
    args = parser.parse_args()
    
    directory = os.path.join('/home/lixin/graph tool agent/benchmark/benchmark_others/evaluation/pass_1/', args.dir)
    # save_file_path = '../run_result/gpt4o_rag_5.txt'
    save_file_path = os.path.join(args.save_path, args.dir.split('/')[-1] + '.txt')
    print(f'save as {save_file_path}')

    total_successes = 0
    total_timeouts = 0
    total_errors = 0

    if not os.path.exists(save_file_path):
        with open(save_file_path, 'w') as file:
            pass
        print(f"File {save_file_path} was created.")
    else:
        print(f"File {save_file_path} already exists.")

    # directory = '/home/lixin/graph tool agent/benchmark/evaluation/pass_1/gpt4o_rag_5'
    
    for filename in os.listdir(directory):
        if 'exec' not in filename and 'check' not in filename and filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            print(f"Processing file: {file_path}")
            process_file(file_path, save_file_path)
            
    # 计算总的success的比例，并写入txt文件中
    total_processed = total_successes + total_timeouts + total_errors
    success_ratio = total_successes / total_processed if total_processed > 0 else 0

    with open(save_file_path, 'a') as file:
        file.write(f"Total: {total_processed} processed, {total_successes} successes, {total_timeouts} timeouts, {total_errors} errors.\n")
        file.write(f"Success ratio: {success_ratio:.2%}\n")

    print(f"Total processed: {total_processed}")
    print(f"Total successes: {total_successes}")
    print(f"Total timeouts: {total_timeouts}")
    print(f"Total errors: {total_errors}")
    print(f"Success ratio: {success_ratio:.2%}")
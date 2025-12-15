import sys, json, math, random
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import os, re, io
import subprocess
import pickle
import traceback
import contextlib
import importlib.util
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models.tongyi import ChatTongyi
from config import MODEL_NAME, API_KEY, BASE_URL

def read_excel(file_path: str) -> pd.DataFrame:
    return pd.read_excel(file_path, sheet_name = 'Sheet1')

def read_json(file_path: str) -> Dict:
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def save_json(data, path: str) -> Dict:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def read_txt(file_path: str) -> str:
    if not os.path.exists(file_path):
        print(f"not find {file_path}")
        return ""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

def save_txt(data: str, path: str) -> None:
    with open(path, 'w', encoding='utf-8') as file:
        file.write(data)

def save_pkl(data: List, path: str) -> None:
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def sampling(data: List, pred: List, num: int = 100) -> List:
    """
    data: historical data
    pred: Predicted values and interval bounds
    num: Number of samples
    """
    samples = []
    for value, low, high in pred:
        tmp = [x for x in data if x >= low and x <= high]
        if len(tmp) == 0:
            line = [random.choice(data) for _ in range(num)]
        else:
            line = [random.choice(tmp) for _ in range(num)]
        samples.append(line)
    return samples

def extract_python_code(text):
    pattern = r"```python\n(.*?)\n```"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return ""
    return match.group(1).strip()

def extract_json(text):
    pattern = r"```json\n(.*?)\n```"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return {}
    return match.group(1).strip()

def safe_exec(code_string: str):
    stdout = io.StringIO()
    original_stdout = sys.stdout
    try:
        sys.stdout = stdout
        with contextlib.redirect_stdout(stdout):
            exec_globals = {}
            exec(code_string, exec_globals)
    except ModuleNotFoundError as e:
        missing_module = str(e).split("'")[1]
        sys.stdout = original_stdout
        stdout.close()
        subprocess.run([sys.executable, "-m", "pip", "install", missing_module], check=True)
        return safe_exec(code_string)
    except Exception:
        return traceback.format_exc()
    finally:
        sys.stdout = original_stdout
    return stdout.getvalue()

def safe_exec_with_flag(code_string: str):
    stdout = io.StringIO()
    original_stdout = sys.stdout
    try:
        sys.stdout = stdout
        with contextlib.redirect_stdout(stdout):
            exec_globals = {
                '__name__': '__main__',
                '__sys_executable': sys.executable
            }
            exec(code_string, exec_globals)
    except ModuleNotFoundError as e:
        missing_module = str(e).split("'")[1]
        sys.stdout = original_stdout
        stdout.close()
        subprocess.run([sys.executable, "-m", "pip", "install", missing_module], check=True)
        return safe_exec_with_flag(code_string)
    except Exception:
        return 0, traceback.format_exc()
    finally:
        sys.stdout = original_stdout
    return 1, stdout.getvalue()

def extract_pred(text):
    pattern = r"Predicted value: (-?[\d.]+), Lower bound: (-?[\d.]+), Upper bound: (-?[\d.]+)"
    matches = re.findall(pattern, text)
    return [(float(low), float(high)) for value, low, high in matches]

def calculate_bounds(data_matrix):
    data_matrix = np.array(data_matrix)
    m, n = data_matrix.shape
    
    lower_bounds = np.zeros(m)
    upper_bounds = np.zeros(m)
    
    for i in range(m):
        row_data = data_matrix[i, :]
        
        if n > 30:
            sorted_data = np.sort(row_data)
            lower_bound = np.percentile(sorted_data, 5)
            upper_bound = np.percentile(sorted_data, 95)
        else:
            lower_bound = np.min(row_data)
            upper_bound = np.max(row_data)
        
        lower_bounds[i] = lower_bound
        upper_bounds[i] = upper_bound
    
    return lower_bounds.reshape(-1, 1), upper_bounds.reshape(-1, 1)

def calculate_mean(data_matrix):
    data_matrix = np.array(data_matrix)
    m, n = data_matrix.shape

    d_bar = []
    for i in range(m):
        row_data = data_matrix[i, :]
        bar = np.mean(row_data)
        d_bar.append(bar)
    
    return d_bar

class MYLLM:
    def __init__(self, model = MODEL_NAME, temperature = 0):
        TEMP_DEEPSEEK_API_KEY = API_KEY
        llm = ChatOpenAI(model_name=model, temperature=temperature,api_key=TEMP_DEEPSEEK_API_KEY,base_url=BASE_URL)
        self.llm = llm
    # def __init__(self, model="qwen-max", temperature=0):
    #     llm = ChatTongyi(
    #         model=model, 
    #         temperature=temperature,
    #         dashscope_api_key=API_KEY
    #     )
    #     self.llm = llm


if __name__ == "__main__":
    code = """for i in range(5):
    printf("Hello, World!", i)
"""    
    flag, output = safe_exec_with_flag(code)
    print(output)
    print("Flag:", flag)




import random
import os
import pandas as pd
from typing import Dict
from baselines.CoT import Optimization_Modeling
from utils import save_txt, extract_python_code, safe_exec, safe_exec_with_flag, calculate_mean

def pipline(result_dir: str, problem: Dict, data_discription: Dict, train_dir: str, test_dir: str, model: str, temperature: float, method: str) -> None:
    train_df = pd.read_excel(train_dir)
    test_df = pd.read_excel(test_dir)
    history_data = train_df
    real_test_data = test_df.iloc[:,-1]
    #------------------------------------------------------------------------------------------------------------------------
    code_path = f'{result_dir}/code.txt'
    if os.path.exists(code_path):
        with open(code_path, 'r', encoding='utf-8') as f:
            code_string = f.read()
    else:
        code_generating = Optimization_Modeling()
        code_response = code_generating.run(problem, data_discription, test_dir, result_dir, train_dir, method)
        code_string = extract_python_code(code_response)
        save_txt(code_string, f'{result_dir}/code.txt')
    result_path = f'{result_dir}/result.txt'
    if os.path.exists(result_path):
        pass
    else:
        flag, result = safe_exec_with_flag(code_string)
        save_txt(result, f'{result_dir}/result.txt')
        print("solving finish!")

if __name__ == '__main__':
    pass
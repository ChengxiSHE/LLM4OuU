import random
import pickle
import os
import pandas as pd
import numpy as np
from typing import Dict
from agents.SAA_experts import SAA_Auto_Modeling, Code_Generating
from agents.code_repair import Code_Repair
from utils import save_txt, save_pkl, extract_python_code, safe_exec, safe_exec_with_flag

def pipline(result_dir: str, problem: Dict, data_discription: Dict, train_dir: str, test_dir: str, model: str, temperature: float) -> None:
    train_df = pd.read_excel(train_dir)
    test_df = pd.read_excel(test_dir)
    data_matrix = train_df.values
    row, col = data_matrix.shape  
    #------------------------------------------------------------------------------------------------------------------------
    math_model_path = f'{result_dir}/math_model.txt'
    if os.path.exists(math_model_path):
        with open(math_model_path, 'r', encoding='utf-8') as f:
            math_model = f.read()
    else:
        saa_auto_modeling = SAA_Auto_Modeling()
        math_model = saa_auto_modeling.run(problem, row, col)
        save_txt(math_model, f'{result_dir}/math_model.txt')
        print("modeling finish!")
    #------------------------------------------------------------------------------------------------------------------------
    code_path = f'{result_dir}/code.txt'
    if os.path.exists(code_path):
        with open(code_path, 'r', encoding='utf-8') as f:
            code_string = f.read()
    else:
        code_generating = Code_Generating()
        code_response = code_generating.run(problem, data_discription, row, col, math_model, train_dir, result_dir, test_dir)
        code_string = extract_python_code(code_response)
        save_txt(code_string, f'{result_dir}/code.txt')
        print("coding finish!")
    result_path = f'{result_dir}/result.txt'
    if os.path.exists(result_path):
        pass
    else:
        flag, result = safe_exec_with_flag(code_string)
        if flag == 0:
            code_repair = Code_Repair()
            code_response = code_repair.run(code=code_string, error=result)
            code_string = extract_python_code(code_response)
            flag, result = safe_exec_with_flag(code_string)
        save_txt(result, f'{result_dir}/result.txt')

if __name__ == '__main__':
    pass

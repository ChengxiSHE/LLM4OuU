import random
import os
import pandas as pd
from typing import Dict
from agents.RO_Ellipsoidal_experts import Optimization_Modeling, Code_Generating, Transfer_Modeling
from agents.code_repair import Code_Repair
from utils import save_txt, extract_python_code, safe_exec, safe_exec_with_flag, calculate_mean

def pipline(result_dir: str, omega, problem: Dict, data_discription: Dict, train_dir: str, test_dir: str, model: str, temperature: float) -> None:
    train_df = pd.read_excel(train_dir)
    test_df = pd.read_excel(test_dir)
    history_data = train_df
    real_test_data = test_df.iloc[:,-1]
    row = len(real_test_data)
    #------------------------------------------------------------------------------------------------------------------------
    initial_model_path = f'{result_dir}/initial_model.txt'
    if os.path.exists(initial_model_path):
        with open(initial_model_path, 'r', encoding='utf-8') as f:
            initial_model = f.read()
    else:
        initial_modeling = Optimization_Modeling()
        initial_model = initial_modeling.run(problem, row)
        save_txt(initial_model, f'{result_dir}/initial_model.txt')
        print("initial modeling finish!")
    #------------------------------------------------------------------------------------------------------------------------
    math_model_path = f'{result_dir}/math_model.txt'
    if os.path.exists(math_model_path):
        with open(math_model_path, 'r', encoding='utf-8') as f:
            math_model = f.read()
    else:
        transfer_modeling = Transfer_Modeling()
        math_model = transfer_modeling.run(problem, row, omega, initial_model)
        save_txt(math_model, f'{result_dir}/math_model.txt')
        print("modeling finish!")
    #------------------------------------------------------------------------------------------------------------------------
    code_path = f'{result_dir}/code.txt'
    d_bar = calculate_mean(history_data)
    if os.path.exists(code_path):
        with open(code_path, 'r', encoding='utf-8') as f:
            code_string = f.read()
    else:
        code_generating = Code_Generating()
        code_response = code_generating.run(problem, data_discription, row, omega, initial_model, math_model, test_dir, result_dir, train_dir, d_bar)
        code_string = extract_python_code(code_response)
        save_txt(code_string, f'{result_dir}/code.txt')
    result_path = f'{result_dir}/result.txt'
    if os.path.exists(result_path):
        pass
    else:
        flag, result = safe_exec_with_flag(code_string)
        if flag == 0:
            code_repair = Code_Repair()
            code_string = code_repair.run(code=code_string, error=result)
            result = safe_exec(code_string)
        save_txt(result, f'{result_dir}/result.txt')
        print("solving finish!")

if __name__ == '__main__':
    pass

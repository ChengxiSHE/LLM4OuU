import os
import argparse
from utils import read_txt
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def build_tasks(args):
    problems = []

    for dir in os.listdir(args.dataset_path):
        data_path = os.path.join(args.dataset_path, dir)
        if not os.path.isdir(data_path):
            continue

        problem = read_txt(os.path.join(data_path, "problem_description.txt"))
        data_description = read_txt(os.path.join(data_path, "train_description.txt"))
        train_dir = os.path.join(data_path, "train.xlsx")
        test_dir = os.path.join(data_path, "test.xlsx")

        problems.append({
            "data_dir": data_path,
            "omega": args.omega,
            "gamma": args.gamma,
            "theta": args.theta,
            "problem": problem,
            "data_description": data_description,
            "train_dir": train_dir,
            "test_dir": test_dir,
            "model": args.model,
            "temperature": args.temperature,
        })

    return problems


def get_solver(method):

    if method == 'saa':
        import execute.saa as saa

        def solve_single(item: dict):
            result_path = os.path.join(item['data_dir'], "Deepseek-chat", "saa")  # Deepseek-R1, Qwen3-max, Deepseek-chat
            os.makedirs(result_path, exist_ok=True)
            saa.pipline(result_path, item['problem'], item['data_description'],
                        item['train_dir'], item['test_dir'], item['model'], item['temperature'])

        return solve_single

    elif method == 'ro_box':
        import execute.ro_box as ro_box

        def solve_single(item: dict):
            result_path = os.path.join(item['data_dir'], "Deepseek-chat", "ro_box")
            os.makedirs(result_path, exist_ok=True)
            ro_box.pipline(result_path, item['problem'], item['data_description'],
                           item['train_dir'], item['test_dir'], item['model'], item['temperature'])

        return solve_single

    elif method == 'ro_ellipsoidal':
        import execute.ro_ellipsoidal as ro_ellipsoidal

        def solve_single(item: dict):
            result_path = os.path.join(item['data_dir'], "Deepseek-chat", "ro_ellipsoidal")
            os.makedirs(result_path, exist_ok=True)
            ro_ellipsoidal.pipline(result_path, item['omega'], item['problem'], item['data_description'],
                                   item['train_dir'], item['test_dir'], item['model'], item['temperature'])

        return solve_single

    elif method == 'ro_budget':
        import execute.ro_budget as ro_budget

        def solve_single(item: dict):
            result_path = os.path.join(item['data_dir'], "Deepseek-chat", "ro_budget")
            os.makedirs(result_path, exist_ok=True)
            ro_budget.pipline(result_path, item['gamma'], item['problem'], item['data_description'],
                              item['train_dir'], item['test_dir'], item['model'], item['temperature'])

        return solve_single

    elif method == 'dro_moment1':
        import execute.dro_moment1 as dro_moment1

        def solve_single(item: dict):
            result_path = os.path.join(item['data_dir'], "Deepseek-chat", "dro_moment1")
            os.makedirs(result_path, exist_ok=True)
            dro_moment1.pipline(result_path, item['problem'], item['data_description'],
                                item['train_dir'], item['test_dir'], item['model'], item['temperature'])

        return solve_single

    elif method == 'dro_wasserstein':
        import execute.dro_wasserstein as dro_wasserstein

        def solve_single(item: dict):
            result_path = os.path.join(item['data_dir'], "Deepseek-chat", "dro_wasserstein")
            os.makedirs(result_path, exist_ok=True)
            dro_wasserstein.pipline(result_path, item['theta'], item['problem'], item['data_description'],
                                    item['train_dir'], item['test_dir'], item['model'], item['temperature'])

        return solve_single

    elif method == 'dro_kl':
        import execute.dro_kl as dro_kl

        def solve_single(item: dict):
            result_path = os.path.join(item['data_dir'], "Deepseek-chat", "dro_kl")
            os.makedirs(result_path, exist_ok=True)
            dro_kl.pipline(result_path, item['problem'], item['data_description'],
                           item['train_dir'], item['test_dir'], item['model'], item['temperature'])

        return solve_single


def main():
    parser = argparse.ArgumentParser(description='Generate math model and code.')
    parser.add_argument('--dataset_path', type=str, default="data")
    parser.add_argument('--omega', type=str, default=2)
    parser.add_argument('--gamma', type=str, default=2)
    parser.add_argument('--theta', type=str, default=2)
    parser.add_argument('--model', type=str, default='deepseek-chat')  # qwen-max, deepseek-reasoner, deepseek-chat
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max_workers', type=int, default=16)
    args = parser.parse_args()

    methods = [
        'saa',
        'ro_box',
        'ro_ellipsoidal',
        'ro_budget',
        'dro_moment1',
        'dro_wasserstein',
        'dro_kl'
    ]

    problems = build_tasks(args)

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:

        for method in methods:
            solve_single = get_solver(method)
            print(f"\n===== Running method: {method} =====")

            futures = [
                executor.submit(solve_single, item)
                for item in problems
            ]

            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {method}"):
                future.result()

            print(f"===== Finished method: {method} =====\n")


if __name__ == '__main__':
    main()

import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from utils import MYLLM

class Optimization_Modeling(MYLLM):

    SYSTEM = """You are an expert in operation research and optimization.
    You are now facing a optimization problem under uncertainty. 
    Now you need to use distributionally robust optimization method to model uncertain optimization problem. 
    You need to do:
    1. Analyze optimization problem, historical data and identify the uncertain parameter.
    2. Construct DRO model with Latex language for formulation, which need to contain the initial min-max/ max-min model.
    3. When establishing an optimization model, it is necessary to consider the realistic constraints in the problem, such as inventory cannot be negative.
    4. Please output three information: Variables, Constraints, Min-max or max-min bjective.
    """

    INPUT = """Optimization problem: 
    A company needs to transport materials for multiple projects and stockpile materials in the warehouse in advance. 
    The monthly material consumption is uncertain.
    The goal is to minimize inventory costs while meeting supply and demand. 
    Inventory costs include holding costs and shortage costs, both of which have the same weight of 1 in the objective function. 
    The decision variable is the monthly inventory level.
    """

    OUTPUT = """
    ### **1. Variables**

    * $x_t$ is the decision variable: inventory level in month $t$.

    ---  
    
    ### **2. Constraints**

    * Non-negative inventory: $x_t \geq 0$.

    ### **3. Min-max / max-min objective**

    $$
    \min_{{\substack{{x_t \geq 0}} \\ t=1,2,\dots,T}} 
    \max_{{\substack{{d_t \in \mathcal{{U}} \\ t=1,2,\dots,T}} 
    \sum_{{t=1}}^{{T}} |x_t - d_t|
    $$

    where:

    * $|x_t - d_t|$ represents the inventory cost (carrying cost or shortage cost) in month $t$,
    * $\mathcal{{U}}$ denotes the uncertainty set,
    * $T$ denotes the total number of months.
    """

    TASK = """
    Optimization problem: 
    {description}
    """

    def __init__(self):
        super().__init__()
        prompt = ChatPromptTemplate([
            ("system",self.SYSTEM), 
            ("human", self.INPUT),
            ("ai", self.OUTPUT),
            ("human", self.TASK),
        ])
        self.chain = prompt | self.llm | StrOutputParser()

    def run(self, description: str, row: int) -> str:
        return self.chain.invoke({"description" : description, "row": str(row)})


class Transfer_Modeling(MYLLM):

    SYSTEM = """You are an expert in operation research and optimization.
    You are now facing a optimization problem under uncertainty. 
    You have conducted problem anaysis and modeled initial mathematical DRO model.
    Now you need to transfer the initial DRO model into tractable form based on Kullback-Leibler (KL) divergence ambiguity set using the strong duality theory.
    You need to do:
    1. Analyze optimization problem and initial mathematical model.
    2. Transfer the initial DRO model into tractable form based on Kullback-Leibler (KL) divergence ambiguity set using the strong duality theory, and give the final optimized model obtained.
    3. Please output two information: the specific model conversion process, and the final optimized model obtained.
    """

    INPUT = """Optimization problem: 
    A company needs to transport materials for multiple projects and stockpile materials in the warehouse in advance. 
    The monthly material consumption is uncertain. 
    The goal is to minimize inventory costs while meeting supply and demand. 
    Inventory costs include holding costs and shortage costs, both of which have the same weight of 1 in the objective function. 
    The decision variable is the monthly inventory level.

    Initial mathematical DRO model:
    ### **1. Variables**

    * $x_t$ is the decision variable: inventory level in month $t$.

    ---  
    
    ### **2. Constraints**

    * Non-negative inventory: $x_t \geq 0$.

    ### **3. Min-max / max-min objective**

    $$
    \min_{{\substack{{x_t \geq 0}} \\ t=1,2,\dots,T}} 
    \max_{{\substack{{d_t \in \mathcal{{U}} \\ t=1,2,\dots,T}} 
    \sum_{{t=1}}^{{T}} |x_t - d_t|
    $$

    where:

    * $|x_t - d_t|$ represents the inventory cost (carrying cost or shortage cost) in month $t$,
    * $\mathcal{{U}}$ denotes the uncertainty set,
    * $T$ denotes the total number of months.
    """

    OUTPUT = """
    (1) Assumptions and notation

    - Time horizon: \( t = 1, \ldots, T \). Decision variables \( x_t \geq 0 \).

    - Loss per scenario (vector of demands) \( d^s = (d_1^s, \ldots, d_T^s) \) and sample space \( s \in S \). Define the per-scenario loss
    
    \[
    L(x, d^s) = \sum_{{t=1}}^T |x_t - d_t^s|.
    \]
    
    - Nominal distribution \( P_0 \) on scenarios. Two common choices: \textbf{{Empirical discrete}}: \( P_0 \) supported on \( s = 1, \ldots, N \) with probabilities \( p_s \) (often \( p_s = 1/N \)).
    
    - KL ambiguity set of radius \( \rho \):
    \[
    \mathcal{{P}}_{{KL}} = \{{P \ll P_0 : D_{{KL}}(P\|P_0) \leq \rho\}}.
    \]
    
    - DRO problem:
    \[
    \min_{{x \geq 0}} \sup_{{P \in \mathcal{{P}}_{{KL}}}} \mathbb{{E}}_P [L(x, D)].
    \]

    (2) Key duality / reformulation (exponential-tilt representation)

    Use the well-known KL-duality (Donsker-Varadhan / exponential-tilt) identity:

    for any measurable random variable \(Z\),

    \[
    \sup_{{P: D_{{\text{{KL}}}}(P\|P_{{0}})\leq\rho}} \mathbb{{E}}_{{P}}[Z] = \inf_{{\lambda >0}} \left\{{ \lambda\rho + \lambda\log\mathbb{{E}}_{{P_{{0}}}}[e^{{Z/\lambda}}] \right\}}.
    \]

    Apply that with \(Z = L(x,D) = \sum_{{t}}|x_{{t}}-D_{{t}}|\). Therefore the DRO objective becomes

    \[
    \sup_{{P\in\mathcal{{P}}_{{\text{{KL}}}}}} \mathbb{{E}}_{{P}}[L(x,D)] = \inf_{{\lambda>0}} \left\{{ \lambda\rho + \lambda\log\mathbb{{E}}_{{P_{{0}}}} \left[ e^{{\frac{{1}}{{\lambda}} \sum_{{i=1}}^{{T}}|x_{{i}}-D_{{i}}|}} \right] \right\}}.
    \]

    Interchanging min over \(x\) and inf over \(\lambda>0\) (valid since \(\lambda\) appears nicely and the right-hand expression is convex in \(x\)) yields the saddle reformulation

    \[
    \min_{{x\geq 0}} \inf_{{\lambda>0}} \left\{{ \lambda\rho + \lambda\log\mathbb{{E}}_{{P_{{0}}}} \left[ e^{{\frac{{1}}{{\lambda}} \sum_{{i=1}}^{{T}}|x_{{i}}-D_{{i}}|}} \right] \right\}}
    \]

    or equivalently

    \[
    \min_{{x\geq 0,\,\lambda>0}} \left\{{ \lambda\rho + \lambda\log\mathbb{{E}}_{{P_{{0}}}} \left[ e^{{\frac{{L(x,D)}}{{\lambda}}}} \right] \right\}}.
    \]

    This is the standard finite-dimensional reformulation: the infinite-dimensional sup over distributions becomes an optimization over scalar \(\lambda>0\) plus an expectation under the known \(P_{{0}}\).

    (3) Practical discrete (empirical) implementation

    If \( P_0 \) is empirical with scenarios \( s = 1, \ldots, N \) and probabilities \( p_s \) (commonly \( p_s = \frac{{1}}{{N}} \)), then
    
    \[
    \mathbb{{E}}_{{P_0}} \left[ e^{{L(x,D)/\lambda}} \right] = \sum_{{s=1}}^N p_s e^{{L(x,d^s)/\lambda}}.
    \]

    Introduce auxiliary per-scenario variables \( y_s \) to represent the scenario loss:
    \[
    y_s \geq L(x,d^s) = \sum_{{t=1}}^T |x_t - d_t^s|, \quad s = 1, \ldots, N.
    \]

    Then the DRO is equivalently
    \[
    \begin{{aligned}}
    & \min_{{x,\lambda,y}} \lambda \rho + \lambda \log \left( \sum_{{s=1}}^N p_s e^{{y_s/\lambda}} \right) \\
    & \text{{s.t.}} \quad y_s \geq \sum_{{t=1}}^T |x_t - d_t^s|, \quad s = 1, \ldots, N, \\
    & \qquad x_t \geq 0, \quad t = 1, \ldots, T, \\
    & \qquad \lambda > 0.
    \end{{aligned}}
    \]

    (4) Linearizing the absolute values (to feed into a solver that wants linear constraints + convex objective)

    Replace each \(|x_{{t}}-d_{{t}}^{{s}}|\) by an auxiliary \(a_{{t,s}}\geq 0\) with linear constraints:

    \[
    \begin{{aligned}}
    & a_{{t,s}} \geq x_{{t}}-d_{{t}}^{{s}}, \\
    & a_{{t,s}} \geq -(x_{{t}}-d_{{t}}^{{s}}), \\
    & y_{{s}} \geq \sum_{{t=1}}^{{T}}a_{{t,s}}, \\
    & a_{{t,s}} \geq 0.
    \end{{aligned}}
    \]

    So the full implementable form is

    \[
    \min_{{\begin{{subarray}}{{c}}x,\lambda,y,a\end{{subarray}}}} \lambda\rho+\lambda\log\biggl(\sum_{{s=1}}^{{N}}p_{{s}}\,e^{{y_{{s}}/\lambda}}\biggr)
    \]
    s.t.
    \[
    \begin{{aligned}}
    & a_{{t,s}}\geq x_{{t}}-d_{{t}}^{{s}}, && \forall t,s \\
    & a_{{t,s}}\geq -(x_{{t}}-d_{{t}}^{{s}}), && \forall t,s \\
    & y_{{s}}\geq \sum_{{t=1}}^{{T}}a_{{t,s}}, && \forall s \\
    & x_{{t}}\geq 0, ~ a_{{t,s}}\geq 0, ~ \lambda\geq \varepsilon > 0.
    \end{{aligned}}
    \]
    """

    TASK = """
    Optimization problem: 
    {description}

    Initial mathematical DRO model:
    {initial_model}
    """
        
    def __init__(self):
        super().__init__()
        prompt = ChatPromptTemplate([
            ("system",self.SYSTEM), 
            ("human", self.INPUT),
            ("ai", self.OUTPUT),
            ("human", self.TASK),
        ])
        self.chain = prompt | self.llm | StrOutputParser()

    def run(self, description: str, row: int, initial_model: str) -> str:
        return self.chain.invoke({"description" : description, "row": str(row), "initial_model": initial_model})
    

class Code_Generating(MYLLM):

    SYSTEM = """You are a Python programmer working in the fields of operations research and optimization.
    You are proficient in using the third-party solving library like Mosek. (Note that you need to use Mosek to implement the DRO model solving code if the model is convex but not linear)
    You will receive a specific task description, data, and a mathematical model based on distributionally robust optimization method.
    Your goal is to write a usable Python program. 
    You need to do:
    1. Analyze optimization problem, historical data and mathematical model based on DRO model.
    2. Generate the solving code for the DRO model (Golden section search need to be used to find the best \lambda, which is the parameter needed in (KL) divergence ambiguity set DRO)
    3. Print the final solution of optimization problems.
    4. After obtaining the final decision, evaluate it using real data. This requires you to explicitly implement the following processing flow in your code: table data extraction → list conversion → inputting actual values and final optimal solution into the objective function for calculation.
    5. Print the real objective value.
    6. Save the solution, actual value, corresponding decision loss for each decision step, and the real objective value as a text file format in {result_dir}.
    Note: 
    1. Ensure the data loading path in your code is consistent with the provided {train_dir} and {test_dir}. Please strictly import empirical data from {train_dir} and real test data from {test_dir} , and do not fabricate data yourself. Additionally, strictly use “/” as the file path separator.
    2. Please use Golden section search to find the best \lambda, which is the parameter needed in (KL) divergence ambiguity set DRO.
    3. Please note that the table has headers, so do not set header=None when reading it.
    4. Don't output any content contating 'exit()', which will lead to program crash.
    5. Please give the complete code, including the code using ```python```.
    """

    INPUT = """
    The parameter(radius) needed in (KL) divergence ambiguity set is \lambda. Please use Golden section search to find the best \lambda.
    The empirical data is from {train_dir} and real test data is from {test_dir}.

    Empirical data description is as follows:
    {data_description}
    
    Specific task: 
    A company needs to transport materials for multiple projects and stockpile materials in the warehouse in advance. 
    The monthly material consumption is uncertain. 
    The goal is to minimize inventory costs while meeting supply and demand. 
    Inventory costs include holding costs and shortage costs, both of which have the same weight of 1 in the objective function. 
    The decision variable is the monthly inventory level.

    Initial mathematical DRO model:
    ### **1. Variables**

    * $x_t$ is the decision variable: inventory level in month $t$.

    ---  
    
    ### **2. Constraints**

    * Non-negative inventory: $x_t \geq 0$.

    ### **3. Min-max / max-min objective**

    $$
    \min_{{\substack{{x_t \geq 0}} \\ t=1,2,\dots,T}} 
    \max_{{\substack{{d_t \in \mathcal{{U}} \\ t=1,2,\dots,T}} 
    \sum_{{t=1}}^{{T}} |x_t - d_t|
    $$

    where:

    * $|x_t - d_t|$ represents the inventory cost (carrying cost or shortage cost) in month $t$,
    * $\mathcal{{U}}$ denotes the uncertainty set,
    * $T$ denotes the total number of months.

    Each row of the test data represents a decision at a timestep, with {row} rows.

    Transferred DRO model:
    (1) Assumptions and notation

    - Time horizon: \( t = 1, \ldots, T \). Decision variables \( x_t \geq 0 \).

    - Loss per scenario (vector of demands) \( d^s = (d_1^s, \ldots, d_T^s) \) and sample space \( s \in S \). Define the per-scenario loss
    
    \[
    L(x, d^s) = \sum_{{t=1}}^T |x_t - d_t^s|.
    \]
    
    - Nominal distribution \( P_0 \) on scenarios. Two common choices: \textbf{{Empirical discrete}}: \( P_0 \) supported on \( s = 1, \ldots, N \) with probabilities \( p_s \) (often \( p_s = 1/N \)).
    
    - KL ambiguity set of radius \( \rho \):
    \[
    \mathcal{{P}}_{{KL}} = \{{P \ll P_0 : D_{{KL}}(P\|P_0) \leq \rho\}}.
    \]
    
    - DRO problem:
    \[
    \min_{{x \geq 0}} \sup_{{P \in \mathcal{{P}}_{{KL}}}} \mathbb{{E}}_P [L(x, D)].
    \]

    (2) Key duality / reformulation (exponential-tilt representation)

    Use the well-known KL-duality (Donsker-Varadhan / exponential-tilt) identity:

    for any measurable random variable \(Z\),

    \[
    \sup_{{P: D_{{\text{{KL}}}}(P\|P_{{0}})\leq\rho}} \mathbb{{E}}_{{P}}[Z] = \inf_{{\lambda >0}} \left\{{ \lambda\rho + \lambda\log\mathbb{{E}}_{{P_{{0}}}}[e^{{Z/\lambda}}] \right\}}.
    \]

    Apply that with \(Z = L(x,D) = \sum_{{t}}|x_{{t}}-D_{{t}}|\). Therefore the DRO objective becomes

    \[
    \sup_{{P\in\mathcal{{P}}_{{\text{{KL}}}}}} \mathbb{{E}}_{{P}}[L(x,D)] = \inf_{{\lambda>0}} \left\{{ \lambda\rho + \lambda\log\mathbb{{E}}_{{P_{{0}}}} \left[ e^{{\frac{{1}}{{\lambda}} \sum_{{i=1}}^{{T}}|x_{{i}}-D_{{i}}|}} \right] \right\}}.
    \]

    Interchanging min over \(x\) and inf over \(\lambda>0\) (valid since \(\lambda\) appears nicely and the right-hand expression is convex in \(x\)) yields the saddle reformulation

    \[
    \min_{{x\geq 0}} \inf_{{\lambda>0}} \left\{{ \lambda\rho + \lambda\log\mathbb{{E}}_{{P_{{0}}}} \left[ e^{{\frac{{1}}{{\lambda}} \sum_{{i=1}}^{{T}}|x_{{i}}-D_{{i}}|}} \right] \right\}}
    \]

    or equivalently

    \[
    \min_{{x\geq 0,\,\lambda>0}} \left\{{ \lambda\rho + \lambda\log\mathbb{{E}}_{{P_{{0}}}} \left[ e^{{\frac{{L(x,D)}}{{\lambda}}}} \right] \right\}}.
    \]

    This is the standard finite-dimensional reformulation: the infinite-dimensional sup over distributions becomes an optimization over scalar \(\lambda>0\) plus an expectation under the known \(P_{{0}}\).

    (3) Practical discrete (empirical) implementation

    If \( P_0 \) is empirical with scenarios \( s = 1, \ldots, N \) and probabilities \( p_s \) (commonly \( p_s = \frac{{1}}{{N}} \)), then
    
    \[
    \mathbb{{E}}_{{P_0}} \left[ e^{{L(x,D)/\lambda}} \right] = \sum_{{s=1}}^N p_s e^{{L(x,d^s)/\lambda}}.
    \]

    Introduce auxiliary per-scenario variables \( y_s \) to represent the scenario loss:
    \[
    y_s \geq L(x,d^s) = \sum_{{t=1}}^T |x_t - d_t^s|, \quad s = 1, \ldots, N.
    \]

    Then the DRO is equivalently
    \[
    \begin{{aligned}}
    & \min_{{x,\lambda,y}} \lambda \rho + \lambda \log \left( \sum_{{s=1}}^N p_s e^{{y_s/\lambda}} \right) \\
    & \text{{s.t.}} \quad y_s \geq \sum_{{t=1}}^T |x_t - d_t^s|, \quad s = 1, \ldots, N, \\
    & \qquad x_t \geq 0, \quad t = 1, \ldots, T, \\
    & \qquad \lambda > 0.
    \end{{aligned}}
    \]

    (4) Linearizing the absolute values (to feed into a solver that wants linear constraints + convex objective)

    Replace each \(|x_{{t}}-d_{{t}}^{{s}}|\) by an auxiliary \(a_{{t,s}}\geq 0\) with linear constraints:

    \[
    \begin{{aligned}}
    & a_{{t,s}} \geq x_{{t}}-d_{{t}}^{{s}}, \\
    & a_{{t,s}} \geq -(x_{{t}}-d_{{t}}^{{s}}), \\
    & y_{{s}} \geq \sum_{{t=1}}^{{T}}a_{{t,s}}, \\
    & a_{{t,s}} \geq 0.
    \end{{aligned}}
    \]

    So the full implementable form is

    \[
    \min_{{\begin{{subarray}}{{c}}x,\lambda,y,a\end{{subarray}}}} \lambda\rho+\lambda\log\biggl(\sum_{{s=1}}^{{N}}p_{{s}}\,e^{{y_{{s}}/\lambda}}\biggr)
    \]
    s.t.
    \[
    \begin{{aligned}}
    & a_{{t,s}}\geq x_{{t}}-d_{{t}}^{{s}}, && \forall t,s \\
    & a_{{t,s}}\geq -(x_{{t}}-d_{{t}}^{{s}}), && \forall t,s \\
    & y_{{s}}\geq \sum_{{t=1}}^{{T}}a_{{t,s}}, && \forall s \\
    & x_{{t}}\geq 0, ~ a_{{t,s}}\geq 0, ~ \lambda\geq \varepsilon > 0.
    \end{{aligned}}
    \]
    """

    OUTPUT = """```python
    import cvxpy as cp
    import numpy as np
    import pandas as pd
    import math
    import os

    # 1. Load data
    train_dir = "{train_dir}"
    test_dir = "{test_dir}"

    historical_data = pd.read_excel(train_dir)
    real_data = pd.read_excel(test_dir)
    real_demands = real_data.values.flatten().tolist()

    d_samples = historical_data.values
    T, N = d_samples.shape
    rho = 0.1  

    # 2. Define a function: Given λ, solve DRO problem
    def solve_for_lambda(lambda_val):
        x = cp.Variable(T, nonneg=True)
        
        #  L_s = sum_t |x_t - d_t^s|
        scenario_losses = [cp.sum(cp.abs(x - d_samples[:, s])) for s in range(N)]
        L = cp.hstack(scenario_losses)

        # KL-dual objective:
        # λρ + λ * log( (1/N) * sum_s exp(L_s / λ) )
        objective = lambda_val * rho + lambda_val * cp.log_sum_exp(L / lambda_val - np.log(N))
        
        prob = cp.Problem(cp.Minimize(objective))
        prob.solve(solver=cp.MOSEK, verbose=False)

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            return np.inf, None
        return prob.value, x.value

    # 3. Golden section search for best λ
    def golden_section_search(a=1e-3, b=10.0, tol=1e-2, max_iter=20):
        gr = (math.sqrt(5) + 1) / 2
        c = b - (b - a) / gr
        d = a + (b - a) / gr

        val_c, _ = solve_for_lambda(c)
        val_d, _ = solve_for_lambda(d)

        iter_count = 0
        while abs(b - a) > tol and iter_count < max_iter:
            iter_count += 1
            print(f"[Iter {{iter_count}}] λ ∈ [{{a:.4f}}, {{b:.4f}}] | f(c)={{val_c:.4f}}, f(d)={{val_d:.4f}}")

            if val_c < val_d:
                b, d = d, c
                c = b - (b - a) / gr
                val_d = val_c
                val_c, _ = solve_for_lambda(c)
            else:
                a, c = c, d
                d = a + (b - a) / gr
                val_c = val_d
                val_d, _ = solve_for_lambda(d)

        lambda_star = (a + b) / 2
        best_val, x_star = solve_for_lambda(lambda_star)
        return lambda_star, best_val, x_star

    # 4. Perform search
    lambda_star, dro_value, x_star = golden_section_search()
    print("\n========== Optimal Results ==========")
    print(f"Optimal λ*: {{lambda_star:.4f}}")
    print(f"DRO objective value: {{dro_value:.4f}}")
    print(f"Optimal decision x*: {{x_star}}")

    # 5. Evaluate using real demand
    real_losses = np.abs(x_star - real_demands)
    total_loss = float(np.sum(real_losses))
    print("\n========== Evaluation on Real Data ==========")
    for i, v in enumerate(real_losses, start=1):
        print(f"t={{i}}: |x - d| = {{v:.4f}}")
    print(f"Total real objective (sum of losses): {{total_loss:.4f}}")

    # 6. Save results
    result_dir = "results"
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, "dro_result.txt")

    with open(result_path, "w", encoding="utf-8") as f:
        f.write("===== DRO Solution Report =====\n")
        f.write(f"Optimal λ*: {{lambda_star:.6f}}\n")
        f.write(f"DRO Objective Value: {{dro_value:.6f}}\n\n")
        f.write("Per-step decisions (x_t):\n")
        for i, val in enumerate(x_star, 1):
            f.write(f"t={{i}}: x_t={{val:.4f}}\n")
        f.write("\nReal evaluation:\n")
        for i, v in enumerate(real_losses, 1):
            f.write(f"t={{i}}: |x - d| = {{v:.4f}}\n")
        f.write(f"\nTotal Real Objective: {{total_loss:.4f}}\n")

    print(f"\n✅ Results saved to {{result_path}}")
    ```
    """

    TASK = """
    The parameter(radius) needed in wasserstein ambiguity set is \lambda. Please use Golden section search to find the best \lambda, which is the parameter needed in (KL) divergence ambiguity set DRO.
    The empirical data is from {train_dir} and real test data is from {test_dir}.

    Empirical data description is as follows:
    {data_description}

    Specific task: 
    {description}

    Initial mathematical DRO model: 
    {initial_model}

    Transferred DRO model:
    {transfered_model}
    """
        
    def __init__(self):
        super().__init__()
        prompt = ChatPromptTemplate([
            ("system",self.SYSTEM), 
            ("human", self.INPUT),
            ("ai", self.OUTPUT),
            ("human", self.TASK),
        ])
        self.chain = prompt | self.llm | StrOutputParser()

    def run(self, description: str, data_description: str, row: int, initial_model: str, transfered_model: str, train_dir: str, test_dir: str, result_dir: str) -> str:
        return self.chain.invoke({"description" : description, "data_description" : data_description, "row": str(row), "initial_model": initial_model, "transfered_model": transfered_model, "train_dir" : train_dir, "test_dir" : test_dir, "result_dir" : result_dir})
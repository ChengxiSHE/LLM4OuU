import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from utils import MYLLM

class Optimization_Modeling(MYLLM):

    SYSTEM = """You are an expert in operation research and optimization.
    You are now facing a optimization problem under uncertainty. 
    Now you need to use robust optimization method to model uncertain optimization problem. 
    You need to do:
    1. Analyze optimization problem, historical data and identify the uncertain parameter.
    2. Construct RO model with Latex language for formulation, which need to contain the initial min-max/ max-min model.
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
    You have conducted problem anaysis and modeled initial mathematical RO model.
    Now you need to transfer the initial RO model based on budget uncertainty set.
    You have conducted historical data analysis and obtained the mean and variance of the empirical data.
    You need to do:
    1. Analyze optimization problem, initial mathematical model, mean and variance of the empirical data, and parameter omega of ebudget uncertainty set.
    2. Transfer the initial RO model based on budget uncertainty set, and give the final optimized model obtained.
    3. Please output two information: the specific model conversion process, and the final optimized model obtained.
    """

    INPUT = """Optimization problem: 
    A company needs to transport materials for multiple projects and stockpile materials in the warehouse in advance. 
    The monthly material consumption is uncertain, so the company needs to decide the monthly inventory level based on the predicted demand range. 
    The goal is to minimize inventory costs while meeting supply and demand. 
    Inventory costs include holding costs and shortage costs, both of which have the same weight of 1 in the objective function. 
    The decision variable is the monthly inventory level.

    Initial mathematical RO model:
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

    {Gamma} is the budget parameter of budget uncertainty set, controlling the degree of uncertainty.
    """

    OUTPUT = """
    (1) Definition of of Budget Uncertainty Set

    The budget uncertainty set is typically expressed as:

    \[
    \mathcal{{U}} = \left\{{ d_t = \bar{{d}}_t + \hat{{d}}_t \zeta_t \ \middle| \ |\zeta_t| \leq 1, \ \sum_{{t=1}}^T |\zeta_t| \leq \Gamma \right\}}
    \]
    
    where:
    - \(\bar{{d}}_t\) is the nominal demand (the mean value) of empirical data),
    - \(\hat{{d}}_t\) is the maximum demand deviation (magnitude of uncertainty),
    - \(\zeta_t\) is the perturbation variable,
    - \(\Gamma\) is the budget parameter, controlling the degree of uncertainty (\(0 \leq \Gamma \leq T\)).

    (2) Reformulation of the Inner Maximization Problem

    For fixed inventory decisions \(x_t\), the inner problem is to maximize the total cost over the uncertainty set \(\mathcal{{U}}\):

    \[
    \max_{{d_t \in \mathcal{{U}}}} \sum_{{t=1}}^T |x_t - d_t|
    \]

    Substituting \(d_t = \bar{{d}}_t + \hat{{d}}_t \zeta_t\) and letting \(c_t = x_t - \bar{{d}}_t\), the inner problem becomes:

    \[
    \max_{{\zeta_t}} \sum_{{t=1}}^T |c_t - \hat{{d}}_t \zeta_t|
    \]
    subject to the constraints \(|\zeta_t| \leq 1\) and \(\sum_{{t=1}}^T |\zeta_t| \leq \Gamma\).

    However, subject to the budget constraint, the adversary can only choose to set \(\zeta_t = \pm 1\) (i.e., increase the cost by \(\hat{{d}}_t\)) for at most \(\Gamma\) periods. For the remaining periods, \(\zeta_t\) must be set to 0 (cost is \(|x_t - \bar{{d}}_t|\)).

    Thus, the inner maximum is equivalent to (This is a theoretical conclusion that can be used directly):

    \[
    \sum_{{t=1}}^{{T}} |x_t - \bar{{d}}_t| + \max_{{S \subseteq \{{1, \ldots, T\}}, |S| \leq \Gamma}} \sum_{{t \in S}} \hat{{d}}_t
    \]

    where \(S\) is the set of selected periods. The term \(\max_S \sum_{{t \in S}} \hat{{d}}_t\) is the sum of the largest \(\Gamma\) values of \(\hat{{d}}_t\). Let \(\hat{{d}}_{{[1]}} \geq \hat{{d}}_{{[2]}} \geq \cdots \geq \hat{{d}}_{{[T]}}\) denote the ordered deviations. Then:

    \[
    \max_S \sum_{{t \in S}} \hat{{d}}_t = \sum_{{t=1}}^{{\Gamma}} \hat{{d}}_{{[t]}}
    \]

    Therefore, the inner maximum value is:

    \[
    \sum_{{t=1}}^{{T}} |x_t - \bar{{d}}_t| + \sum_{{t=1}}^{{\Gamma}} \hat{{d}}_{{[t]}}
    \]

    (3) Reformulation of the Outer Minimization Problem

    Substituting the result of the inner problem into the outer problem, the original problem transforms into:

    \[
    \min_{{x_t \geq 0}} \left[ \sum_{{t=1}}^T |x_t - \bar{{d}}_t| + \sum_{{t=1}}^\Gamma \hat{{d}}_{{[t]}} \right]
    \]

    The second term, \(\sum_{{t=1}}^\Gamma \hat{{d}}_{{[t]}}\), is a constant independent of the decision variables \(x_t\). Thus, the optimization problem simplifies to:

    \[
    \min_{{x_t \geq 0}} \sum_{{t=1}}^T |x_t - \bar{{d}}_t|
    \]

    (4) Final Transformed RO Model

    Considering the budget uncertainty set, the robust optimization model is equivalent to the following deterministic model:

    \[
    \min_{{x_t \geq 0}} \sum_{{t=1}}^T |x_t - \bar{{d}}_t|
    \]

    where \(\bar{{d}}_t\) is the nominal demand.
    """

    TASK = """
    Optimization problem: 
    {description}

    Initial mathematical RO model:
    {initial_model}
    
    {Gamma} is the budget parameter of budget uncertainty set, controlling the degree of uncertainty.
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
    def run(self, description: str, row: int, Gamma: int, initial_model: str) -> str:
        return self.chain.invoke({"description" : description, "row": str(row), "Gamma": str(Gamma), "initial_model": initial_model})
    

class Code_Generating(MYLLM):

    SYSTEM = """You are a Python programmer working in the fields of operations research and optimization.
    You are proficient in using the third-party solving library like gurobipy. 
    You will receive a specific task description, data, and a mathematical model based on robust optimization method. 
    Your goal is to write a usable Python program. 
    You need to do:
    1. Analyze optimization problem, historical data and mathematical model based on RO model.
    2. Generate the solving code for the RO model.
    3. Print the final solution of optimization problems.
    4. After obtaining the final decision, evaluate it using real data. This requires you to explicitly implement the following processing flow in your code: table data extraction → list conversion → inputting actual values and final optimal solution into the objective function for calculation.
    5. Print the real objective value.
    6. Save the solution, actual value, corresponding decision loss for each decision step, and the real objective value as a text file format in {result_dir}.
    Note: 
    1. Ensure the data loading path in your code is consistent with the provided {train_dir} and {test_dir}. Please strictly import empirical data from {train_dir} and real test data from {test_dir} , and do not fabricate data yourself. Additionally, strictly use “/” as the file path separator.
    2. Ensure strict consistency between imported modules and code usage. For example, if GRB is used in the code, the import statement must contain `from gurobipy import GRB`.
    3. Please note that the table has headers, so do not set header=None when reading it.
    4. Don't output any content contating 'exit()', which will lead to program crash.
    5. Please give the complete code, including the code using ```python```
    """

    INPUT = """
    We already have calculated parameters {d_bar} needed in budget uncertainty set, and {omega} is a parameter of budget uncertainty set.
    The empirical data is from {train_dir} and real test data is from {test_dir}.
    
    Empirical data description is as follows:
    {data_description}

    Specific task: 
    A company needs to transport materials for multiple projects and stockpile materials in the warehouse in advance. 
    The monthly material consumption is uncertain. 
    The goal is to minimize inventory costs while meeting supply and demand. 
    Inventory costs include holding costs and shortage costs, both of which have the same weight of 1 in the objective function. 
    The decision variable is the monthly inventory level.

    Initial mathematical RO model:
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

    Transfered RO model:
    (1) Definition of of Budget Uncertainty Set

    The budget uncertainty set is typically expressed as:

    \[
    \mathcal{{U}} = \left\{{ d_t = \bar{{d}}_t + \hat{{d}}_t \zeta_t \ \middle| \ |\zeta_t| \leq 1, \ \sum_{{t=1}}^T |\zeta_t| \leq \Gamma \right\}}
    \]
    
    where:
    - \(\bar{{d}}_t\) is the nominal demand (the mean value) of empirical data),
    - \(\hat{{d}}_t\) is the maximum demand deviation (magnitude of uncertainty),
    - \(\zeta_t\) is the perturbation variable,
    - \(\Gamma\) is the budget parameter, controlling the degree of uncertainty (\(0 \leq \Gamma \leq T\)).

    (2) Reformulation of the Inner Maximization Problem

    For fixed inventory decisions \(x_t\), the inner problem is to maximize the total cost over the uncertainty set \(\mathcal{{U}}\):

    \[
    \max_{{d_t \in \mathcal{{U}}}} \sum_{{t=1}}^T |x_t - d_t|
    \]

    Substituting \(d_t = \bar{{d}}_t + \hat{{d}}_t \zeta_t\) and letting \(c_t = x_t - \bar{{d}}_t\), the inner problem becomes:

    \[
    \max_{{\zeta_t}} \sum_{{t=1}}^T |c_t - \hat{{d}}_t \zeta_t|
    \]
    subject to the constraints \(|\zeta_t| \leq 1\) and \(\sum_{{t=1}}^T |\zeta_t| \leq \Gamma\).

    However, subject to the budget constraint, the adversary can only choose to set \(\zeta_t = \pm 1\) (i.e., increase the cost by \(\hat{{d}}_t\)) for at most \(\Gamma\) periods. For the remaining periods, \(\zeta_t\) must be set to 0 (cost is \(|x_t - \bar{{d}}_t|\)).

    Thus, the inner maximum is equivalent to (This is a theoretical conclusion that can be used directly):

    \[
    \sum_{{t=1}}^{{T}} |x_t - \bar{{d}}_t| + \max_{{S \subseteq \{{1, \ldots, T\}}, |S| \leq \Gamma}} \sum_{{t \in S}} \hat{{d}}_t
    \]

    where \(S\) is the set of selected periods. The term \(\max_S \sum_{{t \in S}} \hat{{d}}_t\) is the sum of the largest \(\Gamma\) values of \(\hat{{d}}_t\). Let \(\hat{{d}}_{{[1]}} \geq \hat{{d}}_{{[2]}} \geq \cdots \geq \hat{{d}}_{{[T]}}\) denote the ordered deviations. Then:

    \[
    \max_S \sum_{{t \in S}} \hat{{d}}_t = \sum_{{t=1}}^{{\Gamma}} \hat{{d}}_{{[t]}}
    \]

    Therefore, the inner maximum value is:

    \[
    \sum_{{t=1}}^{{T}} |x_t - \bar{{d}}_t| + \sum_{{t=1}}^{{\Gamma}} \hat{{d}}_{{[t]}}
    \]

    (3) Reformulation of the Outer Minimization Problem

    Substituting the result of the inner problem into the outer problem, the original problem transforms into:

    \[
    \min_{{x_t \geq 0}} \left[ \sum_{{t=1}}^T |x_t - \bar{{d}}_t| + \sum_{{t=1}}^\Gamma \hat{{d}}_{{[t]}} \right]
    \]

    The second term, \(\sum_{{t=1}}^\Gamma \hat{{d}}_{{[t]}}\), is a constant independent of the decision variables \(x_t\). Thus, the optimization problem simplifies to:

    \[
    \min_{{x_t \geq 0}} \sum_{{t=1}}^T |x_t - \bar{{d}}_t|
    \]

    (4) Final Transformed RO Model

    Considering the budget uncertainty set, the robust optimization model is equivalent to the following deterministic model:

    \[
    \min_{{x_t \geq 0}} \sum_{{t=1}}^T |x_t - \bar{{d}}_t|
    \]

    where \(\bar{{d}}_t\) is the nominal demand.
    """

    OUTPUT = """```python
    import gurobipy as gp
    import numpy as np
    from gurobipy import GRB
    import pandas as pd
    import os

    # Load Data
    historical_data = pd.read_excel("{train_dir}")
    real_data = pd.read_excel("{test_dir}").values.flatten().tolist()
    T = len(real_data)  # Number of time periods

    # Calculate nominal demand (mean of historical data)
    d_bar = {d_bar}

    # Model initialization
    model = gp.Model("RobustInventoryManagement")

    # Budget parameter Γ
    Gamma = 2

    # Decision variables
    x = model.addVars(T, lb=0, name="x")  

    # Auxiliary variables for absolute values
    y = model.addVars(T, lb=0, name="y")

    # Constraints for absolute value linearization
    for t in range(T):
        model.addConstr(y[t] >= x[t] - d_bar[t], name=f"abs_pos_{{t}}")
        model.addConstr(y[t] >= d_bar[t] - x[t], name=f"abs_neg_{{t}}")

    # Objective: minimize sum of absolute deviations
    model.setObjective(gp.quicksum(y[t] for t in range(T)), GRB.MINIMIZE)

    # Solve the model
    model.optimize()

    if model.status == GRB.OPTIMAL:
        print("\n=== OPTIMAL SOLUTION FOUND ===")

        # Extract solution
        x_opt = [x[t].X for t in range(T)]
        obj_value = model.objVal

        # Evaluate with real historical data
        real_objective_value = 0
        decision_losses = []

        for t in range(T):
            actual_demand = real_data[t]
            decision_loss = abs(x_opt[t] - actual_demand)
            decision_losses.append(decision_loss)
            real_objective_value += decision_loss

        print(f"\nReal objective value (sum of absolute deviations): {{real_objective_value:.2f}}")

        # Save results
        filename = os.path.join("{result_dir}")
        with open(filename, 'w') as f:
            f.write("Month\tInventory Level\tActual Demand\tLoss\n")
            for t in range(T):
                f.write(f"{{t+1}}\t{{x_opt[t]:.2f}}\t{{real_data[t]}}\t{{decision_losses[t]:.2f}}\n")
            f.write(f"\nReal Objective Value: {{real_objective_value:.2f}}\n")

        print(f"\nResults saved to: {{filename}}")  

    else:
        print("No optimal solution found")  
    ```
    """

    TASK = """
    We already have calculated parameters {d_bar} needed in budget uncertainty set, and {omega} is a parameter of budget uncertainty set.
    The empirical data is from {train_dir} and real test data is from {test_dir}.

    Empirical data description is as follows:
    {data_description}

    Specific task: 
    {description}

    Initial mathematical RO model: 
    {initial_model}

    Transfered RO model:
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

    def run(self, description: str, data_description: str, row: int, omega: int, initial_model: str, transfered_model: str, test_dir: str, result_dir: str, train_dir: str, d_bar: list) -> str:
        return self.chain.invoke({"description" : description, "data_description" : data_description, "row": str(row), "omega": str(omega), "initial_model": initial_model, "transfered_model": transfered_model, "test_dir": test_dir, "result_dir" : result_dir, "train_dir": train_dir, "d_bar": d_bar})
    
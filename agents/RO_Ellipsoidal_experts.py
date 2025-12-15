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
    Now you need to transfer the initial RO model based on ellipsoidal uncertainty set.
    You have conducted historical data analysis and obtained the mean and variance for each decision step.
    You need to do:
    1. Analyze optimization problem, initial mathematical model, mean and variance of the empirical data, and parameter omega of ellipsoidal uncertainty set.
    2. Transfer the initial RO model based on ellipsoidal uncertainty set, and give the final optimized model obtained.
    3. Please output two information: the definition of  uncertainty set, the specific model conversion process, and the final optimized model obtained.
    """

    INPUT = """Optimization problem: 
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

    2 is a parameter controlling the conservatism level of ellipsoidal uncertainty set.
    """

    OUTPUT = """
    (1) Definition of Ellipsoidal Uncertainty Set

    An ellipsoidal uncertainty set is used, defined as follows:

    \[
    \mathcal{{U}} = \left\{{ {{d}} \in \mathbb{{R}}^T: \sum_{{t=1}}^{{T}} \frac{{(d_t - \bar{{d}}_t)^2}}{{\sigma_t^2}} \leq \Omega^2 \right\}}
    \]
    
    Where:
    \item $\bar{{d}}_t$ is the nominal consumption (mean value),
    \item $\sigma_t$ is the standard deviation of empirical data,
    \item $\Omega$ is a parameter controlling the conservatism level (typically $\Omega \geq 0$).

    (2) Treatment of the Inner Maximization Problem

    The inner problem is:
    \[
    \max_{{d_t \in \mathcal{{U}}}} \sum_{{t=1}}^{{T}} |x_t - d_t|
    \]

    The maximum value of the inner problem is (This conclusion can be used directly):
    \[
    \max_{{d_t \in \mathcal{{U}}}} \sum_{{t=1}}^{{T}} |x_t - d_t| = \sum_{{t=1}}^{{T}} |x_t - \bar{{d}}_t| + \Omega \sqrt{{\sum_{{t=1}}^{{T}} \sigma_t^2}}
    \]

    (3) Transformation of the Outer Minimization Problem

    Substituting the result of the inner problem into the outer problem:
    \[
    \min_{{x_t \geq 0}} \left[ \sum_{{t=1}}^{{T}} |x_t - \bar{{d}}_t| + \Omega \sqrt{{\sum_{{t=1}}^{{T}} \sigma_t^2}} \right]
    \]

    Here, $\Omega \sqrt{{\sum_{{t=1}}^{{T}} \sigma_t^2}}$ is a constant independent of the decision variables $x_t$. Therefore, the minimization problem is equivalent to:

    \min \sum_{{t=1}}^{{T}} |z - d_t|

    (4) Final Transformed RO Model

    The transformed model is a deterministic optimization problem:
    \[
    \min_{{x_t}} \sum_{{t=1}}^{{T}} |x_t - \bar{{d}}_t|
    \]
    subject to $x_t \geq 0, \quad t = 1, 2, \ldots, T$

    where $\bar{{d}}_t$ is the nominal consumption (mean value).
    """

    TASK = """
    Optimization problem: 
    {description}

    Initial mathematical RO model:
    {initial_model}
    
    {omega} is a parameter controlling the conservatism level of ellipsoidal uncertainty set.
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

    def run(self, description: str, row: int, omega: int, initial_model: str) -> str:
        return self.chain.invoke({"description" : description, "row": str(row), "omega": str(omega), "initial_model": initial_model})
    

class Code_Generating(MYLLM):


    SYSTEM = """You are a Python programmer working in the fields of operations research and optimization.
    You are proficient in using the third-party solving library like gurobipy. 
    You have conducted historical data analysis and obtained the mean and variance for each decision step.
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
    We already have calculated parameters {d_bar} needed in ellipsoidal uncertainty set, and {omega} is a parameter controlling the conservatism level of ellipsoidal uncertainty set.
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
    (1) Definition of Ellipsoidal Uncertainty Set

    An ellipsoidal uncertainty set is used, defined as follows:

    \[
    \mathcal{{U}} = \left\{{ {{d}} \in \mathbb{{R}}^T: \sum_{{t=1}}^{{T}} \frac{{(d_t - \bar{{d}}_t)^2}}{{\sigma_t^2}} \leq \Omega^2 \right\}}
    \]
    
    Where:
    \item $\bar{{d}}_t$ is the nominal consumption (mean value),
    \item $\sigma_t$ is the standard deviation of empirical data,
    \item $\Omega$ is a parameter controlling the conservatism level (typically $\Omega \geq 0$).

    (2) Treatment of the Inner Maximization Problem

    The inner problem is:
    \[
    \max_{{d_t \in \mathcal{{U}}}} \sum_{{t=1}}^{{T}} |x_t - d_t|
    \]

    The maximum value of the inner problem is (This conclusion can be used directly):
    \[
    \max_{{d_t \in \mathcal{{U}}}} \sum_{{t=1}}^{{T}} |x_t - d_t| = \sum_{{t=1}}^{{T}} |x_t - \bar{{d}}_t| + \Omega \sqrt{{\sum_{{t=1}}^{{T}} \sigma_t^2}}
    \]

    (3) Transformation of the Outer Minimization Problem

    Substituting the result of the inner problem into the outer problem:
    \[
    \min_{{x_t \geq 0}} \left[ \sum_{{t=1}}^{{T}} |x_t - \bar{{d}}_t| + \Omega \sqrt{{\sum_{{t=1}}^{{T}} \sigma_t^2}} \right]
    \]

    Here, $\Omega \sqrt{{\sum_{{t=1}}^{{T}} \sigma_t^2}}$ is a constant independent of the decision variables $x_t$. Therefore, the minimization problem is equivalent to:

    \min \sum_{{t=1}}^{{T}} |z - d_t|

    (4) Final Transformed RO Model

    The transformed model is a deterministic optimization problem:
    \[
    \min_{{x_t}} \sum_{{t=1}}^{{T}} |x_t - \bar{{d}}_t|
    \]
    subject to $x_t \geq 0, \quad t = 1, 2, \ldots, T$

    where $\bar{{d}}_t$ is the nominal consumption (mean value).
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

    # Decision variables
    x = model.addVars(T, lb=0, name="x")  

    # Auxiliary variables for absolute values |x_t - d_bar|
    abs_diff = model.addVars(T, name="abs_diff")

    # Constraints for absolute value linearization
    for t in range(T):
        model.addConstr(abs_diff[t] >= x[t] - d_bar[t])
        model.addConstr(abs_diff[t] >= d_bar[t] - x[t])

    # Objective: minimize sum of absolute deviations
    objective = gp.quicksum(abs_diff[t] for t in range(T))
    model.setObjective(objective, GRB.MINIMIZE)

    # Solve the model
    model.optimize()

    if model.status == GRB.OPTIMAL:
        print("\n=== OPTIMAL SOLUTION FOUND ===")

        # Extract solution
        optimal_x = [x[t].X for t in range(T)]
        optimal_obj_value = model.objVal

        # Evaluate with real historical data
        real_objective_value = 0
        decision_losses = []

        for t in range(T):
            actual_demand = real_data[t]
            decision_loss = abs(optimal_x[t] - actual_demand)
            decision_losses.append(decision_loss)
            real_objective_value += decision_loss

        print(f"\nReal objective value (sum of absolute deviations): {{real_objective_value:.2f}}")

        # Save results
        filename = os.path.join("{result_dir}")
        with open(filename, 'w') as f:
            f.write("Month\tInventory Level\tActual Demand\tLoss\n")
            for t in range(T):
                f.write(f"{{t+1}}\t{{optimal_x[t]:.2f}}\t{{real_data[t]}}\t{{decision_losses[t]:.2f}}\n")
            f.write(f"\nReal Objective Value: {{real_objective_value:.2f}}\n")

    print(f"\nResults saved to: {{filename}}")  

    else:
        print("No optimal solution found")  
    ```
    """

    TASK = """
    We already have calculated parameters {d_bar} needed in ellipsoidal uncertainty set, and {omega} is a parameter controlling the conservatism level of ellipsoidal uncertainty set.
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
        return self.chain.invoke({"description" : description, "data_description" : data_description, "row": str(row), "omega": str(omega), "initial_model": initial_model, "transfered_model": transfered_model, "test_dir" : test_dir, "result_dir" : result_dir, "train_dir" : train_dir, "d_bar": d_bar})

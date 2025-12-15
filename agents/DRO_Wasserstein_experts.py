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
    Now you need to transfer the initial DRO model into tractable form based on wasserstein ambiguity set using the strong duality theory.
    You need to do:
    1. Analyze optimization problem and initial mathematical model.
    2. Transfer the initial DRO model into tractable form based on wasserstein ambiguity set using the strong duality theory, and give the final optimized model obtained.
    3. Please output six information: definition of ambiguity set, strong duality application, inner problem calculation, tractable model conversion, the final optimized model, and the empirical sample mean approximation form.
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
    (1) Ambiguity set definition

    The initial model is:
    \[
    \min_{{x_t \geq 0}} \max_{{P \in \mathcal{{P}}}} \mathbb{{E}}_P \left[ \sum_{{t=1}}^T |x_t - d_t| \right],
    \]
    where $\mathcal{{P}}$ is the Wasserstein ambiguity set with radius $\theta$ around a reference distribution $P_0$. 

    The cost function $\sum_{{t=1}}^T |x_t - d_t|$ represents the total inventory cost (holding and shortage costs with equal weight).

    (2) Strong Duality Application

    For the Wasserstein ambiguity set with L1 norm as the cost function (i.e., $c(\zeta, d) = \|\zeta - d\|_1 = \sum_{{t=1}}^T |\zeta_t - d_t|$), the inner max problem can be expressed using strong duality:
    \[
    \max_{{P \in \mathcal{{P}} \mathbb{{E}}_P \left[ \sum_{{t=1}}^T |x_t - d_t| \right] = \inf_{{\lambda \geq 0}} \left\{{ \lambda \theta + \mathbb{{E}}_{{P_0}} \left[ \sup_\zeta \left\{{ \sum_{{t=1}}^T |x_t - \zeta_t| - \lambda \sum_{{t=1}}^T |\zeta_t - d_t| \right\}} \right] \right\}}.
    \]

    (3) Inner Supremum Calculation

    The inner supremum
    \[
    \sup_\zeta \left\{{ \sum_{{t=1}}^T |x_t - \zeta_t| - \lambda \sum_{{t=1}}^T |\zeta_t - d_t| \right\}}
    \]
    is separable by time period $t$. For each $t$, we compute:
    \[
    \sup_\zeta \left\{{ |x_t - \zeta_t| - \lambda |\zeta_t - d_t| \right\}}.
    \]

    Using the triangle inequality, for $\lambda \geq 1$, this supremum equals $|x_t - d_t|$, achieved at $\zeta_t = d_t$. Thus, the inner expression simplifies to $\sum_{{t=1}}^T |x_t - d_t|$.

    (4) Tractable Model Conversion

    The duality expression becomes:

    \[
    \inf_{{\lambda \geq 0}} \left\{{ \lambda \theta + \mathbb{{E}}_{{P_0}} \left[ \sum_{{t=1}}^{{T}} |x_t - d_t| \right] \right\}} = \inf_{{\lambda \geq 1}} \left\{{ \lambda \theta + \mathbb{{E}}_{{P_0}} \left[ \sum_{{t=1}}^{{T}} |x_t - d_t| \right] \right\}},
    \]

    since for $\lambda < 1$, the supremum is infinite. The infimum is achieved at $\lambda = 1$, yielding:

    \[
    \theta + \mathbb{{E}}_{{P_0}} \left[ \sum_{{t=1}}^{{T}} |x_t - d_t| \right].
    \]   

    The worst-case expected cost is 
    
    $\theta + \mathbb{{E}}_{{P_0}} \left[ \sum_{{t=1}}^{{T}} |x_t - d_t| \right]$. 
    
    Since $\theta$ is constant, minimizing this is equivalent to minimizing
     
    $\mathbb{{E}}_{{P_0}} \left[ \sum_{{t=1}}^{{T}} |x_t - d_t| \right]$ subject to $x_t \geq 0$.

    (5) Final Tractable Model

    The tractable form of the DRO model is:

    \[
    \min_{{x_t \geq 0}} \mathbb{{E}}_{{P_0}} \left[ \sum_{{t=1}}^{{T}} |x_t - d_t| \right],
    \]

    where:
    \begin{{itemize}}
        \item $x_t$ is the inventory level in month $t$ (decision variable),
        \item $d_t$ is the random demand in month $t$,
        \item $P_0$ is the reference distribution for demands (e.g., empirical distribution from historical data),
        \item $T$ is the total number of months.
    \end{{itemize}}

    (6) Empirical Sample Mean Approximation Form
    
    This model can be solved using sample average approximation (SAA) if $P_0$ is empirical. Specifically, with historical demand samples $\{{d_t^{{(i)}}\}}_{{i=1}}^N$ for each $t$, the objective becomes:

    \[
    \min_{{x_t \geq 0}} \frac{{1}}{{N}} \sum_{{i=1}}^N \sum_{{t=1}}^T |x_t - d_t^{{(i)}}|.
    \]

    The optimal solution sets each $x_t$ to the median of the demand distribution for period $t$ under $P_0$, as the median minimizes the expected absolute deviation.
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
    You are proficient in using the third-party solving library like gurobipy. 
    You will receive a specific task description, data, and a mathematical model based on distributionally robust optimization method.
    Your goal is to write a usable Python program. 
    You need to do:
    1. Analyze optimization problem, historical data and mathematical model based on DRO model.
    2. Generate the solving code for the DRO model.
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
    We already have calculated the parameters needed in wasserstein ambiguity set, where {mean} represent the mean of empirical data for each decision step.
    The parameter(radius) needed in wasserstein ambiguity set is {theta}.
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

    Transferred DRO model:
    (1) Ambiguity set definition

    The initial model is:
    \[
    \min_{{x_t \geq 0}} \max_{{P \in \mathcal{{P}}}} \mathbb{{E}}_P \left[ \sum_{{t=1}}^T |x_t - d_t| \right],
    \]
    where $\mathcal{{P}}$ is the Wasserstein ambiguity set with radius $\theta$ around a reference distribution $P_0$. 

    The cost function $\sum_{{t=1}}^T |x_t - d_t|$ represents the total inventory cost (holding and shortage costs with equal weight).

    (2) Strong Duality Application

    For the Wasserstein ambiguity set with L1 norm as the cost function (i.e., $c(\zeta, d) = \|\zeta - d\|_1 = \sum_{{t=1}}^T |\zeta_t - d_t|$), the inner max problem can be expressed using strong duality:
    \[
    \max_{{P \in \mathcal{{P}} \mathbb{{E}}_P \left[ \sum_{{t=1}}^T |x_t - d_t| \right] = \inf_{{\lambda \geq 0}} \left\{{ \lambda \theta + \mathbb{{E}}_{{P_0}} \left[ \sup_\zeta \left\{{ \sum_{{t=1}}^T |x_t - \zeta_t| - \lambda \sum_{{t=1}}^T |\zeta_t - d_t| \right\}} \right] \right\}}.
    \]

    (3) Inner Supremum Calculation

    The inner supremum
    \[
    \sup_\zeta \left\{{ \sum_{{t=1}}^T |x_t - \zeta_t| - \lambda \sum_{{t=1}}^T |\zeta_t - d_t| \right\}}
    \]
    is separable by time period $t$. For each $t$, we compute:
    \[
    \sup_\zeta \left\{{ |x_t - \zeta_t| - \lambda |\zeta_t - d_t| \right\}}.
    \]

    Using the triangle inequality, for $\lambda \geq 1$, this supremum equals $|x_t - d_t|$, achieved at $\zeta_t = d_t$. Thus, the inner expression simplifies to $\sum_{{t=1}}^T |x_t - d_t|$.

    (4) Expectation under Reference Distribution

    The duality expression becomes:

    \[
    \inf_{{\lambda \geq 0}} \left\{{ \lambda \theta + \mathbb{{E}}_{{P_0}} \left[ \sum_{{t=1}}^{{T}} |x_t - d_t| \right] \right\}} = \inf_{{\lambda \geq 1}} \left\{{ \lambda \theta + \mathbb{{E}}_{{P_0}} \left[ \sum_{{t=1}}^{{T}} |x_t - d_t| \right] \right\}},
    \]

    since for $\lambda < 1$, the supremum is infinite. The infimum is achieved at $\lambda = 1$, yielding:

    \[
    \theta + \mathbb{{E}}_{{P_0}} \left[ \sum_{{t=1}}^{{T}} |x_t - d_t| \right].
    \]   

    The worst-case expected cost is 
    
    $\theta + \mathbb{{E}}_{{P_0}} \left[ \sum_{{t=1}}^{{T}} |x_t - d_t| \right]$. 
    
    Since $\theta$ is constant, minimizing this is equivalent to minimizing
     
    $\mathbb{{E}}_{{P_0}} \left[ \sum_{{t=1}}^{{T}} |x_t - d_t| \right]$ subject to $x_t \geq 0$.

    (5) Final Tractable Model

    The tractable form of the DRO model is:

    \[
    \min_{{x_t \geq 0}} \mathbb{{E}}_{{P_0}} \left[ \sum_{{t=1}}^{{T}} |x_t - d_t| \right],
    \]

    where:
    \begin{{itemize}}
        \item $x_t$ is the inventory level in month $t$ (decision variable),
        \item $d_t$ is the random demand in month $t$,
        \item $P_0$ is the reference distribution for demands (e.g., empirical distribution from historical data),
        \item $T$ is the total number of months.
    \end{{itemize}}

    (6) Empirical Sample Mean Approximation Form
    
    This model can be solved using sample average approximation (SAA) if $P_0$ is empirical. Specifically, with historical demand samples $\{{d_t^{{(i)}}\}}_{{i=1}}^N$ for each $t$, the objective becomes:

    \[
    \min_{{x_t \geq 0}} \frac{{1}}{{N}} \sum_{{i=1}}^N \sum_{{t=1}}^T |x_t - d_t^{{(i)}}|.
    \]

    The optimal solution sets each $x_t$ to the median of the demand distribution for period $t$ under $P_0$, as the median minimizes the expected absolute deviation.
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
    T, N = historical_data.shape
    l_t = {{lower}}  # Lower bounds for each month
    u_t = {{upper}}  # Upper bounds for each month

    # Calculate mean for each timestep
    mu = np.mean(historical_data, axis=1)

    # Calculate lambda_t for each period
    lambda_t = []
    for t in range(T):
        if u_t[t] != l_t[t]:
            lambda_val = (u_t[t] - mu[t]) / (u_t[t] - l_t[t])
            lambda_val = max(0, min(1, lambda_val))
        else:
            lambda_val = 0.5 
        lambda_t.append(lambda_val)

    # Create model
    model = gp.Model("DRO_Inventory_Optimization")

    # Decision variables
    x = model.addVars(T, lb=0.0, name="inventory")  # Inventory levels
    a = model.addVars(T, lb=0.0, name="a")  # Auxiliary variables for |x_t - l_t|
    b = model.addVars(T, lb=0.0, name="b")  # Auxiliary variables for |x_t - u_t|

    # Objective function: minimize sum_t [lambda_t * a_t + (1 - lambda_t) * b_t]
    objective = gp.quicksum(lambda_t[t] * a[t] + (1 - lambda_t[t]) * b[t] for t in range(T))
    model.setObjective(objective, GRB.MINIMIZE)

    # Constraints for absolute values
    for t in range(T):
        model.addConstr(a[t] >= x[t] - l_t[t], f"a_lower_{{t}}")
        model.addConstr(a[t] >= l_t[t] - x[t], f"a_upper_{{t}}")
        model.addConstr(b[t] >= x[t] - u_t[t], f"b_lower_{{t}}")
        model.addConstr(b[t] >= u_t[t] - x[t], f"b_upper_{{t}}")

    # Solve the model
    model.setParam('OutputFlag', 1)
    model.optimize()

    # Check solution status
    if model.status == GRB.OPTIMAL:
        print("\n=== OPTIMAL SOLUTION FOUND ===")

        # Extract solution
        x_opt = np.array([x[t].X for t in range(T)])
        obj_value = model.objVal

        # Evaluate with real historical data
        if real_data.shape[1] > 1:
            # If multiple samples, take mean for each timestep
            real_demand = np.mean(real_data, axis=1)
        else:
            # If single column, flatten
            real_demand = real_data.flatten()
        
        real_obj_value = np.sum(np.abs(x_opt - real_demand))
        decision_losses = np.abs(x_opt - real_demand)

        print("\nDecision losses (|x_t - d_t|):")
        for t in range(T):
            print(f"Month {{t+1}}: {{decision_losses[t]}}")
        print(f"\nObjective value based on real data: {{real_obj_value}}")

    else:
        print("No optimal solution found")  
    ```
    """

    TASK = """
    We already have calculated the parameters needed in wasserstein ambiguity set, where {mean} represent the mean of empirical data for each decision step.
    The parameter(radius) needed in wasserstein ambiguity set is {theta}.
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

    def run(self, description: str, data_description: str, row: int, theta: int, mean: list, initial_model: str, transfered_model: str, train_dir: str, test_dir: str, result_dir: str) -> str:
        return self.chain.invoke({"description" : description, "data_description" : data_description, "row": str(row), "theta": theta, "mean": mean, "initial_model": initial_model, "transfered_model": transfered_model, "train_dir" : train_dir, "test_dir" : test_dir, "result_dir" : result_dir})
    

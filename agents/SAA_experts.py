from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from utils import MYLLM

class SAA_Auto_Modeling(MYLLM):

    SYSTEM = """You are an expert in operation research and optimization.
    You are now facing an optimization problem under uncertainty. 
    Now we need to use SAA method to model uncertain optimization problem based on empirical data. 
    You need to do:
    1. Analyze optimization problem and empirical data matrix.
    2. Construct SAA model with Latex language for formulation.
    3. When establishing an optimization model, it is necessary to consider the realistic constraints in the problem, such as inventory cannot be negative.
    4. Please output four information: Objectives, Variables, Constraints, and SAA Method
    """

    # INPUT = """Optimization problem: 
    # A company needs to transport materials for multiple projects and stockpile materials in the warehouse in advance. 
    # The monthly material consumption is uncertain, so the company needs to decide the monthly inventory level based on the predicted demand range. 
    # The goal is to minimize inventory costs while meeting supply and demand. 
    # Inventory costs include holding costs and shortage costs, both of which have the same weight of 1 in the objective function. 
    # The decision variable is the monthly inventory level.
    # """

    # OUTPUT = """
    # ### **1. Objectives**

    # Minimize the expected total inventory cost across all demand scenarios over $T$ time steps, where the cost consists of both **holding cost** and **shortage cost**, and both are equally weighted:

    # $$
    # \min_{{x_t}} \quad \frac{{1}}{{S}} \sum_{{t=1}}^{{T}} \sum_{{s=1}}^{{S}} \left[ \max(0, x_t - d_t^{{(s)}}) + \max(0, d_t^{{(s)}} - x_t) \right]
    # $$

    # Where:

    # * $x_t$ is the decision variable: inventory level in month $t$
    # * $d_t^{{(s)}}$ is the $s$-th sample of demand in month $t$
    # * $S$ is the number of scenarios, $T$ is the number of time steps

    # ---

    # ### **2. Variables**

    # * $x_t \in \mathbb{{R}}_+$: Inventory level for month $t$, for $t = 1, 2, \dots, T$
    # * $z_t^{{(s)}} \in \mathbb{{R}}_+$: Absolute deviation between decision and sampled demand (auxiliary variable)

    # ---

    # ### **3. Constraints**

    # To linearize the objective with absolute values, we introduce constraints:

    # $$
    # \begin{{aligned}}
    # z_t^{{(s)}} &\geq x_t - d_t^{{(s)}} \quad &\text{{for }} t = 1,\dots,T,\; s = 1,\dots,S \\
    # z_t^{{(s)}} &\geq d_t^{{(s)}} - x_t \quad &\text{{for }} t = 1,\dots,T,\; s = 1,\dots,S \\
    # x_t &\geq 0 \quad &\text{{for }} t = 1,\dots,T \\
    # z_t^{{(s)}} &\geq 0 \quad &\text{{for }} t = 1,\dots,T,\; s = 1,\dots,S
    # \end{{aligned}}
    # $$

    # ---

    # ### **4. SAA Method (Sample Average Approximation)**

    # The expected cost due to uncertain demand $D_t \sim \mathcal{{D}}_t$ is intractable analytically. Hence, we replace the true expectation with an empirical mean based on $S$ demand scenarios:

    # #### True Expected Objective:

    # $$
    # \min_{{x_t}} \sum_{{t=1}}^{{T}} \mathbb{{E}}_{{D_t}} \left[ \max(0, x_t - D_t) + \max(0, D_t - x_t) \right]
    # $$

    # #### SAA Approximation:

    # $$
    # \min_{{x_t}} \quad \frac{{1}}{{S}} \sum_{{t=1}}^{{T}} \sum_{{s=1}}^{{S}} \left| x_t - d_t^{{(s)}} \right|
    # $$

    # This converts the stochastic problem into a **deterministic convex optimization problem** solvable with linear programming.
    # """

    TASK = """
    Optimization problem: 
    {description}
    """
        
    def __init__(self):
        super().__init__()
        prompt = ChatPromptTemplate([
            ("system",self.SYSTEM), 
            # ("human", self.INPUT),
            # ("ai", self.OUTPUT),
            ("human", self.TASK),
        ])
        self.chain = prompt | self.llm | StrOutputParser()

    def run(self, description: str, row: int, col: int) -> str:
        return self.chain.invoke({"description" : description, "row": str(row), "col": str(col)})



class Code_Generating(MYLLM):


    SYSTEM = """You are a Python programmer working in the fields of operations research and optimization.
    You are proficient in using the third-party solving library like gurobipy. 
    You will receive a specific problem description, data, and a mathematical model based on SAA method. 
    Your goal is to write a usable Python program. 
    You need to do:
    1. Analyze optimization problem and empirical data matrix.
    2. Generate the solving code for the SAA optimization model.
    3. Print the final solution of optimization problems.
    4. Evaluate the solution with real test data from an excel file with the path: {test_dir}, and print the evaluation results.
    Note: 
    1. Ensure the data loading path in your code is consistent with the provided {train_dir} and {test_dir}. Please strictly import empirical data from {train_dir} and real test data from {test_dir} , and do not fabricate data yourself.Additionally, strictly use “/” as the file path separator.
    2. Please note that the table has headers, so do not set header=None when reading it.
    3. Don't output any content contating 'exit()', which will lead to program crash.
    4. Please give the complete code, including the code using ```python```
    """

    INPUT = """Each row of the empirical data matrix represents a decision at a timestep, with {row} rows and each column represents the scenarios at that time step, with {col} columns.
    The empirical data is from {train_dir} and real test data is from {test_dir}.
    
    Specific task: 
    A company needs to transport materials for multiple projects and stockpile materials in the warehouse in advance. 
    The monthly material consumption is uncertain, so the company needs to decide the monthly inventory level based on the predicted demand range. 
    The goal is to minimize inventory costs while meeting supply and demand. 
    Inventory costs include holding costs and shortage costs, both of which have the same weight of 1 in the objective function. 
    The decision variable is the monthly inventory level.

    SAA model: 
    ### **1. Objectives**

    Minimize the expected total inventory cost across all demand scenarios over $T$ time steps, where the cost consists of both **holding cost** and **shortage cost**, and both are equally weighted:

    $$
    \min_{{x_t}} \quad \frac{{1}}{{S}} \sum_{{t=1}}^{{T}} \sum_{{s=1}}^{{S}} \left[ \max(0, x_t - d_t^{{(s)}}) + \max(0, d_t^{{(s)}} - x_t) \right]
    $$

    Where:

    * $x_t$ is the decision variable: inventory level in month $t$
    * $d_t^{{(s)}}$ is the $s$-th sample of demand in month $t$
    * $S$ is the number of scenarios, $T$ is the number of time steps

    ---

    ### **2. Variables**

    * $x_t \in \mathbb{{R}}_+$: Inventory level for month $t$, for $t = 1, 2, \dots, T$
    * $z_t^{{(s)}} \in \mathbb{{R}}_+$: Absolute deviation between decision and sampled demand (auxiliary variable)

    ---

    ### **3. Constraints**

    To linearize the objective with absolute values, we introduce constraints:

    $$
    \begin{{aligned}}
    z_t^{{(s)}} &\geq x_t - d_t^{{(s)}} \quad &\text{{for }} t = 1,\dots,T,\; s = 1,\dots,S \\
    z_t^{{(s)}} &\geq d_t^{{(s)}} - x_t \quad &\text{{for }} t = 1,\dots,T,\; s = 1,\dots,S \\
    x_t &\geq 0 \quad &\text{{for }} t = 1,\dots,T \\
    z_t^{{(s)}} &\geq 0 \quad &\text{{for }} t = 1,\dots,T,\; s = 1,\dots,S
    \end{{aligned}}
    $$

    ---

    ### **4. SAA Method (Sample Average Approximation)**

    The expected cost due to uncertain demand $D_t \sim \mathcal{{D}}_t$ is intractable analytically. Hence, we replace the true expectation with an empirical mean based on $S$ demand scenarios:

    #### True Expected Objective:

    $$
    \min_{{x_t}} \sum_{{t=1}}^{{T}} \mathbb{{E}}_{{D_t}} \left[ \max(0, x_t - D_t) + \max(0, D_t - x_t) \right]
    $$

    #### SAA Approximation:

    $$
    \min_{{x_t}} \quad \frac{{1}}{{S}} \sum_{{t=1}}^{{T}} \sum_{{s=1}}^{{S}} \left| x_t - d_t^{{(s)}} \right|
    $$

    This converts the stochastic problem into a **deterministic convex optimization problem** solvable with linear programming.
    """

    OUTPUT = """```python
    import pickle
    import pandas as pd
    import numpy as np
    from gurobipy import Model, GRB, quicksum
    import os

    # === Step 1: Load data ===
    train_dir = "{train_dir}"  
    test_dir = "{test_dir}"     
    
    # Load training data
    historical_data = pd.read_excel(train_dir) # Please note that the table has headers, so do not set header=None when reading it.
    demand_samples = historical_data.values 

    # Load test data
    real_data = pd.read_excel(test_dir) # Please note that the table has headers, so do not set header=None when reading it.
    real_demands = real_data.values.flatten().tolist()
   
    T = len(demand_samples)     # Number of time steps (days)
    S = len(demand_samples[0])  # Number of samples per time step

    # === Step 2: Create optimization model ===
    model = Model("SAA_Power_Generation_Optimization")
    model.Params.OutputFlag = 1

    # === Step 3: Define variables ===
    g = model.addVars(T, vtype=GRB.CONTINUOUS, lb=0.0, ub=500.0, name="g")
    u = model.addVars(T, S, vtype=GRB.CONTINUOUS, lb=0.0, name="u")

    # === Step 4: Add constraints ===
    for t in range(T):
        for s in range(S):
            d_ts = demand_samples[t][s]
            model.addConstr(u[t, s] >= d_ts - g[t], name=f"shortage_{{t}}_{{s}}")

    # === Step 5: Define the objective function ===
    generation_cost = quicksum(50 * g[t] for t in range(T))
    shortage_penalty = quicksum(100 * u[t, s] for t in range(T) for s in range(S))

    total_cost = (generation_cost + shortage_penalty) / S
    model.setObjective(total_cost, GRB.MINIMIZE)

    # === Step 6: Optimize ===
    model.optimize()

    # === Step 7: Extract and save solution ===
    if model.status == GRB.OPTIMAL:
        
        optimal_generation = []
        for t in range(T):
            gen_value = g[t].X
            optimal_generation.append(gen_value)
            print(f"Day {{t+1}}: {{gen_value:.2f}} MW")
        
        # === Step 8: Evaluate with real historical data ===
        test_data_path = "{test_dir}"
        try:
            # Load test data
            test_df = pd.read_excel(test_data_path)
            
            # Assuming the last column contains the real demand data
            last_column_name = test_df.columns[-1]
            real_demands = test_df[last_column_name].tolist()
            
            # Ensure we have enough test data
            if len(real_demands) < T:
                print(f"Warning: Test data has only {{len(real_demands)}} days, but we need {{T}} days")
                real_demands = real_demands + [0] * (T - len(real_demands))
            elif len(real_demands) > T:
                real_demands = real_demands[:T]
            
            # Calculate actual cost with real demands
            total_actual_cost = 0
            print("\n=== EVALUATION WITH REAL HISTORICAL DATA ===")
            print("Day | Generation | Real Demand | Shortage | Daily Cost")
            print("-" * 55)
            
            for t in range(T):
                generation = optimal_generation[t]
                real_demand = real_demands[t]
                shortage = max(0, real_demand - generation)
                daily_cost = 50 * generation + 100 * shortage
                total_actual_cost += daily_cost
                
                print(f"{{t+1:3d}} | {{generation:10.2f}} | {{real_demand:11.2f}} | {{shortage:8.2f}} | ${{daily_cost:9.2f}}")
            
            print("-" * 55)
            print(f"Total actual cost: ${{total_actual_cost:.2f}}")
            
        except FileNotFoundError:
            print(f"Test data file not found: {{test_data_path}}")
        except Exception as e:
            print(f"Error loading test data: {{e}}")
            
    else:
        print("Optimization failed. Status:", model.status)
    ```
    """

    TASK = """
    The empirical data is from {train_dir} and real test data is from {test_dir}.

    Empirical data description is as follows:
    {data_description}

    Specific task: 
    {description}

    SAA model: 
    {math_model}
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

    def run(self, description: str, data_description: str, row: int, col: int, math_model: str, train_dir: str, result_dir: str, test_dir: str) -> str:
        return self.chain.invoke({"description" : description, "data_description" : data_description, "row": str(row), "col": str(col), "math_model": math_model, "train_dir" : train_dir, "result_dir" : result_dir, "test_dir" : test_dir})
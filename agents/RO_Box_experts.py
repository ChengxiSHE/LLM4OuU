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
    Now you need to transfer the initial RO model based on box uncertainty set.
    You have conducted historical data analysis and obtained the lower and upper bounds of box uncertainty set for each decision step.
    You need to do:
    1. Analyze optimization problem, initial mathematical model and boundary of box uncertainty set.
    2. Transfer the initial RO model based on box uncertainty set, and give the final optimized model obtained.
    3. Please output three information: the construction uncertainty set, the specific model conversion process, and the final optimized model obtained.
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
    """

    OUTPUT = """
    (1) Uncertainty set construction

    Since the uncertainty set (the box-shaped set formed by the demand interval $[l_t, u_t]$) is box-shaped and the objective function is separable, the above min-max problem can be decomposed into independent problems for each month, where the “sum of maximums” equals the “maximum of sums”:

    $$
    \max_{{\substack{{d_t \in [l_t, u_t]}} \\ t=1,2,\dots,T}} 
    \sum_{{t=1}}^{{T}} |x_t - d_t| 
    = 
    \sum_{{t=1}}^{{T}} \max_{{d_t \in [l_t, u_t]}} |x_t - d_t|
    $$

    (2) Model Transformation

    For each $t$, compute $\max_{{d_t \in [l_t, u_t]}} |x_t - d_t|$: Since the absolute value function is convex, its maximum value within the interval must occur at one of the interval endpoints. Thus, this maximum can be expressed as:   

    $$
    \max_{{d_t \in [l_t, u_t]}} |x_t - d_t| 
    = 
    \max\left\{{ u_t - x_t, x_t - l_t }}\right\
    $$

    Thus, the original min-max problem transforms into an optimization problem involving only the decision variable $x_t$:

    $$
    \min_{{\substack{{x_t \geq 0 \ t=1,2,\dots,T}}}} 
    \sum_{{t=1}}^{{T}} \max\left\{{ u_t - x_t, x_t - l_t }}\right\
    $$

    To transform this nonlinear model containing $\max$ functions into a linear programming problem, we introduce auxiliary variables $y_t$ (corresponding to the worst-case cost in month $t$), such that:

    $$
    y_t \geq u_t - x_t, 
    \quad 
    y_t \geq x_t - l_t 
    \quad (\forall \, t=1,2,\dots,T)
    $$

    At this point, $y_t$ can equivalently replace 
    $\max\left\{{ u_t - x_t, x_t - l_t }}\right\$, 
    completing the linearization of the model.

    (3) Final Linear Programming Model LaTeX Code:

    $$
    \min \quad \sum_{{t=1}}^{{T}} y_t
    $$

    subject to:
    \[
    \begin{{cases}}
    y_t \geq u_t - x_t & \forall \, t = 1, \ldots, T \\
    y_t \geq x_t - l_t & \forall \, t = 1, \ldots, T \\
    x_t \geq 0 & \forall \, t = 1, \ldots, T \\
    y_t \geq 0 & \forall \, t = 1, \ldots, T
    \end{{cases}}
    \]

    where $y_t$ is automatically non-negative (guaranteed by the constraints), but for completeness, we explicitly include the constraint $y_t \geq 0$.
    """

    TASK = """
    Optimization problem: 
    {description}

    Initial mathematical RO model:
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
    1. Ensure the data loading path in your code is consistent with the provided {train_dir} and {test_dir}. Please strictly import empirical data from {train_dir} and real test data from {test_dir} , and do not fabricate data yourself.Additionally, strictly use “/” as the file path separator.
    2. Ensure strict consistency between imported modules and code usage. For example, if GRB is used in the code, the import statement must contain `from gurobipy import GRB`.
    3. Please note that the table has headers, so do not set header=None when reading it.
    4. Don't output any content contating 'exit()', which will lead to program crash.
    5. Please give the complete code, including the code using ```python```
    """

    INPUT = """
    We already have calculated the box uncertainty set boundary, where {lower} and {upper} represent the lower and upper bounds of box uncertainty respectively.
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

    Each row of the test data represents a decision at a timestep, with 10 rows.

    Transfered RO model:
    (1) Uncertainty set construction

    Since the uncertainty set (the box-shaped set formed by the demand interval $[l_t, u_t]$) is box-shaped and the objective function is separable, the above min-max problem can be decomposed into independent problems for each month, where the “sum of maximums” equals the “maximum of sums”:

    $$
    \max_{{\substack{{d_t \in [l_t, u_t]}} \\ t=1,2,\dots,T}} 
    \sum_{{t=1}}^{{T}} |x_t - d_t| 
    = 
    \sum_{{t=1}}^{{T}} \max_{{d_t \in [l_t, u_t]}} |x_t - d_t|
    $$

    (2) Model Transformation

    For each $t$, compute $\max_{{d_t \in [l_t, u_t]}} |x_t - d_t|$: Since the absolute value function is convex, its maximum value within the interval must occur at one of the interval endpoints. Thus, this maximum can be expressed as:   

    $$
    \max_{{d_t \in [l_t, u_t]}} |x_t - d_t| 
    = 
    \max\left\{{ u_t - x_t, x_t - l_t }}\right\
    $$

    Thus, the original min-max problem transforms into an optimization problem involving only the decision variable $x_t$:

    $$
    \min_{{\substack{{x_t \geq 0 \ t=1,2,\dots,T}}}} 
    \sum_{{t=1}}^{{T}} \max\left\{{ u_t - x_t, x_t - l_t }}\right\
    $$

    To transform this nonlinear model containing $\max$ functions into a linear programming problem, we introduce auxiliary variables $y_t$ (corresponding to the worst-case cost in month $t$), such that:

    $$
    y_t \geq u_t - x_t, 
    \quad 
    y_t \geq x_t - l_t 
    \quad (\forall \, t=1,2,\dots,T)
    $$

    At this point, $y_t$ can equivalently replace 
    $\max\left\{{ u_t - x_t, x_t - l_t }}\right\$, 
    completing the linearization of the model.

    (3) Final Linear Programming Model LaTeX Code:

    $$
    \min \quad \sum_{{t=1}}^{{T}} y_t
    $$

    subject to:
    \[
    \begin{{cases}}
    y_t \geq u_t - x_t & \forall \, t = 1, \ldots, T \\
    y_t \geq x_t - l_t & \forall \, t = 1, \ldots, T \\
    x_t \geq 0 & \forall \, t = 1, \ldots, T \\
    y_t \geq 0 & \forall \, t = 1, \ldots, T
    \end{{cases}}
    \]

    where $y_t$ is automatically non-negative (guaranteed by the constraints), but for completeness, we explicitly include the constraint $y_t \geq 0$.
    """

    OUTPUT = """```python
    import gurobipy as gp
    from gurobipy import GRB
    import pandas as pd
    import os
    from utils import save_txt, extract_python_code, safe_exec, safe_exec_with_flag, calculate_bounds

    # Load Data
    real_data = pd.read_excel("{test_dir}").values.flatten().tolist()
    T = len(real_data)  # Number of months
    l_t = {{lower}}  # Lower bounds for each month
    u_t = {{upper}}  # Upper bounds for each month

    # Model initialization
    model = gp.Model("RobustInventoryManagement")

    # Decision variables
    x = model.addVars(T, lb=0, name="x")  # Inventory levels
    y = model.addVars(T, lb=0, name="y")  # Auxiliary variables for worst-case cost

    # Objective function: minimize sum of worst-case costs
    model.setObjective(gp.quicksum(y[t] for t in range(T)), GRB.MINIMIZE)

    # Constraints for worst-case cost calculation
    # for t in range(T):
    #     model.addConstr((y[t] >= u_t[t] - x[t] for t in range(T)), name="upper_bound_cost")
    #     model.addConstr((y[t] >= x[t] - l_t[t] for t in range(T)), name="lower_bound_cost")

    model.addConstrs((y[t] >= u_t[t] - x[t] for t in range(T)), name="upper_bound_cost")
    model.addConstrs((y[t] >= x[t] - l_t[t] for t in range(T)), name="lower_bound_cost")

    # Solve the model
    model.optimize()

    # Check solution status
    if model.status == GRB.OPTIMAL:
        print("Optimal solution found:")
    else:
        print("No optimal solution found")

    # Extract solution
    solution = [x[t].X for t in range(T)]

    # Evaluation
    real_objective_value = 0
    evaluation_results = []

    for t in range(len(real_data)):
        inventory = solution[t]
        demand = real_data[t]
        loss = abs(inventory - demand)
        evaluation_results.append(loss)
        real_objective_value += loss
    ```
    """

    TASK = """
    We already have calculated the box uncertainty set boundary, where {lower} and {upper} represent the lower and upper bounds of box uncertainty respectively.
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

    def run(self, description: str, data_description: str, row: int, lower: list, upper: list, initial_model: str, transfered_model: str, train_dir: str, test_dir: str, result_dir: str) -> str:
        return self.chain.invoke({"description" : description, "data_description" : data_description, "row": str(row), "lower": lower, "upper": upper, "initial_model": initial_model, "transfered_model": transfered_model, "train_dir" : train_dir, "test_dir" : test_dir, "result_dir" : result_dir})

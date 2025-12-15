from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from utils import MYLLM

class Code_Repair(MYLLM):


    SYSTEM = """You are an expert in Python, good at debugging and fixing code errors, especially familiar with gurobipy, Pandas, Numpy and file processing.
    Below is part (or all) of the Python program I am running, and it has an error during the operation.
    Please help me complete the following tasks:

    1. Analyze the error message and find out the root cause of the error;
    2. Explain the principle of the problem and the solution;
    3. Give the modified complete code and ensure that it can run normally;
    """

    INPUT = """```python
    from gurobipy import *

    model = Model("example")
    x = model.addVar(name='x', lb=0)
    model.setObjective(abs(x - 10), GRB.MINIMIZE)
    model.optimize()
    ```

    TypeError: float() argument must be a string or a number, not 'GenExpr'
   
    """

    OUTPUT = """## Fix analysis

    **Error reason**:

    `abs(x - 10)` is Python's built-in function `abs()`, which is not applicable to Gurobi expressions (`GenExpr` type).
    You have to model the absolute value as a linear expression, introduce a new auxiliary variable `z`, and then add constraints:

    ```
    z >= x - 10
    z >= 10 - x
    ```

    ---

    ## Fixed code (Gurobi linearized absolute value modeling)

    ```python
    from gurobipy import *

    # Create model
    model = Model("abs_example")
    model.setParam("OutputFlag", 0) # Turn off output

    # Define variables
    x = model.addVar(name='x', lb=0)
    z = model.addVar(name='z', lb=0) # Absolute value deviation variable

    # Add absolute value linearization constraints
    model.addConstr(z >= x - 10)
    model.addConstr(z >= 10 - x)

    # Set objective function
    model.setObjective(z, GRB.MINIMIZE)

    # Solution
    model.optimize()

    # Output results
    print(f"Optimal x = {{x.X:.2f}}")
    print(f"Absolute deviation = {{z.X:.2f}}")
    ``
    """

    TASK = """```python
    {code}
    ```

    {error}
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

    def run(self, code: str, error: str) -> str:
        return self.chain.invoke({"code": code, "error": error})

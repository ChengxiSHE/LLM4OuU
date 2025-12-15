import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from utils import MYLLM 

class Optimization_Modeling(MYLLM):

    SYSTEM = """You are a Python programmer in the field of operations research and optimization. Your proficiency in utilizing third-party libraries such as Gurobi is essential. In addition to your expertise in Gurobi, it would be great if you could also provide some background in related libraries or tools, like NumPy, SciPy, PuLP, or Mosek.
    You are now facing a optimization problem under uncertainty. You aim to develop an efficient Python program based on {method} that addresses the given problem.
    Save the solution, actual value, corresponding decision loss for each decision step, and the real objective value as a text file format in {result_dir}.
    Ensure the data loading path in your code is consistent with the provided {train_dir} and {test_dir}. Please strictly import empirical data from {train_dir} and real test data from {test_dir} , and do not fabricate data yourself. Additionally, strictly use “/” as the file path separator.
    Please think step by step and provide the complete Python code to solve the problem.
    Please give the complete code, including the code using ```python```.
    """

    TASK = """
    Optimization problem under uncertainty: 
    {description}
    Empirical data description is as follows:
    {data_description}
    """
        
    def __init__(self):
        super().__init__()
        prompt = ChatPromptTemplate([
            ("system",self.SYSTEM), 
            ("human", self.TASK),
        ])
        self.chain = prompt | self.llm | StrOutputParser()

    def run(self, description: str, data_description: str, train_dir: str, test_dir: str, result_dir: str, method: str) -> str:
        return self.chain.invoke({"description" : description, "data_description" : data_description, "train_dir" : train_dir, "test_dir" : test_dir, "result_dir" : result_dir, "method": method})
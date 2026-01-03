This repository contains the official implementation for paper "Crossroads of Optimization under Uncertainty: How to Choose the Optimal Model"

#### Notice
We provide a first-order moment method here, but it is not mentioned in the paper. This is because the first-order moment method can be transformed into box RO.

#### Installation
```
pip install -r requirements.txt
```

#### Set API_KEY
Set the environment variable: you need to change your OPENAI_API_KEY in config.py

#### Run the experiment
<main.py> 
you can change the "method", "data_dir" and "model" respectively. Please remember to update config.py file and agent file when modifying the mod<img width="3982" height="1980" alt="workflow" src="https://github.com/user-attachments/assets/47c061fd-2067-4195-94c7-41942b243ddc" />
el to meet the format requirements of different APIs.

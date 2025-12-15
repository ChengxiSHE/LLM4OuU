## CI-SAA

### CISAA is an automatic modeling method for contextual optimization using LLM

#### Installation

```bash
conda create -n cisaa python=3.10
conda activate cisaa
pip install -r requirements.txt
```

#### Set API_KEY

```bash
export OPENAI_API_KEY=<your openai api key>
```

#### Run the experiment

```
python main.py --model deepseek-reasoner --method saa

python main.py --model deepseek-reasoner --method cisaa
```

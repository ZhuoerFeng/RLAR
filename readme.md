# RLAR: An Agentic Reward System for Multi-task Reinforcement Learning on Large Language Models

**ICLR 2025 Anonymous Submission**

---

## 📖 Overview

**RLAR** (An Agentic Reward System for Multi-task Reinforcement Learning on Large Language Models) introduces a novel reward design to train LLMs on a variety of tasks via reinforcement learning.  


This repository contains:

1. **RLAR methodology implementation** (entry point: `pipeline.py`)  
2. **Modified VERL framework** for RL training  
3. **Data preparation instructions** with download links for the training set

---

## 🏗️ Repository Structure

```
.
├── code/
│   ├── RLAR/
│   │   ├── glm_api_request/
│   │   │   └── model.py          # GLM model API request handling
│   │   ├── agent_prompts.py      # Agent prompt templates
│   │   └── pipeline.py           # Main RLAR pipeline (entry point)
│   └── verl/                     # VERL RL framework (with modifications)
│
├── data/
│   └── download.md   # Training dataset download URL
│
├── .gitignore
└── readme.md
```

---

## 🚀 Installation

**Prerequisites**  
- Python = 3.10.12
- [PyTorch](https://pytorch.org/) = 2.6.0
- vllm = 0.8.5
- CUDA toolkit (if training on GPU) version 12.6
- Other dependencies (see `requirements.txt`) 

**Setup**
```bash
git clone https://github.com/ICLR_2025_RLAR.git
cd ICLR_2025_RLAR
pip install -r requirements.txt
```

---

## 📂 Dataset

The training dataset is available via:

```bash
cd data

wget https://drive.google.com/file/d/1klpKjWjZMTbROnqp9Yg_dBFVCwWdEd7t/view?usp=sharing

wget https://drive.google.com/file/d/1M--Ik7tHcH_wm6uz3SVxulR2EiLtj5BS/view?usp=sharing
```

You will get ```train.parquet```, ```valid.parquet```, which suits for `verl` framework training.

---

## ⚙️ Usage

### 1. **Run RLAR Pipeline**

Visit `verl/examples/grpo` to check the training scripts
```bash
cd code/RLAR
python pipeline.py --config <config_file>
```
**Arguments:**
- `--config`: Path to config JSON/YAML specifying tasks, model, RLAR parameters

---

### 2. **Integrating with VERL**
Your modified VERL implementation can be run from:
```bash
cd code/verl/example/grpo;

sh run_rlar_8b.sh;
```

---

### 3. **Model API Requests**
`glm_api_request/model.py` contains the utilities for calling OpenAI SDK-based models.
```python
from glm_api_request import model

model = GateWays(model_name='gpt-4.1')

model.get_api_result(messages=[...])
```

---

## 📜 Methodology

---

## 📊 Experimental Results

---

## 📝 Citation

If you find this work useful, please cite:

```
@inproceedings{yourname2025rlar,
  title={RLAR: An Agentic Reward System for Multi-task Reinforcement Learning on Large Language Models},
  author={Annonymous Authors},
  booktitle={International Conference on Learning Representations Submission},
  year={2025}
}
```

---

## 📄 License


```
MIT License

Copyright (c) [2025] [anonymous author]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```


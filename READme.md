# SynLLM: A Comparative Analysis of Large Language Models for Medical Synthetic Data Generation

The generation of synthetic medical data is crucial for advancing healthcare research, particularly when privacy regulations restrict access to real patient data. This paper presents SynLLM, a framework for generating and evaluating synthetic medical data using different Large Language Models (LLMs). We conduct a comprehensive comparative analysis of open-source models, including GPT-2, LLaMA, Mistral models, etc. examining their capabilities in maintaining statistical fidelity, preserving privacy, and ensuring medical consistency. Our evaluation framework incorporates multiple dimensions: numerical accuracy, categorical similarity, feature correlations, medical relationship preservation, and privacy guarantees. The results reveal distinct trade-offs between models: GPT-2 achieves superior statistical accuracy (8.79\% relative mean error in key metrics) and distribution preservation, Mistral demonstrates balanced performance with strong privacy guarantees (0.2-0.4\% violation rates), while LLaMA exhibits perfect privacy preservation but with reduced statistical fidelity. We introduce novel prompt engineering strategies and automated quality assessment metrics, particularly focusing on data quality, medical consistency, and feature correlation preservation. Our findings indicate that model selection should be use-case dependent, with GPT-2 suited for statistical accuracy requirements, Mistral for balanced applications, and LLaMA for privacy-critical scenarios. This work provides a foundation for automatic prompt generation, systematic evaluation, and selection of LLMs in medical data synthesis, contributing to the development of more reliable and privacy-preserving synthetic data generation methods.

## Installation

```bash
# Clone the repository
git clone https://github.com/arshiailaty/SynLLM.git
cd SynLLM

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

SynLLM/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   └── __init__.py
│   ├── models/
│   │   └── __init__.py
│   └── utils/
│       └── __init__.py
├── tests/
│   └── __init__.py
├── configs/
│   └── config.yaml
├── notebooks/
│   └── examples.ipynb
├── requirements.txt
├── setup.py
├── .gitignore
└── README.md

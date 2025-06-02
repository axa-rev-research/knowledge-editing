# SAKE: Steering Activations for Knowledge Editing

SAKE (Steering Activations for Knowledge Editing) is a novel method for LLM knowledge editing that models facts as distributions rather than single prompts. Using Optimal Transport theory, SAKE alters LLM behavior over whole fact-related distributions, including paraphrases and logical implications, enabling more robust and generalizable knowledge edits.

## Key Innovations

Unlike other Knowledge Editing (KE) methods that optimize for individual prompts, SAKE addresses three major limitations of existing approaches:

1. **Logical Implications**: Better generalization to various types of logical implications related to edited facts
2. **Contextual Robustness**: Improved performance on realistic conversational contexts, including long/noisy prompts and doubt-raising scenarios  
3. **Flexible Editing**: More efficient revision and removal of prior edits through activation steering

## Features

- **Distribution-Based Editing**: Models facts as distributions of paraphrases and implications rather than single prompts
- **Optimal Transport**: Uses optimal transport mappings to transform model representations between source and target distributions
- **Threshold Mechanisms**: Edit criterion based on hidden state or prompt distance 
- **Multiple Model Support**: Compatible with various transformer architectures (GPT-2, Llama, etc.)
- **Evaluation Metrics**: Testing for accuracy, generalization, and specificity (CounterFact), logical implications (RippleEdits Popular), contextual robustness (Doubts)

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/knowledge-editing.git
cd knowledge-editing
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up Hugging Face authentication:**
- You'll need a Hugging Face account for model access
- Distribution modeling requires OpenAI or Anthropic API keys (or other providers of your choice, modifying the sentence generation functions)

## Quick Start

### Running Examples

The repository includes comprehensive Jupyter notebook examples:

1. **CounterFact Dataset Example:**
```bash
jupyter notebook examples/SAKE-gpt2xl-cf.ipynb
```

2. **Popular Dataset Example:**
```bash
jupyter notebook examples/SAKE-gpt2xl-pop.ipynb
```

3. **Raising Doubts Example:**
```bash
jupyter notebook examples/SAKE-gpt2xl-doubts.ipynb
```

## Project Structure

```
knowledge-editing/
├── SAKE/                    # Core SAKE implementation
│   ├── edit.py             # Optimal transport mapping functions
│   ├── threshold.py        # Threshold mechanisms for edit activation
│   └── distributions.py    # Data generation and representation extraction
├── examples/               # Example notebooks
│   ├── SAKE-gpt2xl-cf.ipynb    # CounterFact dataset example
│   ├── SAKE-gpt2xl-pop.ipynb   # Popular dataset example  
│   └── SAKE-gpt2xl-doubts.ipynb # Doubts dataset example
├── data/                   # Data storage
│   ├── cf/                 # CounterFact dataset files
│   └── ripple/             # Additional datasets
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Core Components

### 1. Knowledge Editing (`SAKE/edit.py`)
- `learn_mappings_counterfact()`: Learn optimal transport mappings between source and target distributions
- Supports various optimal transport algorithms with configurable regularization

### 2. Threshold Mechanisms (`SAKE/threshold.py`)
- `hid_threshold()`: Hidden state-based edit activation
- `prompt_threshold()`: Prompt similarity-based edit activation
- Support for cosine similarity, Euclidean, and Mahalanobis distances

### 3. Data Processing (`SAKE/distributions.py`)
- `generate_paraphrases_counterfact()`: Generate paraphrases using LLM APIs
- `extract_representations_counterfact()`: Extract model representations
- `compute_means()`: Compute mean embeddings for threshold mechanisms

## Usage Workflow

### 1. Data Preparation
```python
# Generate paraphrases (requires API key)
cf = generate_paraphrases_counterfact(
    cf=cf, 
    provider="anthropic",  # or "openai"
    api_key="your_api_key",
    model="claude-3-5-sonnet-latest",
    indexes=(0, 10)
)
```

### 2. Model Loading
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = 'gpt2-xl'
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    output_hidden_states=True
).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

### 3. Representation Extraction
```python
cf = extract_representations_counterfact(
    model, tokenizer, cf, 
    indexes=(0, 10), 
    device='cuda'
)
```

### 4. Learning Mappings
```python
maps = learn_mappings_counterfact(cf, indexes=(0, 10))
```

### 5. Evaluation
```python
# Compute embeddings for threshold mechanism
source_mean_embs, target_mean_embs = compute_means(cf, indexes=(0, 10))

# Test accuracy, generalization, and specificity
# (See example notebooks for complete evaluation code)
```

## Configuration

### Device Support
- **CUDA**: Automatic GPU detection for NVIDIA cards
- **MPS**: Apple Silicon GPU support for M1/M2 Macs
- **CPU**: Fallback for systems without GPU acceleration

### Threshold Parameters
- Adjust thresholds for edit activation sensitivity
- Configure distance metrics (euclidean, cosine, mahalanobis)
- Set regularization parameters for optimal transport

### API Integration
- **OpenAI**: GPT models for paraphrase generation
- **Anthropic**: Claude models for paraphrase generation
- **Hugging Face**: Model loading and tokenization

## Requirements

- Python 3.8+
- PyTorch 1.12+
- Transformers 4.20+
- Additional dependencies listed in `requirements.txt`

## Data

The repository includes example data in the `data/` directory:
- **CounterFact**: Factual knowledge editing dataset
- **Popular**: Knowledge editing for popular entities
- Pre-computed embeddings for quick testing

## Authors

- **Marco Scialanga**\* - EPFL, Lausanne (worked done when at AXA, Paris) 
- **Thibault Laugel**\* - AXA, Paris & TRAIL, LIP6, Sorbonne Université  
- **Vincent Grari** - AXA, Paris & TRAIL, LIP6, Sorbonne Université
- **Marcin Detyniecki** - AXA, Paris & TRAIL, LIP6, Sorbonne Université & Polish Academy of Science

\*Equal contribution

Correspondence: marco.scialanga@epfl.ch, thibault.laugel@axa.com

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use SAKE in your research, please cite the original paper:

```bibtex
@article{scialanga2025sake,
  title={SAKE: Steering Activations for Knowledge Editing},
  author={Scialanga, Marco and Laugel, Thibault and Grari, Vincent and Detyniecki, Marcin},
  journal={arXiv preprint arXiv:2503.01751},
  year={2025},
  url={https://arxiv.org/abs/2503.01751}
}
```

## Support

- **Issues**: Report bugs and feature requests via GitHub Issues
- **Documentation**: See example notebooks for detailed usage
- **Community**: Discussions welcome in GitHub Discussions

## Acknowledgments

- Built with PyTorch and Hugging Face Transformers
- Uses Python Optimal Transport (POT) library
- Research conducted at AXA, EPFL, TRAIL/LIP6 Sorbonne Université, and Polish Academy of Science
- Paper: [SAKE: Steering Activations for Knowledge Editing](https://arxiv.org/abs/2503.01751)

## Method Overview

SAKE operates through three main phases:

### 1. Distribution Modeling
```python
# Generate paraphrases and logical implications
cf = generate_paraphrases_counterfact(
    cf=cf, 
    provider="anthropic",  # or "openai"
    api_key="your_api_key",
    model="claude-3-5-sonnet-latest",
    indexes=(0, 10)
)
```

### 2. Optimal Transport Mapping
```python
# Learn mappings between source and target distributions
maps = learn_mappings_counterfact(cf, indexes=(0, 10))
```

### 3. Threshold-Based Activation
```python
# Apply edits based on prompt/hidden state similarity
prompt_sim, prompt_idx = prompt_threshold(
    prompt_enc, prompt_embs, threshold=6.5, dist_type="euc"
)
if prompt_idx is not None:
    # Apply learned mapping
    edited_hidden_state = maps[edit_idx].transform(Xs=hidden_state)
```

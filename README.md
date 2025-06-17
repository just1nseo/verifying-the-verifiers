# Verifying the Verifiers: Unveiling Pitfalls and Potentials in Fact Verifiers

<p align="center">
  <a href="https://arxiv.org/abs/2506.13342">
    <img src="https://img.shields.io/badge/📝-Paper-blue">
  </a>
  <a href="https://huggingface.co/datasets/just1nseo/ClearFacts">
    <img src="https://img.shields.io/badge/🤗-ClearFacts-white">
  </a>
  <a href="https://huggingface.co/datasets/just1nseo/GrayFacts">
    <img src="https://img.shields.io/badge/🤗-GrayFacts-gray">
  </a>
  <a href="https://huggingface.co/just1nseo/ClearCheck-8B">
    <img src="https://img.shields.io/badge/🤗-Model-orange">
  </a>
</p>

**Authors:**
[Wooseok Seo](https://just1nseo.github.io/) ⭐,
[Seungju Han](https://seungjuhan.me) ⭐,
[Jaehun Jung](https://jaehunjung.com/),
[Benjamin Newman](https://bnewm0609.github.io/),
[Seungwon Lim](https://sngwonlim.github.io/),
[Seungbeen Lee](https://seunbite.github.io/),
[Ximing Lu](https://gloriaximinglu.github.io/),
[Yejin Choi](https://yejinc.github.io/),
[Youngjae Yu](https://yj-yu.github.io/home/)

⭐ Co-first authors

We provide an evaluation framework for running [ClearFacts](https://huggingface.co/datasets/just1nseo/ClearFacts) and [GrayFacts](https://huggingface.co/datasets/just1nseo/GrayFacts). We provide both vLLM and API deployment. You can also add your custom dataset for evaluation.

## Installation

```bash
pip install -r requirements.txt 
```

We recommend using Python 3.10.

## Key Features

### Datasets
- **`CLEARFACTS`**: Refined Fact Verification Dataset (S/NS labels)
  - **Key Metrics**: Macro F1
  - Evaluates overall model performance
- **`GRAYFACTS`**: Ambiguous Fact Verification Dataset (only AMBIG label)
  - **Key Metrics**: Predicted Supported %
  - Shows prediction distribution for ambiguous examples

### Label System
- **Model Outputs**: Models return "Attributable", "Not Attributable", "Contradictory"
- **Internal Mapping**: 
  - "Attributable" → "S" (Supported)
  - "Not Attributable" → "NS" (Not Supported)  
  - "Contradictory" → "NS" (Not Supported)

## Supported Models

We support a total of six model classes. We provide API support for OpenAI models, Anthropic models, and Gemini models. We also provide open model supports via vLLM.

- API Models
  - `APIFactAttributorFewShot`: API-based few-shot attribution
  - `APIFactAttributorZeroShot`: API-based zero-shot attribution
- Open Models 
  - `OpenFactAttributorFewShot`: Open model few-shot attribution
  - `OpenFactAttributorZeroShot`: Open model zero-shot attribution
- [ClearCheck](https://huggingface.co/just1nseo/ClearCheck-8B)
  - `ClearCheckDirect`: ClearCheck direct attribution
  - `ClearCheckCoT`: ClearCheck reasoning attribution

## Usage

1. For API models, set up keys in `common/shared_config.py`:
```python
OPENAI_API_KEY = 'your_openai_key'
GEMINI_API_KEY = 'your_gemini_key'
ANTHROPIC_API_KEY = 'your_anthropic_key'
```

You can also add additional API models in `common/shared_config.py`.

2. Place datasets in `data/` folder:
   - `data/clearfacts.jsonl`
   - `data/grayfacts.jsonl`
   - `data/[your_custom_dataset].jsonl`

### API Model Usage Examples

1. Running CLEARFACTS with Few-shot API model.
```bash
python evaluation/eval.py run clearfacts APIFactAttributorFewShot gpt_4o
```

2. Running GRAYFACTS with Zero-shot API model and multi-processing (16 threads). 
```bash
python evaluation/eval.py run grayfacts APIFactAttributorZeroShot gemini_2_5_pro 16
```

### Open Model Usage Examples

For open models, you can shard both the _model_ and _data_ for efficient inference. 

Sharding a large _model_ to run CLEARFACTS on 4 GPUs. vLLM will automatically distribute model weights to 4 GPUs.
```bash
python evaluation/eval.py run grayfacts OpenFactAttributorZeroShot Qwen/Qwen3-235B-A22B 4
```

Sharding the _data_ for faster results. `run_multigpu` will automatically shard the dataset into 4 shards and run on individual GPUs. Results are automatically merged. 
```bash
python evaluation/eval.py run_multigpu 4 clearfacts OpenFactAttributorFewShot meta-llama/Llama-3.1-8B-Instruct 
```

You can shard both _model_ and _data_ at the same time. If you have 8 GPUs but have to shard the model on 2 GPUs, you can run like this.
```bash
CUDA_VISIBLE_DEVICES=0,1 python evaluation/eval.py run_shard clearfacts 4 0 OpenFactAttributorFewShot Qwen/Qwen2.5-72B-Instruct
CUDA_VISIBLE_DEVICES=2,3 python evaluation/eval.py run_shard clearfacts 4 1 OpenFactAttributorFewShot Qwen/Qwen2.5-72B-Instruct
CUDA_VISIBLE_DEVICES=4,5 python evaluation/eval.py run_shard clearfacts 4 2 OpenFactAttributorFewShot Qwen/Qwen2.5-72B-Instruct
CUDA_VISIBLE_DEVICES=6,7 python evaluation/eval.py run_shard clearfacts 4 3 OpenFactAttributorFewShot Qwen/Qwen2.5-72B-Instruct
```

You can run [ClearCheck](https://huggingface.co/just1nseo/ClearCheck-8B) as well.
```bash
python evaluation/eval.py run clearfacts ClearCheckDirect just1nseo/ClearCheck-8B
python evaluation/eval.py run grayfacts ClearCheckCoT just1nseo/ClearCheck-8B
```

### Additional topic-wise/category-wise analysis

```bash
python evaluation/eval.py run clearfacts APIFactAttributorZeroShot gpt_4o --topic_wise=True --category_wise=True
```

### Example Output

**For ClearFacts:**
```
Dataset Type: Binary Fact Verification (clearfacts)
Key Metrics for Binary Evaluation:
Macro F1: 0.847
Balanced Accuracy: 0.845
Model Output Attribution Breakdown:
Predicted 'Attributable': 127
Predicted 'Not Attributable': 98
Predicted 'Contradictory': 25
```

**For GrayFacts:**
```
Dataset Type: Ambiguous Fact Verification (grayfacts)
--- Normal Examples (S/NS) Metrics ---
Macro F1: 0.782
--- Ambiguous Examples Prediction Distribution ---
Predicted as Supported: 45 (63.4%)
Predicted as Not Supported: 26 (36.6%)
Key Metrics for Ambiguous Evaluation:
Normal Set Macro F1: 0.782
Ambiguous Supported %: 63.4%
```

## Dataset Format

```json
{
  "topic": "Source dataset",
  "statement": "Fact to be verified",
  "reference_documents": ["doc1", "doc2"],
  "label": "S|NS|AMBIG",
  "category": "Specific source dataset",
  "additional_info": "ex: contains original label for GRAYFACTS"
}
```

## Project Structure

```
├── evaluation/
│   ├── eval.py              # Main evaluation script with unified reporting
│   ├── factory.py           # Model and dataset loading
│   └── tasks/
│       ├── __init__.py      # Unified FactVerificationLabel system
│       └── dataset.py       # Dataset loaders for clearfacts/grayfacts
├── models/
│   ├── __init__.py          # Unified FactVerificationResult class
│   └── attributor.py        # All 6 attribution model implementations
├── common/
│   └── shared_config.py     # API keys and model configurations
├── data/                    # Dataset files
│   ├── clearfacts.jsonl     # Contains CLEARFACTS dataset
└── └── grayfacts.jsonl      # Contains GRAYFACTS dataset
```

## Reference
Our code is built upon this codebase: https://github.com/google-deepmind/long-form-factuality.git. 

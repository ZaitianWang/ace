# Agentic Context Engineering (ACE)

This repository provides an automated system for optimizing prompts on your custom tasks, improving model performance through iterative refinement.

> **⚠️ Important:**  
> If you are trying to run experiments and reproduce results in the ACE paper, please go to ```paper-artifact/``` for detailed instructions.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up API Keys

Copy over `.env.example` to a `.env` file and edit `.env` with your actual API keys

```bash
cp .env.example .env
```

You can use one or multiple API keys (comma-separated).

### 3. Prepare Your Data

Create three JSONL files (one example per line):

**train.jsonl:**
```json
{"question": "What is the capital of France?", "context": "France is in Europe...", "answer": "Paris"}
{"question": "Who wrote Hamlet?", "context": "Shakespeare was...", "answer": "Shakespeare"}
```

**val.jsonl and test.jsonl:** Same format as train.jsonl

### 4. Create Your Task Configuration

Create a `config.json` file:

```json
{
    "task_config": {
        "task_name": "my_task",
        "base_instruction": "Answer the question based on the provided context.",
        "eval_metric": "exact_match",
        "data_config": [
            {
                "name": "question",
                "type": "input",
                "description": "The question to answer"
            },
            {
                "name": "context",
                "type": "input",
                "description": "Background information"
            },
            {
                "name": "answer",
                "type": "output",
                "description": "The correct answer"
            }
        ]
    },
    "train_data": "./data/train.jsonl",
    "val_data": "./data/val.jsonl",
    "test_data": "./data/test.jsonl"
}
```

**Important:** Field names in `data_config` must exactly match the keys in your JSONL files.

### 5. Run Optimization

```bash
python optimize.py \
    --task_config_path ./config.json \
    --save_path ./results \
    --num_train_samples 100 \
    --num_val_samples 50
```

## Configuration Guide

### Task Config Fields

- **task_name**: Identifier for your task (used in result folder names)
- **base_instruction**: Instructions describing what the model should do
- **eval_metric**: How to evaluate predictions
  - `exact_match`: Predictions must match ground truth exactly
  - `contains`: Ground truth must be contained in prediction
- **data_config**: Define your input and output fields
  - `name`: Field name (must match JSONL keys)
  - `type`: Either `"input"` or `"output"`
  - `description`: Optional description of the field

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--task_config_path` | Path to your config file | `./sample_data/sample_config.json` |
| `--save_path` | Where to save results | `./results` |
| `--num_train_samples` | Number of training examples (-1 = all) | -1 |
| `--num_val_samples` | Number of validation examples (-1 = all) | -1 |
| `--generator_model` | Model for generation | `sambanova/DeepSeek-V3.1` |
| `--optimizer_model` | Model for optimization | `sambanova/DeepSeek-V3.1` |
| `--reflection_model` | Model for reflection | `sambanova/DeepSeek-V3.1` |
| `--generator_max_tokens` | Max tokens to generate for generator model | 4096 |
| `--auto` | Optimization intensity (`light`, `medium`, `heavy`) | `heavy` |

## Example Use Cases

### Classification Task

```json
{
    "task_config": {
        "task_name": "sentiment_classification",
        "base_instruction": "Classify the sentiment of the text as positive, negative, or neutral.",
        "eval_metric": "exact_match",
        "data_config": [
            {"name": "text", "type": "input"},
            {"name": "sentiment", "type": "output"}
        ]
    },
    "train_data": "./data/sentiment_train.jsonl",
    "val_data": "./data/sentiment_val.jsonl",
    "test_data": "./data/sentiment_test.jsonl"
}
```

**Data format:**
```json
{"text": "I love this product!", "sentiment": "positive"}
```

### Summarization Task

```json
{
    "task_config": {
        "task_name": "summarization",
        "base_instruction": "Generate a concise summary of the document.",
        "eval_metric": "contains",
        "data_config": [
            {"name": "document", "type": "input"},
            {"name": "summary", "type": "output"}
        ]
    },
    "train_data": "./data/docs_train.jsonl",
    "val_data": "./data/docs_val.jsonl",
    "test_data": "./data/docs_test.jsonl"
}
```

**Data format:**
```json
{"document": "Long article text...", "summary": "Brief summary..."}
```

## Output

Results are saved to `{save_path}/{task_name}_gepa_{auto}/`:

- **results.json**: Initial and final accuracy metrics
- **gepa_optimized_pipeline.json**: Optimized pipeline (reusable)
- **training_log_*.log**: Complete training logs
- **stdout_*.log** and **stderr_*.log**: Console output

## Tips

1. **Start small**: Test with `--num_train_samples 10` first to verify everything works
2. **Separate files**: Use different files for train/val/test when possible
3. **Field names**: Ensure JSONL field names exactly match your config (case-sensitive)
4. **Evaluation metric**: Choose based on your task:
   - Use `exact_match` for classification or short answers
   - Use `contains` for generation tasks where partial matches are acceptable

## Troubleshooting

**"Field 'X' not found in data"**
- Check that field names in config match JSONL keys exactly

**"SAMBANOVA_API_KEYS not found"**
- Create a `.env` file with your API keys

**Low accuracy**
- Try increasing training samples
- Adjust `base_instruction` to be more specific
- Use `--auto heavy` for more intensive optimization

## Support

For issues or questions, please check that:
1. Your JSONL files are valid (one JSON object per line)
2. All required fields are present in your data
3. API keys are properly configured in `.env`
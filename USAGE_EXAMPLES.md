# LLM Complete Toolkit - Usage Examples

## 📖 Comprehensive Usage Guide

This document provides detailed usage examples for all components of the LLM Complete Toolkit.

---

## 🚀 Quick Start Examples

### 1. Complete Workflow Example
```bash
# Step 1: Create sample data
python main.py create-samples --output-dir data/

# Step 2: Extract documents to training data
python main.py extract data/ outputs/ --format jsonl --instruction-format

# Step 3: Train with LoRA
python main.py train-lora --train-data outputs/instruction_data.jsonl --epochs 3

# Step 4: Train with reinforcement learning
python main.py train-rl --algorithm ppo --episodes 500
```

### 2. Basic Document Processing
```bash
# Extract PDF and Markdown files
python main.py extract documents/ training_data/ --format jsonl

# With custom chunk size
python main.py extract documents/ training_data/ --chunk-size 1024

# Generate instruction format
python main.py extract documents/ training_data/ --instruction-format
```

---

## 📄 Document Processing Examples

### PDF Processing
```bash
# Basic PDF extraction
python main.py extract pdf_folder/ output/ --format jsonl

# Process with specific chunk size
python main.py extract pdf_folder/ output/ --chunk-size 512

# Generate both formats
python main.py extract pdf_folder/ output/ --format jsonl --instruction-format
```

**Example Input**: `document.pdf`
**Example Output**: `training_data.jsonl`
```json
{
  "text": "This is the extracted text from the PDF document...",
  "source": "document.pdf",
  "title": "Document Title",
  "type": "pdf",
  "chunk_id": 1,
  "total_chunks": 5,
  "metadata": {
    "page_count": 10,
    "file_size": 1024000
  }
}
```

### Markdown Processing
```bash
# Process Markdown files
python main.py extract markdown_folder/ output/ --format jsonl

# With instruction format
python main.py extract markdown_folder/ output/ --instruction-format
```

**Example Input**: `guide.md`
```markdown
---
title: "AI Guide"
author: "Expert"
---

# Introduction
This is an AI guide...

## Chapter 1
Content here...
```

**Example Output**: `training_data.jsonl`
```json
{
  "text": "This is an AI guide...",
  "source": "guide.md",
  "title": "AI Guide",
  "type": "markdown",
  "chunk_id": 1,
  "total_chunks": 3,
  "metadata": {
    "frontmatter": {
      "title": "AI Guide",
      "author": "Expert"
    },
    "section_count": 2
  }
}
```

### Mixed Document Processing
```bash
# Process multiple document types
python main.py extract mixed_documents/ output/ --format jsonl --instruction-format

# Directory structure:
# mixed_documents/
# ├── report.pdf
# ├── guide.md
# └── notes.txt
```

---

## 🧠 Transfer Learning Examples

### LoRA Fine-tuning

#### Basic LoRA Training
```bash
# Simple LoRA training
python main.py train-lora --train-data data/train.jsonl

# With evaluation data
python main.py train-lora \
  --train-data data/train.jsonl \
  --eval-data data/eval.jsonl

# Custom model and parameters
python main.py train-lora \
  --train-data data/train.jsonl \
  --model-name microsoft/DialoGPT-medium \
  --epochs 5 \
  --batch-size 8 \
  --learning-rate 1e-4
```

#### Advanced LoRA Configuration
```bash
# Using configuration file
python main.py train-lora \
  --train-data data/train.jsonl \
  --config configs/lora_config.yaml \
  --output-dir outputs/lora_experiment

# With verbose logging
python main.py train-lora \
  --train-data data/train.jsonl \
  --verbose
```

**Example Training Data**: `train.jsonl`
```json
{"instruction": "Explain machine learning", "input": "", "output": "Machine learning is a subset of artificial intelligence..."}
{"instruction": "What is deep learning?", "input": "", "output": "Deep learning is a subset of machine learning..."}
{"instruction": "Define neural networks", "input": "", "output": "Neural networks are computing systems inspired by biological neural networks..."}
```

### QLoRA Fine-tuning

#### Memory-Efficient Training
```bash
# Basic QLoRA training (low memory)
python main.py train-qlora --train-data data/train.jsonl

# Large model with QLoRA
python main.py train-qlora \
  --train-data data/train.jsonl \
  --model-name microsoft/DialoGPT-large \
  --batch-size 1 \
  --epochs 2

# With gradient accumulation
python main.py train-qlora \
  --train-data data/train.jsonl \
  --batch-size 1 \
  --learning-rate 2e-4
```

#### QLoRA with Custom Settings
```bash
# Custom QLoRA configuration
python main.py train-qlora \
  --train-data data/train.jsonl \
  --model-name microsoft/DialoGPT-medium \
  --epochs 3 \
  --batch-size 1 \
  --output-dir outputs/qlora_custom \
  --verbose
```

---

## 🎯 Reinforcement Learning Examples

### PPO Training

#### Basic PPO Training
```bash
# Simple PPO training
python main.py train-rl --algorithm ppo

# With custom episodes
python main.py train-rl \
  --algorithm ppo \
  --episodes 1000

# With configuration
python main.py train-rl \
  --algorithm ppo \
  --config configs/ppo_config.yaml \
  --output-dir outputs/ppo_experiment
```

#### Advanced PPO Configuration
```bash
# PPO with custom settings
python main.py train-rl \
  --algorithm ppo \
  --episodes 2000 \
  --config configs/config.yaml \
  --verbose
```

### DQN Training

#### Basic DQN Training
```bash
# Simple DQN training
python main.py train-rl --algorithm dqn

# With custom episodes
python main.py train-rl \
  --algorithm dqn \
  --episodes 1500

# With output directory
python main.py train-rl \
  --algorithm dqn \
  --output-dir outputs/dqn_experiment
```

#### DQN with Custom Parameters
```bash
# DQN with verbose logging
python main.py train-rl \
  --algorithm dqn \
  --episodes 1000 \
  --verbose \
  --output-dir outputs/dqn_detailed
```

---

## ⚙️ Configuration Examples

### Custom Configuration File
Create `configs/my_config.yaml`:
```yaml
common:
  device: "cuda"
  seed: 123
  log_level: "DEBUG"
  
transfer_learning:
  lora:
    model_name: "microsoft/DialoGPT-small"
    lora_r: 8
    lora_alpha: 16
    training:
      num_train_epochs: 5
      per_device_train_batch_size: 2
      learning_rate: 1e-4
      
reinforcement_learning:
  ppo:
    learning_rate: 1e-4
    training:
      max_episodes: 2000
      
logging:
  use_tensorboard: true
  use_wandb: true
  wandb_project: "my-llm-project"
```

Use custom configuration:
```bash
python main.py train-lora \
  --train-data data/train.jsonl \
  --config configs/my_config.yaml
```

---

## 📊 Data Preparation Examples

### Creating Training Data

#### From Extracted Documents
```bash
# Extract documents with instruction format
python main.py extract documents/ data/ --instruction-format

# This creates:
# data/training_data.jsonl      # Basic format
# data/instruction_data.jsonl   # Instruction format
```

#### Manual Data Creation
Create `data/custom_train.jsonl`:
```json
{"instruction": "Summarize this text", "input": "Long text here...", "output": "Summary here..."}
{"instruction": "Translate to French", "input": "Hello world", "output": "Bonjour le monde"}
{"instruction": "Explain the concept", "input": "Machine Learning", "output": "Machine learning is..."}
```

#### Data Validation
```python
# Python script to validate data
from shared_utils.data_loader import DataLoaderFactory, DataValidator

# Load and validate instruction data
instructions = DataLoaderFactory.load_instruction_data("data/train.jsonl")
validation = DataValidator.validate_instruction_data(instructions)
print(f"Validation result: {validation}")
```

---

## 🔧 Advanced Usage Examples

### Pipeline Automation

#### Batch Processing Script
```bash
#!/bin/bash
# batch_process.sh

# Process multiple document folders
for folder in documents/*/; do
    echo "Processing $folder"
    python main.py extract "$folder" "outputs/$(basename $folder)" --format jsonl
done

# Train models on all processed data
for data_file in outputs/*/training_data.jsonl; do
    echo "Training on $data_file"
    python main.py train-lora --train-data "$data_file" --output-dir "models/$(dirname $data_file)"
done
```

#### Python Pipeline Script
```python
# pipeline.py
import subprocess
import sys
from pathlib import Path

def run_pipeline(input_dir, output_dir):
    """Complete pipeline from documents to trained model"""
    
    # Step 1: Extract documents
    extract_cmd = [
        sys.executable, "main.py", "extract",
        str(input_dir), str(output_dir),
        "--format", "jsonl",
        "--instruction-format"
    ]
    subprocess.run(extract_cmd, check=True)
    
    # Step 2: Train LoRA model
    train_cmd = [
        sys.executable, "main.py", "train-lora",
        "--train-data", str(output_dir / "instruction_data.jsonl"),
        "--output-dir", str(output_dir / "lora_model")
    ]
    subprocess.run(train_cmd, check=True)
    
    # Step 3: Train RL agent
    rl_cmd = [
        sys.executable, "main.py", "train-rl",
        "--algorithm", "ppo",
        "--output-dir", str(output_dir / "rl_model")
    ]
    subprocess.run(rl_cmd, check=True)

if __name__ == "__main__":
    run_pipeline(Path("documents"), Path("outputs"))
```

### Integration with External Tools

#### Using with Jupyter Notebooks
```python
# In Jupyter notebook
import sys
sys.path.append('path/to/llm-complete-toolkit')

from document_processing.parsers.pdf_parser import PDFParser
from training_methods.transfer_learning.models.lora_model import LoRAFineTuner

# Process documents
parser = PDFParser()
documents = parser.parse("sample.pdf")

# Train model
fine_tuner = LoRAFineTuner(
    model_name="microsoft/DialoGPT-medium",
    lora_r=16,
    lora_alpha=32
)

# Prepare dataset
train_dataset = fine_tuner.prepare_dataset(
    ["Sample training text..."],
    max_length=512
)

# Train
trainer = fine_tuner.train(
    train_dataset=train_dataset,
    output_dir="./notebook_output"
)
```

#### Docker Usage
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Run document processing
CMD ["python", "main.py", "extract", "/input", "/output", "--format", "jsonl"]
```

```bash
# Build and run
docker build -t llm-toolkit .
docker run -v $(pwd)/documents:/input -v $(pwd)/outputs:/output llm-toolkit
```

---

## 🎛️ Monitoring and Logging Examples

### TensorBoard Integration
```bash
# Start training with TensorBoard
python main.py train-lora --train-data data/train.jsonl --verbose

# In another terminal, start TensorBoard
tensorboard --logdir=outputs/logs/tensorboard

# Open browser to http://localhost:6006
```

### Weights & Biases Integration
```bash
# Enable W&B in config
# configs/wandb_config.yaml
logging:
  use_wandb: true
  wandb_project: "my-llm-experiment"
  wandb_entity: "my-team"

# Train with W&B logging
python main.py train-lora \
  --train-data data/train.jsonl \
  --config configs/wandb_config.yaml
```

### Custom Logging
```python
# custom_training.py
import logging
from shared_utils.training_utils import setup_logging

# Setup custom logging
setup_logging("DEBUG", "training.log")
logger = logging.getLogger(__name__)

# Your training code here
logger.info("Starting custom training...")
```

---

## 🔍 Debugging and Troubleshooting Examples

### Debug Mode
```bash
# Enable debug mode
python main.py train-lora \
  --train-data data/train.jsonl \
  --verbose \
  --config configs/debug_config.yaml
```

### Memory Profiling
```python
# memory_profile.py
import psutil
import torch
from training_methods.transfer_learning.models.qlora_model import QLoRAFineTuner

def monitor_memory():
    """Monitor memory usage during training"""
    process = psutil.Process()
    
    print(f"CPU Memory: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")

# Use QLoRA for memory efficiency
fine_tuner = QLoRAFineTuner(
    model_name="microsoft/DialoGPT-medium",
    load_in_4bit=True
)

monitor_memory()
```

### Error Handling
```python
# error_handling_example.py
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

try:
    from document_processing.parsers.pdf_parser import PDFParser
    
    parser = PDFParser()
    documents = parser.parse("non_existent_file.pdf")
    
except FileNotFoundError as e:
    print(f"File not found: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
    import traceback
    traceback.print_exc()
```

---

## 📈 Performance Optimization Examples

### Memory Optimization
```bash
# Use QLoRA for large models
python main.py train-qlora \
  --train-data data/train.jsonl \
  --model-name microsoft/DialoGPT-large \
  --batch-size 1

# Enable gradient accumulation
python main.py train-lora \
  --train-data data/train.jsonl \
  --batch-size 2 \
  --learning-rate 2e-4
```

### Speed Optimization
```bash
# Use smaller chunks for faster processing
python main.py extract documents/ output/ --chunk-size 256

# Parallel processing (automatically handled)
python main.py extract large_documents/ output/ --format jsonl
```

### GPU Optimization
```python
# gpu_optimization.py
import torch
from training_methods.transfer_learning.models.lora_model import LoRAFineTuner

# Check GPU availability
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Use mixed precision for faster training
    fine_tuner = LoRAFineTuner(
        model_name="microsoft/DialoGPT-medium",
        device="cuda"
    )
else:
    print("Using CPU training")
```

---

## 🎯 Real-world Use Cases

### Use Case 1: Academic Paper Processing
```bash
# Process academic papers
python main.py extract academic_papers/ paper_data/ --format jsonl --instruction-format

# Fine-tune for Q&A
python main.py train-lora \
  --train-data paper_data/instruction_data.jsonl \
  --model-name microsoft/DialoGPT-medium \
  --epochs 3 \
  --output-dir models/academic_qa
```

### Use Case 2: Technical Documentation
```bash
# Process technical docs
python main.py extract tech_docs/ doc_data/ --chunk-size 1024 --format jsonl

# Train for technical assistance
python main.py train-qlora \
  --train-data doc_data/training_data.jsonl \
  --model-name microsoft/DialoGPT-large \
  --epochs 2 \
  --output-dir models/tech_assistant
```

### Use Case 3: Multi-language Documents
```bash
# Process mixed language documents
python main.py extract multilang_docs/ output/ --format jsonl --instruction-format

# Train with different models
python main.py train-lora \
  --train-data output/instruction_data.jsonl \
  --model-name microsoft/DialoGPT-medium \
  --epochs 4 \
  --output-dir models/multilang
```

---

This comprehensive usage guide covers all major use cases and configurations for the LLM Complete Toolkit. Refer to specific sections based on your needs and use cases.
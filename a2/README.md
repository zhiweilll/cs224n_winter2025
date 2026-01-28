# CS224N Assignment 2: Neural Dependency Parsing

| Project                   | Model                         | Date
| ------------------------- | ----------------------------- | -----------------------------
| Neural Dependency Parser  | Transition-based parsing      | 1-26-2026

This repository contains the implementation of a neural dependency parser using transition-based parsing with a feedforward neural network. The parser predicts dependency relations between words in a sentence using a shift-reduce parsing algorithm.

## Overview

This assignment implements:
- **Transition-based dependency parsing** with shift, left-arc, and right-arc operations
- **Feedforward neural network** for predicting parsing transitions
- **Minibatch parsing** for efficient processing of multiple sentences
- **Model training and evaluation** with PyTorch

## Project Structure

```
a2/
├── parser_model.py          # Feedforward neural network model (original)
├── parser_transitions.py    # Transition-based parsing algorithms (original)
├── run.py                   # Main training script (modified: added result saving)
├── evaluate.py              # ⭐ ADDED: Standalone evaluation script
├── utils/
│   ├── parser_utils.py      # Parser utilities and data loading (original)
│   └── general_utils.py     # General utility functions (original)
├── data/                    # Training, dev, and test datasets (original)
├── results/                 # Saved model weights and evaluation results
│   ├── model.weights        # Saved model weights
│   └── test_results.txt     # ⭐ ADDED: Auto-saved evaluation results
└── local_env.yml            # Conda environment configuration (original)


## Setup

### Installation

1. Create an environment with dependencies specified in local_env.yml (note that this can take some time depending on your laptop):
    
    ```bash
    conda env create -f local_env.yml
   ```
2. Activate the new environment:
    
    ```bash
    conda activate cs224n_a2
    ```
3. To deactivate an active environment, use
    
    ```bash
    conda deactivate
    ```

## Usage

### Training

Train the dependency parser on the training set:

```bash
python run.py
```

For debug mode (uses smaller dataset):

```bash
python run.py -d
```

The training script will:
- Train the model for 10 epochs
- Save the best model weights (based on dev UAS) to `results/<timestamp>/model.weights`
- ⭐ **ADDED**: Automatically evaluate on the test set and save results to `results/<timestamp>/test_results.txt`

### Evaluation ⭐ (ADDED)

Evaluate a saved model on the test set without training again:

```bash
python evaluate.py -m results/20260126_212811/model.weights
```

Options:
- `-m, --model`: Path to model weights file (required)
- `-o, --output`: Path to save results (default: same directory as model)
- `-d, --debug`: Use debug mode

Example:
```bash
python evaluate.py -m results/20260126_212811/model.weights -o my_results.txt
```

## Model Architecture

The parser uses a feedforward neural network with:

1. **Embedding Layer**: Word embeddings lookup
2. **First Hidden Layer**: Linear transformation with ReLU activation and dropout
3. **Output Layer**: Linear transformation to predict parsing transitions

The model predicts one of three transition types:
- **SHIFT (S)**: Move a word from buffer to stack
- **LEFT-ARC (LA)**: Create a dependency arc from top stack item to second stack item
- **RIGHT-ARC (RA)**: Create a dependency arc from second stack item to top stack item

## Key Components

### `parser_transitions.py`
- `PartialParse`: Implements the transition-based parsing algorithm
- `parse_step()`: Applies a single parsing transition
- `minibatch_parse()`: Processes multiple sentences in batches

### `parser_model.py`
- `ParserModel`: Feedforward neural network model
- `embedding_lookup()`: Retrieves word embeddings
- `forward()`: Forward pass through the network

### `run.py`
- Training loop with Adam optimizer
- Cross-entropy loss for transition prediction
- Automatic model checkpointing
- ⭐ **ADDED**: Automatic result saving to `test_results.txt` after evaluation

### `evaluate.py` ⭐ (ADDED)
- Standalone script for evaluating saved models
- Loads model weights without retraining
- Saves evaluation results to file
- Supports custom output paths

## Results

Model performance is measured using **Unlabeled Attachment Score (UAS)**, which is the percentage of words with correctly predicted dependency heads.

Results are automatically saved to:
- `results/<timestamp>/test_results.txt` ⭐ - Test UAS scores (auto-saved feature)
- `results/<timestamp>/model.weights` - Trained model weights (original)

**Note:** 
- The automatic result saving feature (⭐) was added by me and is not part of the original assignment.
- AI help me write this README


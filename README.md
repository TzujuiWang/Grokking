
# README: Modular Arithmetic Task with Neural Networks

## Overview
This project investigates the grokking phenomenon in modular arithmetic tasks using different neural network architectures (MLP, LSTM, and Transformer). The project explores how factors such as training data fraction (`alpha`), modular arithmetic complexity (`p`), input dimension (`K`), and optimizer configurations affect learning and generalization. Additionally, it includes tools to analyze weight L2 norms and evaluate performance across various experimental settings.

## Directory Structure
```
grokking/
├── model_kit.py                             # Core module containing models, data generation, and utilities
├── Train_Transformer.py                     # Script for training Transformer models
├── Train_LSTM.py                            # Script for training LSTM models
├── Train_MLP_2_layer.py                     # Script for training a 2-layer MLP model
├── Train_MLP_3_layer.py                     # Script for training a 3-layer MLP model
├── Train_Transformer_half_dataset.py        # Script for training Transformer models with half dataset
├── Transformer_l2_norm.py                   # Script for training Transformer models and tracking L2 norms
├── Transformer_diff_commutative_fraction.py # Script for training Transformer models with varying commutative fractions
├── different_optimizer.py                   # Script for comparing Transformer performance across different optimizers
├── Train_Transformer_diff_k.py              # Script for training Transformer models with different `K` values
└── README.md                                # Project documentation
```

---

## File Descriptions

### 1. `model_kit.py`
This is the core module that defines reusable components for modular arithmetic experiments:
- **Data Generation**:
  - `generate_data`: Generates datasets for modular arithmetic tasks with optional commutative property and customizable fractions.
- **Model Definitions**:
  - `MLPModel`: Multi-layer Perceptron.
  - `LSTMModel`: Long Short-Term Memory network.
  - `TransformerModel`: Transformer-based model for sequence learning.
- **Training Utilities**:
  - `select_model`: Initializes a model based on its type (MLP, LSTM, or Transformer).
  - `select_optimizer`: Configures an optimizer (e.g., AdamW, SGD).
  - `train_model` and `train_model_with_l2_norm`: Functions for training models and tracking L2 norms.
- **Visualization**:
  - `plot_metrics`: Plots loss curves.
  - `plot_accuracy`: Plots accuracy curves.
  - `plot_weight_norms`: Visualizes accuracy and L2 norm trends.

### 2. `Train_Transformer.py`
Trains Transformer models on the modular arithmetic task with varying training data fractions (`alpha`). It includes:
- Data generation with `generate_data`.
- Model initialization using `select_model`.
- Training using `train_model`.
- Visualization of accuracy and loss metrics.

### 3. `Train_LSTM.py`
Trains LSTM models on the modular arithmetic task. It supports additional configurations like embedding dimensions (`embed_dim`). Key steps:
- Data generation with `generate_data`.
- Model initialization with LSTM-specific parameters.
- Training using `train_model`.
- Visualization of accuracy and loss metrics.

### 4. `Train_MLP_2_layer.py`
Trains a 2-layer MLP model on the modular arithmetic task. This script demonstrates the flexibility of MLPs for modular addition. Key steps:
- Data generation with `generate_data`.
- Model initialization with a 2-layer MLP structure.
- Training using `train_model`.
- Visualization of accuracy and loss metrics.

### 5. `Train_MLP_3_layer.py`
Trains a 3-layer MLP model on the modular arithmetic task. This script extends the experiments with a deeper MLP architecture. Key steps:
- Data generation with `generate_data`.
- Model initialization with a 3-layer MLP structure.
- Training using `train_model`.
- Visualization of accuracy and loss metrics.

### 6. `Train_Transformer_half_dataset.py`
Trains Transformer models on a subset of the modular arithmetic dataset. This script utilizes the commutative property to remove duplicate data samples, reducing the dataset size. Key steps:
- Data generation with `generate_data` (commutative property enabled).
- Model initialization with `select_model`.
- Training using `train_model`.
- Visualization of accuracy and loss metrics.

### 7. `Transformer_l2_norm.py`
Trains Transformer models and tracks L2 norms of model weights during training. This script includes:
- Data generation with `generate_data` (commutative property enabled).
- Model initialization with `select_model`.
- Training using `train_model_with_l2_norm`, which tracks L2 norms.
- Visualization of weight L2 norms, accuracy, and loss metrics.

### 8. `Transformer_diff_commutative_fraction.py`
Trains Transformer models with varying commutative fractions in the training data. This script evaluates the impact of different commutative fractions on model performance. Key features:
- Generates datasets with customizable commutative fractions using `generate_data`.
- Trains Transformer models with `train_model`.
- Visualizes results in 3x3 grid subplots using `plot_9_subplots`.

### 9. `different_optimizer.py`
Compares Transformer performance across different optimizers under various experimental configurations. Key features:
- Supports multiple optimizers, including AdamW, SGD (with Nesterov momentum), and others.
- Trains models on datasets with varying training data fractions (`alpha`).
- Saves partial results to avoid redundant computations.
- Visualizes optimizer performance using `plot_optimizer_comparison`.

### 10. `Train_Transformer_diff_k.py`
Trains Transformer models with varying input dimensions (`K`). This script evaluates how changing the number of input tokens impacts model performance. Key features:
- Generates datasets for different values of `K` using `generate_data`.
- Trains Transformer models with `train_model`.
- Visualizes accuracy and loss metrics for each combination of `K` and `alpha`.

---

## How to Use

### Prerequisites
Install the required Python dependencies:
```bash
pip install torch matplotlib scikit-learn
```

### Running the Code
1. **Train a Transformer Model**
   Run the `Train_Transformer.py` script to train a Transformer model with different `alpha` values:
   ```bash
   python Train_Transformer.py
   ```

2. **Train an LSTM Model**
   Run the `Train_LSTM.py` script to train an LSTM model:
   ```bash
   python Train_LSTM.py
   ```

3. **Train a 2-layer MLP Model**
   Run the `Train_MLP_2_layer.py` script to train a 2-layer MLP model:
   ```bash
   python Train_MLP_2_layer.py
   ```

4. **Train a 3-layer MLP Model**
   Run the `Train_MLP_3_layer.py` script to train a 3-layer MLP model:
   ```bash
   python Train_MLP_3_layer.py
   ```

5. **Train a Transformer Model on Half Dataset**
   Run the `Train_Transformer_half_dataset.py` script to train a Transformer model using a subset of the dataset:
   ```bash
   python Train_Transformer_half_dataset.py
   ```

6. **Track L2 Norms with Transformer**
   Run the `Transformer_l2_norm.py` script to train a Transformer model and track its weight L2 norms:
   ```bash
   python Transformer_l2_norm.py
   ```

7. **Train Transformer with Different Commutative Fractions**
   Run the `Transformer_diff_commutative_fraction.py` script to train Transformer models with varying commutative fractions:
   ```bash
   python Transformer_diff_commutative_fraction.py
   ```

8. **Compare Optimizers**
   Run the `different_optimizer.py` script to evaluate Transformer performance under different optimizers:
   ```bash
   python different_optimizer.py
   ```

9. **Train Transformer with Different `K` Values**
   Run the `Train_Transformer_diff_k.py` script to train Transformer models with different `K` values:
   ```bash
   python Train_Transformer_diff_k.py
   ```
   The script outputs accuracy and loss curves for each combination of `K` and `alpha`.

---

## Dependencies
This project requires the following Python libraries:
- `torch` >= 2.0
- `matplotlib`
- `scikit-learn`

Install dependencies using:
```bash
pip install torch matplotlib scikit-learn
```

---

## Notes
- GPU support is recommended for faster training. The scripts automatically detect GPU availability.
- The modular arithmetic parameter `p` determines the problem complexity. Choose appropriately for your experiments.
- For larger datasets or more complex models, consider adjusting batch size and learning rate for optimal performance.
- You can expand the project by adding more network architectures or training tasks.

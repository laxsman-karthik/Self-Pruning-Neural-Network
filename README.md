# Self-Pruning Neural Network

This project implements a neural network that learns to prune its own weights during training. Instead of removing weights after training, the model uses learnable gates to automatically identify and suppress less important connections.

---

## Core Idea

Each weight in the network is associated with a learnable gate:

```
effective_weight = weight × sigmoid(gate_scores)
```

- Gate ≈ 1 → connection is active  
- Gate ≈ 0 → connection is effectively removed  

During training, the model learns both:
- the weight values  
- and which weights should exist  

To encourage pruning, an L1 regularization term is added on the gate values:

```
Loss = Classification Loss + λ × Sparsity Loss
```

This pushes many gate values toward zero, making the network sparse.

---

## Objective

The goal is to:
- Build a self-pruning model  
- Demonstrate how pruning happens during training  
- Show the trade-off between accuracy and sparsity using different λ values  

---

## How to Run

### 1. Install dependencies

```bash
pip install torch torchvision matplotlib
```

### 2. Run the project

```bash
python main.py
```

### 3. What happens when you run

- CIFAR-10 dataset is downloaded automatically (only the first time)
- The model trains for multiple values of λ
- For each λ, the program prints:
  - Test Accuracy
  - Sparsity Level (% of pruned weights)

---

## Output

- Console output showing accuracy and sparsity for each λ  
- A histogram plot of gate values:
  - Saved as `gate_distribution.png`
  - Shows distribution of pruned vs active weights  

---

## Project Structure

```
model.py      → Prunable CNN implementation  
train.py      → Training loop + sparsity loss  
main.py       → Runs experiments  
report.md     → Detailed report  
gate_distribution.png → Output graph  
```

---

## Notes

- Dataset is not included in the repository (auto-downloaded)
- Focus is on sparsity–accuracy trade-off, not maximizing accuracy

---

# Self-Pruning Neural Network

This project implements a neural network that learns to prune its own weights during training. Instead of removing weights after training, the model uses learnable gates to automatically identify and suppress less important connections.

---

## Core Idea

Each weight in the network is associated with a learnable gate:

effective_weight = weight × sigmoid(gate_scores)

- Gate ≈ 1 → connection is active  
- Gate ≈ 0 → connection is effectively removed  

During training, the model learns both:
- the weight values  
- and which weights should exist  

To encourage pruning, an L1 regularization term is added on the gate values:

Loss = Classification Loss + λ × Sparsity Loss

This pushes many gate values toward zero, making the network sparse.

---

## Objective

The goal is to:
- Build a **self-pruning model**
- Demonstrate how pruning happens during training  
- Show the **trade-off between accuracy and sparsity** using different λ values  

---

## How to Run

### 1. Install dependencies
```bash
pip install torch torchvision matplotlib

# ***AlsehoumMiniNN Framework***

AlsehoumMiniNN is a mini neural network framework implemented **from scratch** using pure **NumPy**.
The project aims to demonstrate a deep understanding of how neural networks work internally,
without relying on high-level deep learning frameworks such as PyTorch or TensorFlow.

---

##  Project Purpose

This project was developed as part of the **Neural Networks Lab** course.
Its main goal is to understand and implement:

- Forward Propagation
- Backpropagation
- Gradient computation
- Parameter updates using different optimizers
- Modular neural network design (framework-style)

---

##  Features

- Fully Connected (Dense / Affine) layers
- Activation functions:
  - ReLU
  - Sigmoid
  - Tanh
  - Linear
- Loss functions:
  - Mean Squared Error (MSE)
  - Softmax with Cross-Entropy (combined)
- Optimizers:
  - SGD
  - Momentum
  - Adam
- Mini-batch training
- Validation accuracy tracking
- Modular and extensible design

---

##  Project Structure

AlsehoumMiniNN/
├── layers/
│ ├── base.py
│ ├── dense.py
│ ├── activations.py
│ └── losses.py
├── optimizers/
│ ├── base.py
│ ├── sgd.py
│ ├── momentum.py
│ └── adam.py
├── examples/
│ └── train_iris.py
├── network.py
├── trainer.py
├── tuning.py
└── train.py



---

##  Example Usage

Train a neural network on the Iris dataset:

```bash
python train.py

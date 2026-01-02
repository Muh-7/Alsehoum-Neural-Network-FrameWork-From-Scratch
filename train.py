# examples/train.py

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from layers.Dense import Dense
from layers.Activations import ReLU
from layers.Losses import SoftmaxCrossEntropy
from network import NeuralNetwork
from trainer import Trainer
from Optimizers.sgd import SGD
from Optimizers.adam import Adam



data = load_iris()
X = data.data
y = data.target


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


layers = [
    Dense(4, 16),
    ReLU(),
    Dense(16, 3),
]

loss_fn = SoftmaxCrossEntropy()
model = NeuralNetwork(layers, loss_fn)

# Optimizer
optimizer = Adam(learning_rate=0.001)

# هون تدريب
trainer = Trainer(model, optimizer)

# بلش 
trainer.fit(
    X_train,
    y_train,
    x_val=X_test,
    y_val=y_test,
    epochs=100,
    batch_size=5,
)

# الدقة
acc = model.accuracy(X_test, y_test)
print("Final Test Accuracy:", acc)


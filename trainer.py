# trainer.py

import numpy as np


class Trainer:
 

    def __init__(self, model, optimizer):
       
        self.model = model
        self.optimizer = optimizer

    def train_step(self, x_batch, y_batch):
        
        # Forward + loss
        loss = self.model.loss(x_batch, y_batch)

        # Backward
        self.model.backward()

        # Hon Update p
        for layer in self.model.layers:
            if hasattr(layer, "params"):
                self.optimizer.update(layer.params, layer.grads)

        return loss

    def fit(
        self,
        x_train,
        y_train,
        x_val=None,
        y_val=None,
        epochs=10,
        batch_size=32,
        verbose=True,
    ):

        num_samples = x_train.shape[0]

        for epoch in range(epochs):
            # Shuffle Al bayanat
            indices = np.random.permutation(num_samples)
            x_train = x_train[indices]
            y_train = y_train[indices]

            epoch_loss = 0

            for i in range(0, num_samples, batch_size):
                x_batch = x_train[i : i + batch_size]
                y_batch = y_train[i : i + batch_size]

                loss = self.train_step(x_batch, y_batch)
                epoch_loss += loss

            epoch_loss /= (num_samples // batch_size)

            if verbose:
                msg = f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f}"

                if x_val is not None and y_val is not None:
                    acc = self.model.accuracy(x_val, y_val)
                    msg += f" - Val Accuracy: {acc:.4f}"

                print(msg)


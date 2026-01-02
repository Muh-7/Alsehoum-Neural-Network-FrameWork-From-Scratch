# network.py

import numpy as np

# هون بنبني الشبكة العصبية
class NeuralNetwork:
   

    def __init__(self, layers, loss_fn):
       
        self.layers = layers
        self.loss_fn = loss_fn

# hon al forward
    def forward(self, x):

        for layer in self.layers:
            x = layer.forward(x)
        return x

#Hon Hessab al Loss
    def loss(self, x, y):
       
        scores = self.forward(x)
        loss_value = self.loss_fn.forward(scores, y)
        return loss_value

#Hon Hessab al backward
    def backward(self):

        dout = self.loss_fn.backward()

        for layer in reversed(self.layers):
            dout = layer.backward(dout)

# Hon Sawena al predict
    def predict(self, x):
     
        scores = self.forward(x)
        return np.argmax(scores, axis=1)

# Hon Hessab al deqah Accuracy
    def accuracy(self, x, y):
        
        y_pred = self.predict(x)
        acc = np.mean(y_pred == y)
        return acc

# hon al prarmetrat
    def get_params_and_grads(self):
      
        params = []
        grads = []

        for layer in self.layers:
            if hasattr(layer, "params"):
                for key in layer.params:
                    params.append(layer.params[key])
                    grads.append(layer.grads[key])

        return params, grads


# tuning.py

class HyperparameterTuning:


    def __init__(self):
        self.results = []

    def add_result(self, params, accuracy):
       
        self.results.append({
            "params": params,
            "accuracy": accuracy
        })

    def best_result(self):

        best = max(self.results, key=lambda x: x["accuracy"])
        return best


import numpy as np

class Perceptron:
    def __init__(self, inputSize, hiddenSizes, outputSize):
        
        self.weight_hidden = []

        prev_size = inputSize

        for i in hiddenSizes:
            W = np.zeros((1 + prev_size, i))
            W[0, :] = np.random.randint(0, 3, size=(i))
            W[1:, :] = np.random.randint(-1, 2, size=(prev_size, i))
            self.weight_hidden.append(W)
            prev_size = i

        self.Wout = np.random.randint(0, 2, size=(1 + prev_size, outputSize)).astype(np.float64)

    def predict(self, Xp):
        out = Xp

        hidden_out = []
        
        for W in self.weight_hidden:
            out = np.where((np.dot(out, W[1:, :]) + W[0, :]) >= 0.0, 1, -1).astype(np.float64)
            hidden_out.append(out)

        last_out = np.where((np.dot(out, self.Wout[1:, :]) + self.Wout[0, :]) >= 0.0, 1, -1).astype(np.float64)
        return last_out, hidden_out

    def train(self, X, y, n_iter=5, eta = 0.01):
        for i in range(n_iter):
            print(self.Wout.reshape(1, -1))
            for xi, target, j in zip(X, y, range(X.shape[0])):
                pr, hidden = self.predict(xi)
                last_hidden = hidden[-1]
                self.Wout[1:] += ((eta * (target - pr)) * last_hidden).reshape(-1, 1)
                self.Wout[0] += eta * (target - pr)
        return self


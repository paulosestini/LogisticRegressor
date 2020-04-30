import torch

class LogisticRegressor():
    def __init__(self, lr=0.25):
        self.lr = lr

    def fit(self, x, target, n_iters=1000):
        self.n_dim = x.shape[1]
        self.coefs = torch.randn(self.n_dim + 1, requires_grad=True)

        ones_column = torch.ones((x.shape[0]))
        X = torch.empty(x.shape[0], x.shape[1] + 1)
        X[:, :-1] = x
        X[:, -1] = ones_column

        self.errors = []

        for i in range(n_iters):
            y = self.LogisticFunction(X.matmul(self.coefs))
            error = self.BinaryCrossEntropy(y, target)
            error.backward()
            self.coefs = (self.coefs.data - self.lr*self.coefs.grad).requires_grad_(True)
            self.errors.append(error.detach())

    def predict(self, x):
        ones_column = torch.ones((x.shape[0]))
        X = torch.empty(x.shape[0], x.shape[1] + 1)
        X[:, :-1] = x
        X[:, -1] = ones_column
        return self.LogisticFunction(X.matmul(self.coefs))

    def BinaryCrossEntropy(self, probs, labels):
        return -(labels*torch.log(probs) + (1-labels)*torch.log(1-probs)).mean()

    def LogisticFunction(self, x):
        return 1/(1 + torch.exp(-x))

    def get_error(self):
        return self.errors
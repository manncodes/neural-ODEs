import torch
import torch.nn as nn


class TanhFixedPointLayer(nn.Module):
    """
    A fixed point layer that applies a tanh function to the output.
    """

    def __init__(self, out_features, tolerance=1e-4, max_iter=50):
        super(TanhFixedPointLayer, self).__init__()
        self.linear = nn.Linear(out_features, out_features, bias=False)
        self.tolerance = tolerance
        self.max_iter = max_iter

    def forward(self, x):
        # init output z to be 0
        z = torch.zeros_like(x)
        self.iterations = 0

        # iterate till convergence
        while self.iterations < self.max_iter:
            z_next = torch.tanh(self.linear(z) + x)
            self.err = torch.norm(z - z_next)
            z = z_next
            self.iterations += 1
            # print(f"Iteration #{self.iterations} \t\t\t\t error = {self.err}")
            if self.err < self.tolerance:
                break

        return z


def testFixedPointLayer():
    layer = TanhFixedPointLayer(50)
    x = torch.randn(10, 50)
    z = layer(x)
    print(f"Terminated after {layer.iterations} iterations with error {layer.err}")


# testFixedPointLayer()


class TanhNewtonLayer(nn.Module):
    def __init__(self, out_features, tol=1e-4, max_iter=50):
        super().__init__()
        self.linear = nn.Linear(out_features, out_features, bias=False)
        self.tol = tol
        self.max_iter = max_iter

    def forward(self, x):
        # initialize output z to be zero
        z = torch.tanh(x)
        self.iterations = 0

        # iterate until convergence
        while self.iterations < self.max_iter:
            z_linear = self.linear(z) + x
            g = z - torch.tanh(z_linear)
            self.err = torch.norm(g)
            if self.err < self.tol:
                break

            # newton step
            J = (
                torch.eye(z.shape[1]).to("cuda")[None, :, :]
                - (1 / torch.cosh(z_linear) ** 2)[:, :, None]
                * self.linear.weight[None, :, :]
            )

            z = z - torch.solve(g[:, :, None], J)[0][:, :, 0]
            self.iterations += 1

        g = z - torch.tanh(self.linear(z) + x)
        z[torch.norm(g, dim=1) > self.tol, :] = 0
        return z


def testNewtonLayer():
    layer = TanhNewtonLayer(50)
    X = torch.randn(10, 50)
    Z = layer(X)
    print(f"Terminated after {layer.iterations} iterations with error {layer.err}")


# testNewtonLayer()

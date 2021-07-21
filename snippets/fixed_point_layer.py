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


def test():
    layer = TanhFixedPointLayer(50)
    x = torch.randn(10, 50)
    z = layer(x)
    print(f"Terminated after {layer.iterations} iterations with error {layer.err}")


if __name__ == "__main__":
    test()

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from root_finder import TanhFixedPointLayer, TanhNewtonLayer


mnist_train = datasets.MNIST(
    ".", train=True, download=True, transform=transforms.ToTensor()
)
mnist_test = datasets.MNIST(
    ".", train=False, download=True, transform=transforms.ToTensor()
)
train_loader = DataLoader(mnist_train, batch_size=100, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=100, shuffle=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# %%
# construct the simple model with fixed point layer

torch.manual_seed(0)
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 100),
    TanhFixedPointLayer(100, max_iter=200),
    nn.Linear(100, 10),
).to(device)
opt = optim.SGD(model.parameters(), lr=1e-1)

# %%
# a generic function for running a single epoch (training or evaluation)
from tqdm.notebook import tqdm


def epoch(loader, model, opt=None, monitor=None):
    total_loss, total_err, total_monitor = 0.0, 0.0, 0.0
    model.eval() if opt is None else model.train()
    for X, y in tqdm(loader, leave=False):
        X, y = X.to(device), y.to(device)
        yp = model(X)
        loss = nn.CrossEntropyLoss()(yp, y)
        if opt:
            opt.zero_grad()
            loss.backward()
            if sum(torch.sum(torch.isnan(p.grad)) for p in model.parameters()) == 0:
                opt.step()

        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
        if monitor is not None:
            total_monitor += monitor(model)
    return (
        total_err / len(loader.dataset),
        total_loss / len(loader.dataset),
        total_monitor / len(loader),
    )


# %%
# finally training the model w/ implicit layer solved by Iterative method
for i in range(10):
    if i == 5:
        opt.param_groups[0]["lr"] = 1e-2

    train_err, train_loss, train_fpiter = epoch(
        train_loader, model, opt, lambda x: x[2].iterations
    )
    test_err, test_loss, test_fpiter = epoch(
        test_loader, model, monitor=lambda x: x[2].iterations
    )
    print(
        f"Train Error: {train_err:.4f}, Loss: {train_loss:.4f}, FP Iters: {train_fpiter:.2f} | "
        + f"Test Error: {test_err:.4f}, Loss: {test_loss:.4f}, FP Iters: {test_fpiter:.2f}"
    )


# %%
# testing Newton's method to reach fixed point
torch.manual_seed(0)
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 100),
    TanhNewtonLayer(100, max_iter=40),
    nn.Linear(100, 10),
).to(device)
opt = optim.SGD(model.parameters(), lr=1e-1)

for i in range(8):
    if i == 5:
        opt.param_groups[0]["lr"] = 1e-2

    train_err, train_loss, train_fpiter = epoch(
        train_loader, model, opt, lambda x: x[2].iterations
    )
    test_err, test_loss, test_fpiter = epoch(
        test_loader, model, monitor=lambda x: x[2].iterations
    )
    print(
        f"Train Error: {train_err:.4f}, Loss: {train_loss:.4f}, Newton Iters: {train_fpiter:.2f} | "
        + f"Test Error: {test_err:.4f}, Loss: {test_loss:.4f}, Newton Iters: {test_fpiter:.2f}"
    )

# %%

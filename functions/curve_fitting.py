import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class LinearModel(nn.Module):
    def __init__(self, y):
        super().__init__()
        a_init = np.mean(np.diff(y))
        b_init = y[0]
        self.a = nn.Parameter(torch.tensor([a_init], dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor([b_init], dtype=torch.float32))
    def forward(self, x):
        return self.a * x + self.b

class ExpModel(nn.Module):
    def __init__(self, y):
        super().__init__()
        a_init = y.max() - y.min()
        slope = np.min(np.diff(y))
        b_init = - 1 / (slope + 1e-6)
        c_init = y.min()
        self.a = nn.Parameter(torch.tensor([a_init], dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor([b_init], dtype=torch.float32))
        self.c = nn.Parameter(torch.tensor([c_init], dtype=torch.float32))
    def forward(self, x):
        return self.a * torch.exp(-self.b * x) + self.c

class BiExpModel(nn.Module):
    def __init__(self, y):
        super().__init__()
        turning_point = np.argmax(y)
        a1_init = y[turning_point] - y[0]
        a2_init = y[turning_point] - y[-1]
        slope1 = np.diff(y)[turning_point - 1] if turning_point > 1 else 0
        b1_init = - 1 / (slope1 + 1e-6)
        slope2 = np.diff(y)[turning_point + 1]
        b2_init = - 1 / (slope2 + 1e-6)
        c_init = y.min()
        self.a1 = nn.Parameter(torch.tensor([a1_init], dtype=torch.float32))
        self.a2 = nn.Parameter(torch.tensor([a2_init], dtype=torch.float32))
        self.b1 = nn.Parameter(torch.tensor([b1_init], dtype=torch.float32))
        self.b2 = nn.Parameter(torch.tensor([b2_init], dtype=torch.float32))
        self.c = nn.Parameter(torch.tensor([c_init], dtype=torch.float32))
    def forward(self, x):
        #print(f"Parameters: a1={self.a1.item()}, b1={self.b1.item()}, a2={self.a2.item()}, b2={self.b2.item()}, c={self.c.item()}")
        return self.a1 * torch.exp(-self.b1 * x) + self.a2 * torch.exp(-self.b2 * x) + self.c

class BBModel(nn.Module):
    def __init__(self, y):
        super().__init__()
        a_init = y.max() - y.min()
        b_init = 0.1 #/ np.mean(np.diff(y)) # seems diffucult to estimate
        c_init = y.min()
        self.a = nn.Parameter(torch.tensor([a_init], dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor([b_init], dtype=torch.float32))
        self.c = nn.Parameter(torch.tensor([c_init], dtype=torch.float32))
        self.alpha = nn.Parameter(torch.tensor([5.0], dtype=torch.float32))
    def forward(self, x):
        offset = 1
        return self.a * (x + offset)**(-self.alpha) / (torch.exp(self.b / (x + offset)) - 1) + self.c
    
def model_fitting(x, y, model, device, epoch=500, lr=0.05):
    x_train = torch.tensor(x, dtype=torch.float32).view(-1, 1).to(device)
    y_train = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(device)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    loss_min = np.inf
    best_model_dict = None

    for _ in range(epoch):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(x_train), y_train)
        if loss < loss_min:
            loss_min = loss.item()
            best_model_dict = model.state_dict()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        if best_model_dict is not None:
            model.load_state_dict(best_model_dict)
        model.eval()
        x_test = np.linspace(x.min(), x.max(), 100)
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32).view(-1, 1).to(device)
        y_fit = model(x_test_tensor).detach().cpu().numpy().squeeze()
        return x_test, y_fit, [p.item() for p in model.parameters()]
    
if __name__ == "__main__":
    # Example usage
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = np.linspace(0, 20, 100)
    y = 2 * np.exp(-0.1 * x) - 1.5* np.exp(-2 * x) + 0.1 + 0.1 * np.random.normal(size=x.shape)
    #y = (x+1)**(-5) / (np.exp(1 / (x+1)) - 1) + 0.1 + 0.1 * np.random.normal(size=x.shape)
    #y = np.exp(-x) + 0.1 * np.random.normal(size=x.shape)

    #model = BBModel(y)
    #model = BiExpModel(y)
    #model.to(device)
    #x_fit, y_fit, params = model_fitting(x, y, model, device)
    
    import matplotlib.pyplot as plt
    plt.plot(x, y, label='Data')
    #plt.plot(x_fit, y_fit, label='Fitted Model')
    plt.legend()
    plt.show()

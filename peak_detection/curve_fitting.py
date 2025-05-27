import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# from scipy.optimize import curve_fit # Not used in the fetched content, but might be useful for alternative fitting strategies

class LinearModel(nn.Module):
    def __init__(self, y): # y is expected to be a numpy array for initialization logic
        super().__init__()
        # Ensure y is a numpy array for np.diff, np.mean, etc.
        y_np = y if isinstance(y, np.ndarray) else np.array(y)
        
        a_init = np.mean(np.diff(y_np)) if len(y_np) > 1 else 0.0
        b_init = y_np[0] if len(y_np) > 0 else 0.0
        
        self.a = nn.Parameter(torch.tensor([a_init], dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor([b_init], dtype=torch.float32))
    def forward(self, x):
        return self.a * x + self.b

class ExpModel(nn.Module):
    def __init__(self, y): # y is expected to be a numpy array
        super().__init__()
        y_np = y if isinstance(y, np.ndarray) else np.array(y)

        if len(y_np) == 0: # Handle empty y
            a_init, b_init, c_init = 0.0, 1.0, 0.0
        elif len(y_np) == 1:
             a_init, b_init, c_init = y_np[0], 1.0, 0.0
        else:
            a_init = y_np.max() - y_np.min()
            # Ensure diff_y is not empty and slope calculation is safe
            diff_y = np.diff(y_np)
            slope = np.min(diff_y) if len(diff_y) > 0 else 0.0 
            b_init = -1 / (slope + 1e-6) if slope != -1e-6 else 1.0 # Avoid division by zero or extreme values
            c_init = y_np.min()

        self.a = nn.Parameter(torch.tensor([a_init], dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor([b_init if np.isfinite(b_init) else 1.0], dtype=torch.float32)) # Ensure b_init is finite
        self.c = nn.Parameter(torch.tensor([c_init], dtype=torch.float32))

    def forward(self, x):
        return self.a * torch.exp(-self.b * x) + self.c

class BiExpModel(nn.Module):
    def __init__(self, y): # y is expected to be a numpy array
        super().__init__()
        y_np = y if isinstance(y, np.ndarray) else np.array(y)

        if len(y_np) < 2: # Need at least 2 points for reasonable initial estimates
             a1_init, a2_init, b1_init, b2_init, c_init = 0.0, 0.0, 1.0, 1.0, np.mean(y_np) if len(y_np)>0 else 0.0
        else:
            turning_point = np.argmax(y_np)
            a1_init = y_np[turning_point] - y_np[0] if turning_point > 0 else 0.0
            a2_init = y_np[turning_point] - y_np[-1] if turning_point < len(y_np) -1 else 0.0
            
            diff_y = np.diff(y_np)
            slope1 = diff_y[turning_point - 1] if turning_point > 0 and turning_point <= len(diff_y) else 0.0
            b1_init = -1 / (slope1 + 1e-6) if slope1 != -1e-6 else 1.0
            
            slope2 = diff_y[turning_point] if turning_point < len(diff_y) else 0.0
            b2_init = -1 / (slope2 + 1e-6) if slope2 != -1e-6 else 1.0 # Original was slope2 = np.diff(y)[turning_point + 1]
                                                                   # which could go out of bounds.
            c_init = y_np.min()

        self.a1 = nn.Parameter(torch.tensor([a1_init], dtype=torch.float32))
        self.a2 = nn.Parameter(torch.tensor([a2_init], dtype=torch.float32))
        self.b1 = nn.Parameter(torch.tensor([b1_init if np.isfinite(b1_init) else 1.0], dtype=torch.float32))
        self.b2 = nn.Parameter(torch.tensor([b2_init if np.isfinite(b2_init) else 1.0], dtype=torch.float32))
        self.c = nn.Parameter(torch.tensor([c_init], dtype=torch.float32))
        
    def forward(self, x):
        return self.a1 * torch.exp(-self.b1 * x) + self.a2 * torch.exp(-self.b2 * x) + self.c

class BBModel(nn.Module): # Planck-like function or Blackbody Radiation function
    def __init__(self, y): # y is expected to be a numpy array
        super().__init__()
        y_np = y if isinstance(y, np.ndarray) else np.array(y)
        
        if len(y_np) == 0:
            a_init, b_init, c_init, alpha_init = 0.0, 0.1, 0.0, 5.0
        else:
            a_init = y_np.max() - y_np.min() if y_np.size > 0 else 0.0
            b_init = 0.1 
            c_init = y_np.min() if y_np.size > 0 else 0.0
            alpha_init = 5.0

        self.a = nn.Parameter(torch.tensor([a_init], dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor([b_init], dtype=torch.float32))
        self.c = nn.Parameter(torch.tensor([c_init], dtype=torch.float32))
        self.alpha = nn.Parameter(torch.tensor([alpha_init], dtype=torch.float32)) # alpha is often positive

    def forward(self, x):
        offset = 1 # To prevent division by zero or log(0) if x starts at 0
        # Ensure (x + offset) is positive for fractional powers and exp denominator
        safe_x = x + offset 
        # Ensure b / safe_x doesn't lead to exp overflow if b is large and safe_x is small.
        # Clamp exp_term to avoid overflow if b / safe_x is too large.
        exp_term_input = self.b / torch.clamp(safe_x, min=1e-6)
        exp_term = torch.exp(torch.clamp(exp_term_input, max=80)) # exp(80) is already very large
        
        # Ensure (safe_x)**(-self.alpha) is well-behaved.
        # If alpha is positive, safe_x should not be zero.
        # If alpha can be negative, (safe_x) must be positive.
        # Since safe_x = x+1 and x is usually time (>=0), safe_x is positive.
        term1 = self.a * (torch.clamp(safe_x, min=1e-6))**(-torch.clamp(self.alpha, min=0)) # ensure alpha is positive for decay

        return term1 / torch.clamp(exp_term - 1, min=1e-6) + self.c


def model_fitting(x, y, model, device, epoch=500, lr=0.05):
    # Ensure x and y are numpy arrays before converting to tensors
    x_np = x if isinstance(x, np.ndarray) else np.array(x)
    y_np = y if isinstance(y, np.ndarray) else np.array(y)

    x_train = torch.tensor(x_np, dtype=torch.float32).view(-1, 1).to(device)
    y_train = torch.tensor(y_np, dtype=torch.float32).view(-1, 1).to(device)
    
    model.to(device) # Move model to device
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    loss_min = float('inf') # Use basic float for min loss tracking
    best_model_state_dict = None # Store the best model state

    for _iter_epoch in range(epoch): # Renamed epoch to avoid conflict with outer scope if any
        model.train()
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = loss_fn(y_pred, y_train)
        
        if loss.item() < loss_min:
            loss_min = loss.item()
            best_model_state_dict = model.state_dict()
            
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        if best_model_state_dict is not None:
            model.load_state_dict(best_model_state_dict)
        model.eval()
        
        # Generate points for plotting the fitted curve
        x_test_np = np.linspace(x_np.min(), x_np.max(), 100)
        x_test_tensor = torch.tensor(x_test_np, dtype=torch.float32).view(-1, 1).to(device)
        y_fit_np = model(x_test_tensor).cpu().numpy().squeeze() # Changed to .cpu().numpy()
        
        # Extract final parameters
        final_params = [p.item() for p in model.parameters()]
        
        return x_test_np, y_fit_np, final_params

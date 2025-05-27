import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class BatchedLorentzianFitter(torch.nn.Module):
    def __init__(self, num_spectra, num_peaks, peak_positions, gamma_range=[5,20], init_a=1.0, init_gamma=1.0, init_background=0.1):
        super().__init__()
        self.num_spectra = num_spectra
        self.num_peaks = num_peaks
        self.gamma_min, self.gamma_max = gamma_range

        # Learnable log(a) and logit(gamma)
        self.raw_a = torch.nn.Parameter(torch.log(torch.ones(num_spectra, num_peaks) * init_a))

        self.raw_background = torch.nn.Parameter(torch.log(torch.tensor([init_background])))

        # Rescale init_gamma into [0,1] → logit
        init_gamma_scaled = (init_gamma - self.gamma_min) / (self.gamma_max - self.gamma_min)
        self.raw_gamma = torch.nn.Parameter(torch.logit(torch.ones(num_spectra, num_peaks) * init_gamma_scaled))

        # Fixed peak positions
        self.peak_positions = torch.tensor(peak_positions, dtype=torch.float32)

    def forward(self, x):
        #N, P = self.raw_a.shape
        #F = x.shape[0]
        x = x[None, None, :]  # [1, 1, F], Frequency axis

        a = torch.exp(self.raw_a)[:, :, None]  # [N, P, 1]
        # Background term
        background = torch.exp(self.raw_background)
        
        # gamma = sigmoid(raw) * (max - min) + min
        gamma = torch.sigmoid(self.raw_gamma)[:, :, None] * (self.gamma_max - self.gamma_min) + self.gamma_min
        peak = self.peak_positions[None, :, None]  # [1, P, 1]

        lorentz = a * (gamma**2) / ((x - peak)**2 + gamma**2)  # [N, P, F]
        output = lorentz.sum(dim=1) + background # [N, F]
        return output, lorentz, a, gamma, background


def apply_peak_fitting_batch(data, peak_positions, gamma_range=[5, 20], init_gamma=1.0, initial_bg=0.1, epochs=200, lr=0.01, device='cpu'):
    F = data.shape[1]
    data = data.transpose(0, 2, 3, 4, 1)
    spectra = data.reshape(-1, F)
    
    x_np = np.arange(spectra.shape[1])
    x = torch.tensor(x_np, dtype=torch.float32).to(device)         # [F]
    y = torch.tensor(spectra, dtype=torch.float32).to(device)      # [N, F]
    N = y.shape[0]

    model = BatchedLorentzianFitter(num_spectra=N, num_peaks=len(peak_positions), peak_positions=peak_positions,
                                    gamma_range=gamma_range, init_gamma=init_gamma, init_background=initial_bg)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    loss_min = np.inf
    best_model_dict = None

    for _ in range(epochs):
        optimizer.zero_grad()
        output = model(x)[0]
        loss = criterion(output, y)
        if loss.item() < loss_min:
            loss_min = loss.item()
            best_model_dict = model.state_dict()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        if best_model_dict is not None:
            model.load_state_dict(best_model_dict)
        
        fitted, components = model(x)[0],  model(x)[1] # [N, F], [N, P, F]
        a = torch.exp(model.raw_a)  # [N, P]
        gamma = (torch.sigmoid(model.raw_gamma) * (model.gamma_max - model.gamma_min) + model.gamma_min)  # [N, P]
        bg = torch.exp(model.raw_background)  # [1]

        a = a.cpu().numpy()
        gamma = gamma.cpu().numpy()
        fitted = fitted.cpu().numpy()
        components = components.cpu().numpy()

    return fitted, components, a, gamma, bg

if __name__ == "__main__":
    # Example usage
    data = np.random.rand(20, 250, 10, 10, 10)  
    peak_positions = [20, 30, 50, 60]  # Example peak positions
    fitted, components, a, gamma, bg = apply_peak_fitting_batch(data, peak_positions)
    print("Fitted:", fitted.shape)
    print("Components:", components.shape)
    print("Amplitudes:", a.shape)
    print("Gammas:", gamma.shape)
    print("Background:", bg)
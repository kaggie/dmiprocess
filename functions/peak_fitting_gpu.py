#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class BatchedSmartLorentzianTimeModel(nn.Module):
    def __init__(self, T, F, B, freqs, param_peak, param_gamma, min_gamma=5, max_gamma=20, 
                 peak_shift_limit=2, num_peaks=4, initial_amplitudes=None):
        super().__init__()
        self.T = T # Number of time points
        self.F = F # Number of frequency points
        self.B = B # Batch size
        self.num_peaks = num_peaks
        assert len(param_peak) == num_peaks
        #self.freqs = freqs.view(1, F, 1).expand(B, F, num_peaks)  # (B, F, num_peaks)
        self.register_buffer('freqs', freqs.view(1, F, 1).expand(B, F, num_peaks))  # (B, F, num_peaks)

        self.min_gamma = min_gamma
        self.max_gamma = max_gamma
        self.delta_gamma = max_gamma - min_gamma
        self.peak_shift_limit = peak_shift_limit
        
        if initial_amplitudes is not None:
            self.raw_a = nn.Parameter(torch.log(torch.clamp(initial_amplitudes, min=1e-8)))
        else:
            self.raw_a = nn.Parameter(torch.randn(B, T, num_peaks))  # (B, T, P)

        gamma_init = [(g - min_gamma) / self.delta_gamma for g in param_gamma]
        self.gamma_param = nn.Parameter(torch.tensor(gamma_init).repeat(B, 1))  # (B, P)

        #self.peak_init = torch.tensor(param_peak, dtype=torch.float32).repeat(B, 1)  # (B, P)
        self.register_buffer('peak_init', torch.tensor(param_peak, dtype=torch.float32).repeat(B, 1))  # (B, P)
        self.peak_offset = nn.Parameter(torch.zeros(B, num_peaks))  # (B, P)

        self.background = nn.Parameter(torch.tensor(0.0)) # scalar background

    def _constrain_gamma(self):
        return self.min_gamma + self.delta_gamma * torch.sigmoid(self.gamma_param)  # (B, P)

    def _constrain_peak(self):
        return self.peak_init + self.peak_shift_limit * torch.tanh(self.peak_offset)  # (B, P)

    def forward(self):
        a = torch.exp(self.raw_a)  # (B, T, P)
        gamma = self._constrain_gamma().unsqueeze(1)  # (B, 1, P)
        peak = self._constrain_peak().unsqueeze(1)    # (B, 1, P)

        x = self.freqs  # (B, F, P)
        L = (gamma ** 2) / ((x - peak) ** 2 + gamma ** 2)  # (B, F, P)
        L = L.permute(0, 2, 1)  # (B, P, F)

        spectra = torch.matmul(a, L) + self.background  # (B, T, F)
        #components = torch.matmul(a.unsqueeze(-1), L.unsqueeze(1))  # (B, P, F)
        components = a.unsqueeze(-1)*L.unsqueeze(1) # (B, T, P, F)
        return spectra, components, a, gamma.squeeze(1), self.background.squeeze()


def fit_batched_smart_model(x, y, param_peak, param_gamma, min_gamma=5, max_gamma=20, 
                            peak_shift_limit=2, num_peaks=4, epochs=3000, lr=0.05, verbose=False):
    """
    x: (F,)
    y: (B, T, F)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B, T, F = y.shape

    x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
    y_max = y_tensor.amax(dim=(1, 2), keepdim=True) + 1e-8
    y_norm = y_tensor / y_max

    # Estimate initial amplitudes by sampling y at peak locations
    initial_a = torch.zeros((B, T, len(param_peak)), dtype=torch.float32)
    for p, peak in enumerate(param_peak):
        idx = (np.abs(x - peak)).argmin()
        initial_a[:, :, p] = y_tensor[:, :, idx]
    initial_a = initial_a.to(device)

    model = BatchedSmartLorentzianTimeModel(T=T, F=F, B=B, freqs=x_tensor, 
                                            min_gamma=min_gamma, max_gamma=max_gamma, 
                                            peak_shift_limit=peak_shift_limit, num_peaks=num_peaks, 
                                            param_peak=param_peak, param_gamma=param_gamma,
                                            initial_amplitudes=initial_a
                                            ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    #loss_fn = nn.MSELoss()
    loss_min = np.inf
    best_model_dict = None

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model()[0]  # (B, T, F)
        loss = ((output - y_tensor) ** 2 * y_norm).mean()
        #loss = loss_fn(output, y_tensor)
        if loss.item() < loss_min:
            loss_min = loss.item()
            best_model_dict = model.state_dict()
        loss.backward()
        optimizer.step()
        if verbose and epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    with torch.no_grad():
        if best_model_dict is not None:
            model.load_state_dict(best_model_dict)
        spectra, components, a, gamma, bg = model()
        spectra = spectra.detach().cpu().numpy()  # (B, T, F)
        components = components.detach().cpu().numpy()  # (B, P, F)
        a = a.detach().cpu().numpy()  # (B, T, P)
        gamma = gamma.detach().cpu().numpy()  # (B, P)
        bg = bg.detach().cpu().numpy()  # (B,)

    return model, spectra, components, a, gamma, bg



def fit_volume_gpu(I, x, param_peak, param_gamma, min_gamma=5, max_gamma=20, 
                            peak_shift_limit=2, num_peaks=4, epochs=3000, lr=0.05, batch_size=64):
    """
    I: (T, F, X, Y, Z)
    x: (F,)
    """
    T, F, X, Y, Z = I.shape
    voxels = I.reshape(T, F, -1).transpose(2, 0, 1)  # (N, T, F)
    N = voxels.shape[0]

    fitted_spectra = np.zeros((N, T, F), dtype=np.float32)
    step = batch_size
    components_all = []
    a_all = []
    gamma_all = []
    bg_all = []

    for start in tqdm(range(0, N, step)):
        end = min(start + step, N)
        y_batch = voxels[start:end]  # (B, T, F)

        _, fitted, components, a, gamma, bg= fit_batched_smart_model(x, y_batch,min_gamma=min_gamma, max_gamma=max_gamma, 
                                                                     peak_shift_limit=peak_shift_limit, num_peaks=num_peaks,
                                                                     param_peak=param_peak, param_gamma=param_gamma,epochs=epochs, lr=lr, verbose=False
                                                                     )
        fitted_spectra[start:end] = fitted # (B, T, F)
        components_all.append(components) # (B, T, P, F)
        a_all.append(a) # (B, T, P)
        gamma_all.append(gamma) # (B, P)
        bg_all.append(bg) # scalar

    fitted_spectra = fitted_spectra.reshape(X, Y, Z, T, F).transpose(3, 4, 0, 1, 2)  # (T, F, X, Y, Z)
    components_all = np.concatenate(components_all, axis=0)  # (N, T, P, F)
    a_all = np.concatenate(a_all, axis=0)  # (N, T, P)
    gamma_all = np.concatenate(gamma_all, axis=0)  # (N, P)
    bg_all = np.array(bg_all).reshape(-1) 
    return fitted_spectra, components_all, a_all, gamma_all, bg_all


if __name__ == "__main__":
    # Example usage
    data = np.random.rand(20, 250, 10, 10, 10)  
    x = np.linspace(0, 1, 250)  # Frequency axis
    peak_positions = [20, 30, 50, 60]  # Example peak positions
    fitted, components, a, gamma, bg = fit_volume_gpu(data, x, [10, 15, 20, 25], [10,10,10,10], epochs=1000, batch_size=64)
    print("Fitted:", fitted.shape)
    print("Components:", components.shape)
    print("Amplitudes:", a.shape)
    print("Gammas:", gamma.shape)
    print("Background:", bg)
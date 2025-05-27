import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from config.settings import get_device # Import get_device

# Commented out DEVICE definition removed

class BatchedSmartLorentzianTimeModel(nn.Module):
    def __init__(self, T, F, B, freqs, param_peak, param_gamma, min_gamma=5, max_gamma=20,
                 peak_shift_limit=2, num_peaks=4, initial_amplitudes=None):
        super().__init__()
        self.T = T 
        self.F = F 
        self.B = B 
        self.num_peaks = num_peaks
        assert len(param_peak) == num_peaks, "Length of param_peak must match num_peaks"
        
        self.register_buffer('freqs', freqs.view(1, F, 1).expand(B, F, num_peaks))

        self.min_gamma = min_gamma
        self.max_gamma = max_gamma
        self.delta_gamma = max_gamma - min_gamma
        self.peak_shift_limit = peak_shift_limit

        if initial_amplitudes is not None:
            self.raw_a = nn.Parameter(torch.log(torch.clamp(initial_amplitudes, min=1e-8)))
        else:
            self.raw_a = nn.Parameter(torch.randn(B, T, num_peaks))

        gamma_init_list = [(g - min_gamma) / self.delta_gamma for g in param_gamma]
        self.gamma_param = nn.Parameter(torch.tensor(gamma_init_list, dtype=torch.float32).repeat(B, 1))
        
        self.register_buffer('peak_init', torch.tensor(param_peak, dtype=torch.float32).repeat(B, 1))
        self.peak_offset = nn.Parameter(torch.zeros(B, num_peaks))

        self.background = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def _constrain_gamma(self):
        return self.min_gamma + self.delta_gamma * torch.sigmoid(self.gamma_param)

    def _constrain_peak(self):
        return self.peak_init + self.peak_shift_limit * torch.tanh(self.peak_offset)

    def forward(self):
        a = torch.exp(self.raw_a)
        gamma = self._constrain_gamma().unsqueeze(1)
        peak = self._constrain_peak().unsqueeze(1)

        x = self.freqs
        L_numerator = gamma ** 2
        L_denominator = (x - peak) ** 2 + gamma ** 2
        L = L_numerator / torch.clamp(L_denominator, min=1e-12)
        
        L = L.permute(0, 2, 1)

        spectra = torch.matmul(a, L) + self.background
        components = a.unsqueeze(-1) * L.unsqueeze(1)
        return spectra, components, a, gamma.squeeze(1), self.background


def fit_batched_smart_model(x, y, param_peak, param_gamma, min_gamma=5, max_gamma=20,
                            peak_shift_limit=2, num_peaks=4, epochs=3000, lr=0.05, verbose=False):
    device = get_device() # Use get_device() from config.settings
    B, T, F = y.shape

    x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
    
    y_max_for_weighting = y_tensor.amax(dim=(1, 2), keepdim=True).clamp(min=1e-8) 
    y_norm_for_loss_weighting = y_tensor / y_max_for_weighting

    initial_a = torch.zeros((B, T, len(param_peak)), dtype=torch.float32, device=device)
    for p_idx, peak_val in enumerate(param_peak): 
        idx = (torch.abs(x_tensor - peak_val)).argmin()
        initial_a[:, :, p_idx] = y_tensor[:, :, idx] 

    model = BatchedSmartLorentzianTimeModel(T=T, F=F, B=B, freqs=x_tensor,
                                            min_gamma=min_gamma, max_gamma=max_gamma,
                                            peak_shift_limit=peak_shift_limit, num_peaks=num_peaks,
                                            param_peak=param_peak, param_gamma=param_gamma,
                                            initial_amplitudes=initial_a
                                            ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_min = torch.tensor(float('inf'), device=device)
    best_model_state_dict = None 

    for epoch in range(epochs):
        optimizer.zero_grad()
        output_spectra, _, _, _, _ = model()
        loss = ((output_spectra - y_tensor) ** 2 * y_norm_for_loss_weighting).mean()
        
        if loss.item() < loss_min.item():
            loss_min = loss.clone() 
            best_model_state_dict = model.state_dict() 
        
        loss.backward()
        optimizer.step()
        if verbose and epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    with torch.no_grad():
        if best_model_state_dict is not None:
            model.load_state_dict(best_model_state_dict)
        
        spectra, components, a, gamma, bg = model()
        
        spectra_np = spectra.cpu().numpy()
        components_np = components.cpu().numpy()
        a_np = a.cpu().numpy()
        gamma_np = gamma.cpu().numpy()
        bg_np = bg.cpu().numpy() 

    return model, spectra_np, components_np, a_np, gamma_np, bg_np


def fit_volume_gpu(I, x, param_peak, param_gamma, min_gamma=5, max_gamma=20,
                   peak_shift_limit=2, num_peaks=4, epochs=3000, lr=0.05, batch_size=64):
    T, F, X, Y, Z = I.shape
    voxels = I.transpose(2, 3, 4, 0, 1).reshape(-1, T, F)
    N = voxels.shape[0]

    fitted_spectra_all = np.zeros((N, T, F), dtype=np.float32)
    components_all_list = [] 
    a_all_list = []
    gamma_all_list = []
    bg_all_list = [] 

    for start in tqdm(range(0, N, batch_size), desc="Fitting Volume Batches"):
        end = min(start + batch_size, N)
        y_batch_np = voxels[start:end]

        _, fitted_batch, components_batch, a_batch, gamma_batch, bg_batch = fit_batched_smart_model(
            x, y_batch_np, 
            param_peak=param_peak, param_gamma=param_gamma,
            min_gamma=min_gamma, max_gamma=max_gamma,
            peak_shift_limit=peak_shift_limit, num_peaks=num_peaks,
            epochs=epochs, lr=lr, verbose=False
        )
        
        fitted_spectra_all[start:end] = fitted_batch
        components_all_list.append(components_batch) 
        a_all_list.append(a_batch)                   
        gamma_all_list.append(gamma_batch)           
        bg_all_list.append(bg_batch)

    fitted_spectra_reshaped = fitted_spectra_all.reshape(X, Y, Z, T, F).transpose(3, 4, 0, 1, 2)

    components_final = np.concatenate(components_all_list, axis=0) 
    a_final = np.concatenate(a_all_list, axis=0)                   
    gamma_final = np.concatenate(gamma_all_list, axis=0)           
    
    bg_final = np.mean(bg_all_list) if bg_all_list else 0.0
    
    components_final_reshaped = components_final.reshape(X,Y,Z,T,num_peaks,F)
    a_final_reshaped = a_final.reshape(X,Y,Z,T,num_peaks)
    gamma_final_reshaped = gamma_final.reshape(X,Y,Z,num_peaks)

    return fitted_spectra_reshaped, components_final_reshaped, a_final_reshaped, gamma_final_reshaped, bg_final

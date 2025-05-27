import numpy as np
import pywt # From classic_denoiser_content (Wavelet Thresholding)
from scipy.ndimage import uniform_filter1d, gaussian_filter1d, median_filter # From classic_denoiser_content
from scipy.signal import savgol_filter, wiener # From classic_denoiser_content
from skimage.restoration import denoise_tv_chambolle # From classic_denoiser_content
from sklearn.decomposition import PCA # From classic_denoiser_content
import torch
import torch.nn as nn
import torch.nn.functional as F # From denoise_trans_pe_content, good to have for nn.Module
import math # From denoise_unet_pe_content & denoise_trans_pe_content
from config.settings import get_device # Import get_device

# Global DEVICE definition removed

# --- Content from classic_denoiser.py ---
def apply_spectral_denoising_batch(data, method, **kwargs):
    # This function does not use DEVICE
    T, F_dim, X, Y, Z = data.shape[:5] 
    transposed_data = data.transpose(0, 2, 3, 4, 1) 
    spectra = transposed_data.reshape(-1, F_dim)
    denoised_spectra = None

    if method == 'Mean Filter':
        size = kwargs.get('window_size', 5)
        denoised_spectra = uniform_filter1d(spectra, size=size, axis=1, mode='nearest')
    elif method == 'Median Filter':
        size = kwargs.get('window_size', 5)
        denoised_spectra = median_filter(spectra, size=(1,size), mode='nearest')
    elif method == 'Gaussian Filter':
        sigma = kwargs.get('sigma', 1.0)
        denoised_spectra = gaussian_filter1d(spectra, sigma=sigma, axis=1, mode='nearest')
    elif method == 'Singular Value Decomposition':
        n_components = kwargs.get('num_components', 5)
        U, S, Vh = np.linalg.svd(spectra, full_matrices=False)
        S[n_components:] = 0
        denoised_spectra = (U * S) @ Vh
    elif method == 'Principal Component Analysis':
        n_components = kwargs.get('num_components', 5)
        n_samples, n_features = spectra.shape
        actual_n_components = min(n_components, n_samples, n_features)
        if actual_n_components == 0 : 
             denoised_spectra = spectra.copy()
        else:
            pca = PCA(n_components=actual_n_components)
            transformed_spectra = pca.fit_transform(spectra)
            denoised_spectra = pca.inverse_transform(transformed_spectra)
    elif method == 'Savitzky-Golay Filter':
        window_length = kwargs.get('window_size', 7)
        polyorder = kwargs.get('polyorder', 2)
        if window_length % 2 == 0: window_length += 1
        window_length = min(window_length, F_dim)
        if polyorder >= window_length: polyorder = window_length - 1 
        if window_length <= 0 or polyorder < 0 : 
            denoised_spectra = spectra.copy() 
        else:
            denoised_spectra = np.stack([
                savgol_filter(s, window_length, polyorder, mode='nearest') for s in spectra
            ])
    elif method == 'Wavelet Thresholding':
        wavelet = kwargs.get('wavelet', 'db4')
        threshold_val = kwargs.get('threshold', 0.04) 
        mode = kwargs.get('mode', 'soft')
        level = kwargs.get('level', None)
        denoised_list = [] 
        for s in spectra:
            coeffs = pywt.wavedec(s, wavelet, level=level)
            coeffs_thresh = [pywt.threshold(c, threshold_val, mode=mode) if i > 0 else c
                             for i, c in enumerate(coeffs)]
            s_denoised = pywt.waverec(coeffs_thresh, wavelet)
            denoised_list.append(s_denoised[:F_dim])
        denoised_spectra = np.stack(denoised_list)
    elif method == 'Fourier Filter':
        cutoff = kwargs.get('cutoff_freq', 0.1)
        denoised_list = [] 
        for s in spectra:
            S_fft = np.fft.fft(s) 
            freq_cut = int(len(S_fft) * cutoff)
            S_fft[freq_cut:-freq_cut] = 0
            s_denoised = np.fft.ifft(S_fft).real
            denoised_list.append(s_denoised)
        denoised_spectra = np.stack(denoised_list)
    elif method == 'Total Variation':
        weight = kwargs.get('weight', 0.1)
        denoised_spectra = np.stack([
            denoise_tv_chambolle(s, weight=weight) for s in spectra
        ])
    elif method == 'Wiener Filter':
        kernel_size = kwargs.get('kernel_size', 5)
        if kernel_size % 2 == 0: kernel_size +=1
        denoised_spectra = np.stack([
            wiener(s, mysize=kernel_size) for s in spectra
        ])
    else:
        raise ValueError(f"Unsupported denoising method: {method}")

    if T * X * Y * Z == 1: 
        return denoised_spectra.reshape(F_dim) 
    else: 
        return denoised_spectra.reshape(T, X, Y, Z, F_dim).transpose(0, 4, 1, 2, 3)

class PEwithPeak(nn.Module):
    def __init__(self, embed_dim=32, max_len=256, num_peaks=4, **kwargs):
        super(PEwithPeak, self).__init__()
        self.dim = embed_dim
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        self.peak_embedding = nn.Embedding(max_len, embed_dim)

    def forward(self, x, peak_positions): 
        seq_len, batch_size, embed_dim = x.shape
        x = x + self.pe[:seq_len, :, :] 
        peak_embed_sum = torch.zeros_like(x, device=x.device)
        for i in range(batch_size):
            current_peaks = peak_positions[i]
            valid_mask = (current_peaks >= 0) & (current_peaks < seq_len)
            valid_idx = current_peaks[valid_mask]
            if valid_idx.numel() > 0:
                selected_peak_embeds = self.peak_embedding(valid_idx)
                for j, peak_idx in enumerate(valid_idx):
                    peak_embed_sum[peak_idx, i, :] += selected_peak_embeds[j, :]
        x = x + peak_embed_sum
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.conv = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        x_conv = self.conv(x) 
        x_down = self.pool(x_conv)
        return x_conv, x_down 

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels): 
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            ConvBlock(in_channels, out_channels), 
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x, skip):
        x_up = self.up(x)
        if x_up.shape[2] != skip.shape[2]:
            padding = skip.shape[2] - x_up.shape[2]
            x_up = F.pad(x_up, [padding // 2, padding - padding // 2])
        x_cat = torch.cat([x_up, skip], dim=1)
        return self.conv(x_cat)

class UNet1DWithPEPeak(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim=32, max_len=256, **kwargs): 
        super(UNet1DWithPEPeak, self).__init__()
        self.initial_conv = ConvBlock(in_channels, embed_dim) 
        self.positional_encoding = PEwithPeak(embed_dim, max_len)
        self.enc1 = DownBlock(embed_dim, 64)
        self.enc2 = DownBlock(64, 128)
        self.enc3 = DownBlock(128, 256)
        self.enc4 = DownBlock(256, 512)
        self.bottleneck = ConvBlock(512, 1024)
        self.dec4 = UpBlock(1024, 512)
        self.dec3 = UpBlock(512, 256)
        self.dec2 = UpBlock(256, 128)
        self.dec1 = UpBlock(128, 64)
        self.out_conv = nn.Conv1d(64, out_channels, kernel_size=1)

    def forward(self, x, peak_positions): 
        x = self.initial_conv(x) 
        x = x.permute(2, 1, 0) 
        x = self.positional_encoding(x, peak_positions)
        x = x.permute(1, 2, 0)
        skip1, x = self.enc1(x)
        skip2, x = self.enc2(x)
        skip3, x = self.enc3(x)
        skip4, x = self.enc4(x)
        x = self.bottleneck(x)
        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)
        return self.out_conv(x)

class MHA(nn.Module): 
    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1):
        super(MHA, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads
        self.query_linear = nn.Linear(embed_dim, embed_dim)
        self.key_linear = nn.Linear(embed_dim, embed_dim)
        self.value_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        self.attention_dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None): 
        batch_size, seq_length, _ = query.size() 
        query = self.query_linear(query).view(batch_size, seq_length, self.num_heads, self.head_dim)
        key = self.key_linear(key).view(batch_size, seq_length, self.num_heads, self.head_dim)
        value = self.value_linear(value).view(batch_size, seq_length, self.num_heads, self.head_dim)
        query = query.transpose(1, 2) 
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        attention_output = torch.matmul(attention_weights, value)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embed_dim)
        output = self.out_linear(attention_output)
        return output 

class TransformerDenoiser(nn.Module):
    def __init__(self, num_layers=4, input_dim=1, conv_dim=64, embed_dim=256, 
                 num_heads=8, dim_feedforward=1024, max_len=256, dropout=0.1, **kwargs):
        super(TransformerDenoiser, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, conv_dim, kernel_size=3, padding=1)
        self.embedding = nn.Linear(conv_dim, embed_dim)
        self.pos_encoder = PEwithPeak(embed_dim, max_len=max_len) 
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                 dim_feedforward=dim_feedforward, dropout=dropout, batch_first=False, device=get_device()) # Pass device
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(embed_dim, conv_dim)
        self.conv2 = nn.Conv1d(conv_dim, input_dim, kernel_size=1)

    def forward(self, x, peak_positions): 
        x = self.conv1(x) 
        x = x.permute(2, 0, 1) 
        x = self.embedding(x) 
        x = self.pos_encoder(x, peak_positions) 
        x = self.transformer_encoder(x) 
        x = self.output_layer(x)
        x = x.permute(1, 2, 0) 
        x = self.conv2(x) 
        return x

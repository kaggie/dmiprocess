import math
import torch
import torch.nn as nn


class PEwithPeak(nn.Module):
    """
    Positional encoding with learned peaks for 1D sequences.
    """
    def __init__(self, embed_dim=32, max_len=256, num_peaks=4, **kwargs):
        super(PEwithPeak, self).__init__()
        self.dim = embed_dim

        # Standard positional encoding
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(1000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

        # Learned embeddings for peaks
        self.peak_embedding = nn.Embedding(max_len, embed_dim)

    def forward(self, x, peak_positions):  # x (sequence_length, batch_size, embed_dim)
        x = x + self.pe[:x.size(0), :]
        peak_embed = torch.zeros_like(x)
        seq_len, batch_size, _ = x.shape

        for i in range(batch_size):
            valid_peaks = (peak_positions[i] >= 0) & (peak_positions[i] < seq_len)
            valid_positions = peak_positions[i][valid_peaks]

            if valid_positions.numel() > 0:  # If there are valid peak positions
                embeddings = self.peak_embedding(valid_positions)  # [num_valid_peaks, embedding_dim]
                peak_embed[valid_positions, i, :] = embeddings
        x = x + peak_embed
        return x


class ConvBlock(nn.Module):
    """
    Basic convolutional block: Conv1D + BatchNorm + ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DownBlock(nn.Module):
    """
    Downsampling block for the encoder.
    """
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.conv = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        x = self.conv(x)
        x_down = self.pool(x)
        return x, x_down


class UpBlock(nn.Module):
    """
    Upsampling block for the decoder.
    """
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet1DWithPEPeak(nn.Module):
    """
    1D U-Net with positional encoding and learned peaks.
    """
    def __init__(self, in_channels, out_channels, embed_dim=32, max_len=256):
        super(UNet1DWithPEPeak, self).__init__()
        self.positional_encoding = PEwithPeak(embed_dim, max_len)

        # Encoder
        self.enc1 = DownBlock(embed_dim, 64)
        #self.enc1 = DownBlock(in_channels + embed_dim, 64)
        self.enc2 = DownBlock(64, 128)
        self.enc3 = DownBlock(128, 256)
        self.enc4 = DownBlock(256, 512)

        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024)

        # Decoder
        self.dec4 = UpBlock(1024, 512)
        self.dec3 = UpBlock(512, 256)
        self.dec2 = UpBlock(256, 128)
        self.dec1 = UpBlock(128, 64)

        # Output layer
        self.out_conv = nn.Conv1d(64, out_channels, kernel_size=1)

    def forward(self, x, peak_positions):
        """
        Forward pass with input data and peak positions.
        :param x: Input tensor of shape (batch_size, seq_len, in_channels)
        :param peak_positions: Tensor of peak positions for each batch
        :return: Output tensor of shape (batch_size, out_channels, seq_len)
        """
        batch_size, seq_len, _ = x.size()

        # Positional encoding
        x = x.permute(2, 0, 1)  # (seq_len, batch_size, in_channels)
        #peak_positions = peak_positions.permute(2, 0, 1)
        #print(x.shape,peak_positions.shape)
        x = self.positional_encoding(x, peak_positions)
        x = x.permute(1, 2, 0)  # (batch_size, embed_dim + in_channels, seq_len)

        # Encoder
        skip1, x = self.enc1(x)
        skip2, x = self.enc2(x)
        skip3, x = self.enc3(x)
        skip4, x = self.enc4(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)

        # Output
        x = self.out_conv(x)
        return x


# Example usage
if __name__ == "__main__":
    import numpy as np
    data = np.random.randn(1, 250, 5, 5, 5)  # Example input data
    data  = data.transpose(0, 2, 3, 4, 1) 
    data = np.real(data).reshape(-1, 1, 250)
    pad_width = ((0, 0), (0, 0), (0, 6))
    data = np.pad(data, pad_width, mode='constant', constant_values=0)
    data_tensor = torch.from_numpy(data).to(torch.float32)
    #print(data_tensor.shape)  # Expected: (batch_size, in_channels, seq_len)

    peak_list = "120,130,150,160"  # Example peak positions
    peaks = [int(p.strip()) for p in peak_list.split(',')]
    peaks_tensor = torch.tensor(peaks,dtype=torch.long).unsqueeze(0)
    peaks_tensor = peaks_tensor.expand(data_tensor.shape[0], -1)  # Expand peaks_tensor to match data shape
    
    seq_len = 256
    in_channels = 1
    out_channels = 1
    model = UNet1DWithPEPeak(in_channels, out_channels, embed_dim=256, max_len=seq_len)
    
    output = model(data_tensor, peaks_tensor)
    print(output.shape)  # Expected: (batch_size, out_channels, seq_len)

    '''
    peak_list = params.get("peak_list")
                peaks = [int(p.strip()) for p in peak_list.split(',')]
                peaks_tensor = torch.tensor(peaks,dtype=torch.long).to(DEVICE).unsqueeze(0).unsqueeze(0)
                model = denoise_unet_pe.UNet1DWithPEPeak(in_channels=1, out_channels=1)
                model.to(DEVICE)
                model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                model.eval()
                with torch.no_grad():
                    data  = data.transpose(0, 2, 3, 4, 1) 
                    data = np.real(data).reshape(-1, 1, F)
                    peaks_tensor = peaks_tensor.expand(data.shape[0], -1, -1)  # Expand peaks_tensor to match data shape
                    if F != 256:
                        data = np.pad(data, (0, 256 - F), mode='constant', constant_values=0)
                        data_tensor = torch.from_numpy(data).to(DEVICE).to(torch.float32)
    '''

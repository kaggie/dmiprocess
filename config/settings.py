import torch

# Default device, can be updated by set_device()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_device():
    """Returns the current processing device (torch.device)."""
    return DEVICE

def set_device(new_device_str=None):
    """
    Sets the global processing device.
    Args:
        new_device_str (str, optional): "cuda" or "cpu". 
                                        If None or invalid, auto-detects.
    Returns:
        torch.device: The newly set device.
    """
    global DEVICE
    if new_device_str == "cuda" and torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif new_device_str == "cpu":
        DEVICE = torch.device("cpu")
    else: # Auto-detect or if new_device_str is invalid
        if new_device_str is not None and new_device_str not in ["cuda", "cpu"]:
            print(f"Warning: Invalid device string '{new_device_str}'. Auto-detecting.")
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # It's better to use a proper logger in a real application
    print(f"Device set to: {DEVICE}") 
    return DEVICE

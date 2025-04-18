import torch

# URL to the pre-trained model
# url = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_t_20230227-119e8c9f.pth'
url = "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth"
# Load model state dictionary from URL
checkpoint = torch.hub.load_state_dict_from_url(url, map_location='cpu', check_hash=True)

# Check the keys of the state dictionary
print(checkpoint.keys())



import torch

def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, use_cuda: bool):
  if use_cuda:
    checkpoint = torch.load(checkpoint_path)
  else:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
  
  model.load_state_dict(checkpoint)
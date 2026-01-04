import torch

from model import EMNIST_VGG

model = torch.load("EMNIST_CNN.pth", weights_only=False)

model.eval()



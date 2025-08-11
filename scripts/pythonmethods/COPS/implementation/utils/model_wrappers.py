"""
Model wrappers for feature extraction from large pre-trained vision models.
"""
from abc import abstractmethod, ABC

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T


class ModelWrapper(torch.nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, interpolation, images):
        pass

    @abstractmethod
    def patch_size(self):
        pass


class DINOWrapper(ModelWrapper):
    def __init__(self, device='cpu', small=True, reg=True):
        super().__init__()
        print("Initializing DINOWrapper...")
        self.model_type = "small" if small else "large"
        try:
            if not small:
                if reg:
                    print("Loading large model...")
                    self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg').to(device)
                else:
                    print("Loading large model...")
                    self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14').to(device)
            else:
                if reg:
                    print("Loading small model...")
                    self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg').to(device)
                else:
                    print("Loading small model...")
                    self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
            print("Model loaded.")
        except Exception as e:
            print(f"Error loading model: {e}")
            return
        self.model.eval()

        self.image_transforms = T.Compose([
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def forward(self, images, interpolation=False, raw=False):
        images = self.image_transforms(images.permute(0, 3, 1, 2))
        out = self.model.forward_features(images)
        if raw:
            for key in out:
                print(f"Key: {key}, Shape: {out[key].shape}")
            return out
        if interpolation:
            features = out['x_norm_patchtokens']
            N, num_patches, C = features.shape
            features = torch.permute(features, (0, 2, 1))
            features = features.view(N, C, int(np.sqrt(num_patches)), int(np.sqrt(num_patches)))
            features = torch.nn.functional.interpolate(features, scale_factor=2, mode='bilinear', align_corners=False)
            features = features.view(N, C, -1)
            features = torch.permute(features, (0, 2, 1))
            return features
        else:
            return out['x_norm_patchtokens']

    def patch_size(self):
        return 14
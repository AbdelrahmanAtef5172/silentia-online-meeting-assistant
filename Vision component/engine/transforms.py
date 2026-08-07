"""
engine/transforms.py
────────────────────
Defines all torchvision transform chains.
"""

import torchvision.transforms as T

# ── ImageNet normalization statistics ─────────────────────────────────────────
# These match the timm library defaults for vit_base_patch16_224
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # RGB order # mean pixel values for ImageNet dataset (used for normalization)
IMAGENET_STD  = [0.229, 0.224, 0.225]  # RGB order # standard deviation of pixel values for ImageNet dataset (used for normalization)

# Inference-time transform chain (Face already aligned and resized to 224x224)
def get_inference_transforms() -> T.Compose:
    """
    Inference-time transform chain.
    Input:  (224, 224, 3) float32 numpy array, RGB, values in [0, 1]
            (already aligned and resized by FaceAligner)
    Output: (3, 224, 224) float32 torch.Tensor, ImageNet-normalized
    Note: No augmentation at inference time.
    """
    return T.Compose([
        T.ToTensor(),           # (H,W,C) float32 → (C,H,W) float32 in [0,1]
        T.Normalize(            # Subtract mean, divide by std (per channel)
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])

# Training-time transform chain with augmentation
def get_training_transforms(image_size: int = 224) -> T.Compose:
    """
    Training-time transform chain with augmentation.
    Input:  PIL Image or (H, W, 3) uint8 numpy array
    Output: (3, 224, 224) float32 torch.Tensor, ImageNet-normalized
    Augmentation strategy:
    - RandomHorizontalFlip: Gender is symmetric horizontally (valid augmentation)
    - ColorJitter: Handles lighting variation across different environments
    - RandomRotation(10°): Small rotation robustness
    - RandomAffine(translate=0.05): Minor translation robustness
    """
    return T.Compose([
        T.ToPILImage(),  # Convert numpy array to PIL Image (if input is not already a PIL Image)
        T.Resize((image_size, image_size)), # Resize to the specified image size (default 224x224)
        T.RandomHorizontalFlip(p=0.5),      # Randomly flip the image horizontally with a probability of
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05), # Randomly change the brightness, contrast, saturation, and hue of the image
        T.RandomRotation(degrees=10),  # Randomly rotate the image by up to ±10 degrees
        T.RandomAffine(degrees=0, translate=(0.05, 0.05)), # Randomly translate the image by up to ±5% of its width and height
        T.ToTensor(), # Convert the PIL Image to a PyTorch tensor and scale pixel values to [0, 1]
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), # Normalize the tensor using ImageNet mean and standard deviation
    ])

# Validation/test-time transform (no augmentation, matches inference)
def get_validation_transforms(image_size: int = 224) -> T.Compose:
    """
    Validation/test-time transform (no augmentation, matches inference)
    """
    return T.Compose([
        T.ToPILImage(), # Convert numpy array to PIL Image (if input is not already a PIL Image)
        T.Resize((image_size, image_size)), # Resize to the specified image size (default 224x224)
        T.ToTensor(), # Convert the PIL Image to a PyTorch tensor and scale pixel values to [0, 1]
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), # Normalize the tensor using ImageNet mean and standard deviation
    ])

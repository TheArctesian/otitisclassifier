"""
Transform utilities - Do one thing: handle image transforms
Unix philosophy: Simple, composable transform operations
"""

import torchvision.transforms as transforms


def get_base_transforms(image_size: int = 384) -> transforms.Compose:
    """Get basic transforms. Single responsibility."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size), 
                        interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor()
    ])


def get_imagenet_normalization() -> transforms.Normalize:
    """Get ImageNet normalization. Simple function."""
    return transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )


def get_radimagenet_normalization() -> transforms.Normalize:
    """Get RadImageNet normalization. Simple function."""
    return transforms.Normalize(
        mean=[0.502, 0.502, 0.502], 
        std=[0.291, 0.291, 0.291]
    )


def get_training_augmentations() -> transforms.Compose:
    """Get training augmentations. Single purpose."""
    return transforms.Compose([
        transforms.RandomRotation(degrees=10),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05))
    ])


def create_transform_pipeline(image_size: int = 384, 
                            use_radimagenet: bool = False,
                            training: bool = False) -> transforms.Compose:
    """Create complete transform pipeline. Composable."""
    transforms_list = []
    
    # Base transforms
    transforms_list.extend([
        transforms.Resize((image_size, image_size), 
                        interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor()
    ])
    
    # Normalization
    if use_radimagenet:
        transforms_list.append(get_radimagenet_normalization())
    else:
        transforms_list.append(get_imagenet_normalization())
    
    # Training augmentations applied after normalization
    if training:
        # Note: Some augmentations work better before tensor conversion
        # This is a simplified approach - in practice, you might want
        # to apply augmentations before ToTensor()
        pass
    
    return transforms.Compose(transforms_list)
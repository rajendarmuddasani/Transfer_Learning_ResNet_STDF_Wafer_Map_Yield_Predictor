"""
PyTorch Dataset for Wafer Map Images
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Callable, List, Tuple
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class WaferMapDataset(Dataset):
    """Dataset for wafer map images."""
    
    DEFECT_CLASSES = [
        'Normal',
        'EdgeEffect',
        'CenterCluster',
        'RingPattern',
        'QuadrantFailure',
        'Scratch',
        'RandomFailure',
        'MixedMode'
    ]
    
    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        load_labels: bool = True
    ):
        """
        Initialize wafer map dataset.
        
        Args:
            data_dir: Directory containing wafer map images
            transform: Optional transform to apply to images
            load_labels: Whether to load labels from directory structure or filenames
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.load_labels = load_labels
        
        # Find all image files
        self.image_paths = sorted(list(self.data_dir.glob("**/*.png")))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No PNG images found in {data_dir}")
        
        # Build class to index mapping
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.DEFECT_CLASSES)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        logger.info(
            f"Loaded {len(self.image_paths)} wafer map images from {data_dir}"
        )
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index
            
        Returns:
            (image_tensor, label) tuple
        """
        img_path = self.image_paths[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Extract label from directory structure or filename
        if self.load_labels:
            # Assume directory structure: data_dir/ClassName/image.png
            # OR filename format: WaferID_ClassName.png
            if img_path.parent.name in self.DEFECT_CLASSES:
                label = self.class_to_idx[img_path.parent.name]
            else:
                # Try to extract from filename
                filename = img_path.stem
                label = 0  # Default to Normal
                for cls_name, cls_idx in self.class_to_idx.items():
                    if cls_name.lower() in filename.lower():
                        label = cls_idx
                        break
        else:
            label = -1  # No label
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_distribution(self) -> dict:
        """Get distribution of classes in dataset."""
        class_counts = {cls: 0 for cls in self.DEFECT_CLASSES}
        
        for idx in range(len(self)):
            _, label = self.__getitem__(idx)
            class_name = self.idx_to_class[label]
            class_counts[class_name] += 1
        
        return class_counts


class WaferMapInferenceDataset(Dataset):
    """Dataset for inference (no labels)."""
    
    def __init__(
        self,
        image_paths: List[str],
        transform: Optional[Callable] = None
    ):
        """
        Initialize inference dataset.
        
        Args:
            image_paths: List of image file paths
            transform: Optional transform
        """
        self.image_paths = [Path(p) for p in image_paths]
        self.transform = transform
        
        logger.info(f"Loaded {len(self.image_paths)} images for inference")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Get a sample.
        
        Returns:
            (image_tensor, image_path) tuple
        """
        img_path = self.image_paths[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, str(img_path)


def get_class_weights(dataset: WaferMapDataset) -> torch.Tensor:
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        dataset: WaferMapDataset instance
        
    Returns:
        Tensor of class weights
    """
    class_counts = dataset.get_class_distribution()
    total_samples = sum(class_counts.values())
    
    # Inverse frequency weights
    weights = []
    for cls in dataset.DEFECT_CLASSES:
        count = class_counts[cls]
        if count > 0:
            weight = total_samples / (len(dataset.DEFECT_CLASSES) * count)
        else:
            weight = 1.0
        weights.append(weight)
    
    return torch.tensor(weights, dtype=torch.float32)


if __name__ == "__main__":
    # Test dataset
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = WaferMapDataset(
        data_dir="data/wafer_maps/train",
        transform=transform
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Classes: {dataset.DEFECT_CLASSES}")
    
    # Test sample
    img, label = dataset[0]
    print(f"Image shape: {img.shape}")
    print(f"Label: {label} ({dataset.idx_to_class[label]})")

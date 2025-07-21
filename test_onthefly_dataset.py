#!/usr/bin/env python3
"""
Test script for OnTheFlyMathematicalFunctionDataset
"""

import sys
import os
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Add the pretraining directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'pretraining'))

from onthefly_dataset import OnTheFlyMathematicalFunctionDataset

def test_dataset():
    """Test the on-the-fly dataset generation"""
    print("Testing OnTheFlyMathematicalFunctionDataset...")
    
    # Create transform (similar to training)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Create dataset with small sample size for testing
    dataset = OnTheFlyMathematicalFunctionDataset(
        num_samples=50,
        image_size=224,
        transform=transform
    )
    
    print(f"Dataset length: {len(dataset)}")
    print(f"Function types: {dataset.configs}")
    
    # Test generating a few samples
    for i in range(10):
        try:
            img, label = dataset[i]
            print(f"Sample {i}: Image shape: {img.shape}, Label: {label}, Function type: {dataset.configs[label]}")
            
            # Save first few images for visual inspection
            if i < 5:
                img_pil = transforms.ToPILImage()(img)
                img_pil.save(f"test_sample_{i}_{dataset.configs[label]}.png")
                print(f"Saved test_sample_{i}_{dataset.configs[label]}.png")
                
        except Exception as e:
            print(f"Error generating sample {i}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Test DataLoader
    print("\nTesting with DataLoader...")
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=4, 
        shuffle=True,
        num_workers=0  # Use 0 for testing to avoid multiprocessing issues
    )
    
    try:
        batch_images, batch_labels = next(iter(dataloader))
        print(f"Batch images shape: {batch_images.shape}")
        print(f"Batch labels: {batch_labels}")
        print("DataLoader test successful!")
    except Exception as e:
        print(f"DataLoader test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataset()

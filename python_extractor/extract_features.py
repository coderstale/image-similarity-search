import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
from PIL import Image

MODEL_NAME = "MobileNetV2"
OUTPUT_FILE_PATH = '../c_search/data/vectors.csv'
DATASET_DIR = './dataset'
NUM_IMAGES = 50000
BATCH_SIZE = 64

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def get_model(device):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    
    feature_extractor = torch.nn.Sequential(
        model.features,
        torch.nn.AdaptiveAvgPool2d((1, 1))
    ).to(device)
    
    feature_extractor.eval()
    return feature_extractor

def extract_features(model, dataloader, device):
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting Features"):
            images = images.to(device)
            features = model(images)
            features = features.squeeze(-1).squeeze(-1)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    return np.vstack(all_features), np.concatenate(all_labels)

if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)
    
    device = get_device()
    print(f"Using {str(device).upper()} device.")
    
    model = get_model(device)
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    print("Loading CIFAR-10 dataset...")
    full_dataset = datasets.CIFAR10(root=DATASET_DIR, train=True, download=True, transform=transform)
    
    subset_dataset = torch.utils.data.Subset(full_dataset, range(NUM_IMAGES))
    
    dataloader = DataLoader(subset_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Dataset loaded. Processing {NUM_IMAGES} images.")

    features, labels = extract_features(model, dataloader, device)
    
    print(f"\nSaving {len(features)} feature vectors to {OUTPUT_FILE_PATH}...")
    data_to_save = np.hstack((labels[:, np.newaxis], features))
    np.savetxt(OUTPUT_FILE_PATH, data_to_save, delimiter=',', fmt='%.8f')
    
    print("\n✅ Feature extraction complete!")
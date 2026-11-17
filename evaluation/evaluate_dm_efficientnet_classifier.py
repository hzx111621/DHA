import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


class Config:
    img_size = 256
    batch_size = 64
    dm_models = ['ADM', 'SD1.5', 'VQDM', 'Wukong']
    num_classes = len(dm_models)
    data_root = "./data_root"
    normalize_mean = [0.485, 0.456, 0.406]
    normalize_std = [0.229, 0.224, 0.225]


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_map = {model: idx for idx, model in enumerate(Config.dm_models)}

        for model in Config.dm_models:
            model_dir = os.path.join(root_dir, 'originals', model)
            if not os.path.exists(model_dir):
                continue
            for img_name in os.listdir(model_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(model_dir, img_name), self.class_map[model]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def split_dataset(dataset):
    train_val_idx, test_idx = train_test_split(
        range(len(dataset)),
        test_size=0.1,
        stratify=[label for _, label in dataset.samples],
        random_state=42
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.1 / 0.9,
        stratify=[dataset.samples[i][1] for i in train_val_idx],
        random_state=42
    )
    return (
        torch.utils.data.Subset(dataset, train_idx),
        torch.utils.data.Subset(dataset, val_idx),
        torch.utils.data.Subset(dataset, test_idx)
    )


class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.efficientnet_b4(pretrained=False)
        num_ftrs = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(num_ftrs, Config.num_classes)
        )
        self.model = backbone

    def forward(self, x):
        return self.model(x)


def test_model(model_path='image_classifier_efficientnet_best_DM.pth'):
    transform = transforms.Compose([
        transforms.Resize((Config.img_size, Config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.normalize_mean, std=Config.normalize_std)
    ])

    full_dataset = ImageDataset(root_dir=Config.data_root, transform=transform)
    _, _, test_set = split_dataset(full_dataset)
    test_loader = DataLoader(test_set, batch_size=Config.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImageClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    true_labels = []
    predicted_labels = []
    confidences = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            max_probs, preds = torch.max(probabilities, 1)

            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(preds.cpu().numpy())
            confidences.extend(max_probs.cpu().numpy())

    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=Config.dm_models,
        digits=4
    )
    print("Classification Report:\n", report)

    cm = confusion_matrix(true_labels, predicted_labels)
    print("Confusion Matrix:\n", cm)

    results = []
    for i in range(len(test_set)):
        img_path, true_label = test_set.dataset.samples[test_set.indices[i]]
        results.append({
            "Image": img_path,
            "True Class": Config.dm_models[true_label],
            "Predicted Class": Config.dm_models[predicted_labels[i]],
            "Confidence": confidences[i]
        })

    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    print("Test results saved to test_results.csv")


if __name__ == "__main__":
    test_model('/data/MM_AIGC/image_classifier_efficientnet_best_DM.pth')
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# S4: 粗粒度判为 DM 后，仅针对四种已知 DM 模型分类
# 示例标签：ADM / SD1.5 / Wukong / VQDM

DATA_DIR = './split_20_test_DM'
WEIGHTS_PATH = 'best_modelDM.pth'
BATCH_SIZE = 32
NUM_CLASSES = 4
CLASS_NAMES = ['ADM', 'SD1.5', 'Wukong', 'VQDM']


def build_model(num_classes=4):
    model = models.resnet101(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = build_model(NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()

    correct = 0
    total = 0
    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i].item() == label:
                    class_correct[label] += 1

    print(f"总体准确率: {100 * correct / total:.2f}%")
    for i, name in enumerate(CLASS_NAMES):
        acc = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        print(f"{name} 准确率: {acc:.2f}%")


if __name__ == '__main__':
    evaluate()

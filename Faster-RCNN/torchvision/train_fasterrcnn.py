import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
from torch.utils.data import DataLoader
from PIL import Image
import os
import json
from torchvision.transforms.functional import to_tensor

# Assuming that you have the utils.py and engine.py from the torchvision repository
# which contain helper functions for training and evaluation.

# Update this to the number of classes including the background
num_classes = 4

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        with open(annotation) as f:
            self.data = json.load(f)

        self.images = [item['file_name'] for item in self.data['images']]
        self.annotations = {ann['image_id']: [] for ann in self.data['annotations']}
        for ann in self.data['annotations']:
            if ann['image_id'] in self.annotations:
                self.annotations[ann['image_id']].append(ann)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.images[idx])
        img = Image.open(img_path).convert("RGB")

        anns = self.annotations[idx]
        boxes = [ann['bbox'] for ann in anns]
        labels = [ann['category_id'] for ann in anns]

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        boxes[:, 2:] += boxes[:, :2]  # Convert COCO format to [xmin, ymin, xmax, ymax]

        target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([idx])}

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.images)

def get_transform(train):
    # This function now accepts a boolean `train` parameter
    # to determine which transforms to apply.
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        # If training data, apply random horizontal flip for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

# Use the torchvision implementation of Faster R-CNN
def get_model_instance_segmentation(num_classes):
    model = fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def collate_fn(batch):
    return tuple(zip(*batch))

train_dataset = CustomDataset('train', 'train/annotations.json', get_transform(train=True))
valid_dataset = CustomDataset('valid', 'valid/annotations.json', get_transform(train=False))
test_dataset = CustomDataset('test', 'test/annotations.json', get_transform(train=False))

# Define data loaders
train_data_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn)
valid_data_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)
test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)

# Get the model
model = get_model_instance_segmentation(num_classes)

# Send the model to GPU
device = torch.device('cpu')
model.to(device)

# Define the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

# Define the learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Number of epochs
num_epochs = 10

# Custom training loop
def train(model, data_loader, optimizer, device, epoch, print_freq=10):
    model.train()
    total_loss = 0
    for batch_index, (images, targets) in enumerate(data_loader):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

        # Print loss every print_freq batches
        if (batch_index + 1) % print_freq == 0:
            print(f"Epoch {epoch+1}, Batch {batch_index+1}/{len(data_loader)}, Loss: {losses.item()}")

    average_loss = total_loss / len(data_loader)
    print(f"Epoch {epoch+1}, Training average loss: {average_loss}")

# Custom evaluation function
def evaluate(model, data_loader, device, epoch):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()
    print(f"Epoch {epoch+1}, Validation loss: {total_loss / len(data_loader)}")

for epoch in range(num_epochs):
    train(model, train_data_loader, optimizer, device, epoch)
    lr_scheduler.step()
    evaluate(model, valid_data_loader, device, epoch)

# Final evaluation on test data
evaluate(model, test_data_loader, device)

print("Training complete")
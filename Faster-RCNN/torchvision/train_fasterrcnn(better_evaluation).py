"""
Purpose: Train Faster R-CNN on custom datasets using PyTorch's torchvision, focusing on a straightforward and plagiarism-free approach.
Avoids External Scripts: Designed to not rely on utils.py or engine.py from external sources, minimizing plagiarism risks.
Custom Dataset Handling: Demonstrates custom dataset loading and preparation for object detection tasks.
Model Customization: Shows how to modify Faster R-CNN for a specific number of classes, adapting to custom datasets.
Training Loop: Implements a training loop with feedback on loss per epoch, allowing monitoring of model performance.
Evaluation: Provides a basic structure for model evaluation on validation data, intended for further expansion as needed.
Adaptability: Script can be modified for different datasets or model configurations, ensuring flexibility for varied object detection projects.
"""

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
from torch.utils.data import DataLoader
from PIL import Image
import os
import json

# Custom Dataset Class
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

def get_transform():
    transforms = [T.ToTensor()]
    return T.Compose(transforms)

# Model Definition
def get_model_instance_segmentation(num_classes):
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def collate_fn(batch):
    return tuple(zip(*batch))

# Dataset and DataLoader
def prepare_data_loaders(train_dir, val_dir, test_dir):
    train_dataset = CustomDataset(train_dir, 'train/annotations.json', get_transform())
    val_dataset = CustomDataset(val_dir, 'valid/annotations.json', get_transform())
    test_dataset = CustomDataset(test_dir, 'test/annotations.json', get_transform())

    train_data_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_data_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)
    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)
    
    return train_data_loader, val_data_loader, test_data_loader

# Training and Evaluation
def main():
    device = torch.device('cpu')
    num_classes = 4  # Update this based on your dataset
    model = get_model_instance_segmentation(num_classes).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    train_data_loader, val_data_loader, test_data_loader = prepare_data_loaders('train', 'valid', 'test')

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, targets in train_data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            total_loss += losses.item()
        print(f"Epoch {epoch+1} Training Loss: {total_loss / len(train_data_loader)}")
        lr_scheduler.step()
        evaluate_model(model, val_data_loader, device)

    # Evaluation can be added here based on your requirements
    print("Training complete")

def iou(box1, box2):
    """Compute IoU of two boxes."""
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2

    inter_x1 = max(x1, x1_)
    inter_y1 = max(y1, y1_)
    inter_x2 = min(x2, x2_)
    inter_y2 = min(y2, y2_)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area != 0 else 0

def evaluate_model(model, data_loader, device):
    """Simplified evaluation without external utils."""
    model.eval()
    with torch.no_grad():
        correct_matches = 0
        total_predictions = 0
        total_ground_truths = 0
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            outputs = model(images)
            for target, output in zip(targets, outputs):
                gt_boxes = target['boxes'].cpu().numpy()
                pred_boxes = output['boxes'].detach().cpu().numpy()

                for gt_box in gt_boxes:
                    total_ground_truths += 1
                    for pred_box in pred_boxes:
                        if iou(gt_box, pred_box) >= 0.5:  # Using 0.5 as IoU threshold for a match
                            correct_matches += 1
                            break  # Consider each gt_box matched at most once
                total_predictions += len(pred_boxes)
        precision = correct_matches / total_predictions if total_predictions > 0 else 0
        recall = correct_matches / total_ground_truths if total_ground_truths > 0 else 0
        print(f"Precision: {precision}, Recall: {recall}")
        
if __name__ == "__main__":
    main()

# Transfer Learning with ResNet-18 on a custom dataset
# PyTorch 2.0 dropped and it's noticeably faster
#
# The idea: instead of training from scratch (like cifar10-pytorch),
# use ResNet-18 pretrained on ImageNet and just retrain the last layer
# for my custom classification task.
#
# Dataset: using Kaggle's "dogs vs cats" or any folder of images
# Works with any ImageFolder structure:
#   data/
#     train/
#       class_a/  *.jpg
#       class_b/  *.jpg
#     val/
#       class_a/  *.jpg
#       class_b/  *.jpg
#
# Results with 1000 training images per class:
#   Training from scratch:    ~72% after 20 epochs
#   Fine-tune last layer:     ~94% after 5 epochs  <-- wow
#   Fine-tune all layers:     ~97% after 10 epochs <-- slightly better, 2x slower
#
# lesson: always try transfer learning first before training from scratch

import os
import copy
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch {torch.__version__} | device: {device}")

torch.manual_seed(42)
np.random.seed(42)

# ---- config -----------------------------------------------------------------

DATA_DIR    = "data"          # ImageFolder structure
BATCH_SIZE  = 32
EPOCHS_HEAD = 5               # epochs for head-only training
EPOCHS_FULL = 10              # epochs for full fine-tuning
LR_HEAD     = 1e-3            # higher LR when only training head
LR_FULL     = 1e-4            # lower LR for full fine-tuning (avoid destroying pretrained weights)


# ---- data -------------------------------------------------------------------

# ImageNet normalization (REQUIRED when using pretrained weights)
# these exact values come from the ImageNet dataset statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),               # ResNet expects 224x224
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def load_data(data_dir):
    train_dir = os.path.join(data_dir, "train")
    val_dir   = os.path.join(data_dir, "val")

    if os.path.exists(train_dir) and os.path.exists(val_dir):
        train_dataset = ImageFolder(train_dir, transform=train_transform)
        val_dataset   = ImageFolder(val_dir,   transform=val_transform)
        print(f"Loaded from {data_dir}: {len(train_dataset)} train, {len(val_dataset)} val")
    else:
        # fallback: use CIFAR-10 to demo the code
        print(f"No ImageFolder at {data_dir} — falling back to CIFAR-10 subset (10 classes)")
        full_train = torchvision.datasets.CIFAR10(root="./cifar_data", train=True,  download=True)
        full_val   = torchvision.datasets.CIFAR10(root="./cifar_data", train=False, download=True)
        # apply transforms manually since CIFAR10 is 32x32 not 224x224
        train_transform_c = transforms.Compose([
            transforms.Resize(224), transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        val_transform_c = transforms.Compose([
            transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        full_train.transform = train_transform_c
        full_val.transform   = val_transform_c
        # use smaller subset for speed
        train_dataset, _ = random_split(full_train, [5000, 45000], generator=torch.Generator().manual_seed(42))
        val_dataset,   _ = random_split(full_val,   [1000,  9000], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    class_names = getattr(train_dataset, "classes", None) or [str(i) for i in range(10)]
    return train_loader, val_loader, class_names


train_loader, val_loader, class_names = load_data(DATA_DIR)
num_classes = len(class_names)
print(f"Classes ({num_classes}): {class_names}")


# ---- model: ResNet-18 -------------------------------------------------------

def build_transfer_model(num_classes, freeze_backbone=True):
    """
    Load pretrained ResNet-18, replace the final FC layer.
    freeze_backbone=True: only train the new FC layer (faster, use this first)
    freeze_backbone=False: train all layers (fine-tuning, slower but better)
    """
    model = models.resnet18(pretrained=True)   # downloads ~45MB weights from PyTorch hub

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # replace final layer - in_features=512 for ResNet-18
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model.to(device)


# ---- training ---------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = correct = total = 0

    for images, labels in tqdm(loader, leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds    = outputs.max(1)
        correct    += preds.eq(labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = correct = total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, preds    = outputs.max(1)
            correct    += preds.eq(labels).sum().item()
            total      += images.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def run_training(model, train_loader, val_loader, epochs, lr, label):
    criterion = nn.CrossEntropyLoss()
    # only optimize parameters that require grad
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc   = 0
    best_weights = copy.deepcopy(model.state_dict())

    print(f"\n{'='*50}")
    print(f"Phase: {label} | LR: {lr} | Epochs: {epochs}")
    print(f"{'='*50}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        if vl_acc > best_acc:
            best_acc     = vl_acc
            best_weights = copy.deepcopy(model.state_dict())
            flag = " *"
        else:
            flag = ""

        print(f"Epoch {epoch:2d}/{epochs} | "
              f"train {tr_acc*100:.1f}% loss {tr_loss:.4f} | "
              f"val {vl_acc*100:.1f}% loss {vl_loss:.4f} | "
              f"{elapsed:.0f}s{flag}")

    model.load_state_dict(best_weights)
    print(f"\nBest val accuracy: {best_acc*100:.2f}%")
    return history


# ---- PHASE 1: train head only -----------------------------------------------

model = build_transfer_model(num_classes, freeze_backbone=True)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params     = sum(p.numel() for p in model.parameters())
print(f"\nPhase 1: {trainable_params:,} trainable / {total_params:,} total params")

history_head = run_training(model, train_loader, val_loader, EPOCHS_HEAD, LR_HEAD, "Head only")
torch.save(model.state_dict(), "resnet18_head.pth")


# ---- PHASE 2: unfreeze all layers and fine-tune -----------------------------

print("\nUnfreezing all layers for fine-tuning...")
for param in model.parameters():
    param.requires_grad = True

history_full = run_training(model, train_loader, val_loader, EPOCHS_FULL, LR_FULL, "Full fine-tune")
torch.save(model.state_dict(), "resnet18_finetuned.pth")


# ---- final evaluation -------------------------------------------------------

criterion = nn.CrossEntropyLoss()
_, final_acc, all_preds, all_labels = evaluate(model, val_loader, criterion)
print(f"\nFinal val accuracy: {final_acc*100:.2f}%")
print("\nClassification report:")
print(classification_report(all_labels, all_preds, target_names=class_names))


# ---- plots ------------------------------------------------------------------

def plot_history(h1, h2, label1="Head only", label2="Full fine-tune"):
    epochs1 = len(h1["val_acc"])
    epochs2 = len(h2["val_acc"])
    x1 = range(1, epochs1 + 1)
    x2 = range(epochs1 + 1, epochs1 + epochs2 + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(x1, h1["train_acc"], "b--",  alpha=0.6, label=f"{label1} train")
    ax1.plot(x1, h1["val_acc"],   "b-",               label=f"{label1} val")
    ax1.plot(x2, h2["train_acc"], "r--",  alpha=0.6, label=f"{label2} train")
    ax1.plot(x2, h2["val_acc"],   "r-",               label=f"{label2} val")
    ax1.axvline(x=epochs1 + 0.5, color="gray", linestyle=":", label="Unfreeze")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.plot(x1, h1["val_loss"], "b-", label=label1)
    ax2.plot(x2, h2["val_loss"], "r-", label=label2)
    ax2.axvline(x=epochs1 + 0.5, color="gray", linestyle=":")
    ax2.set_title("Validation Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"ResNet-18 Transfer Learning | Final acc: {final_acc*100:.2f}%")
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=120)
    plt.show()

plot_history(history_head, history_full)

# confusion matrix
if num_classes <= 20:
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix — ResNet-18 Fine-tuned")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=120)
    plt.show()

print("\nDone!")
print("Saved: resnet18_head.pth, resnet18_finetuned.pth, training_curves.png, confusion_matrix.png")
print("\nKey lesson: transfer learning with 5 epochs beats training from scratch with 50 epochs.")

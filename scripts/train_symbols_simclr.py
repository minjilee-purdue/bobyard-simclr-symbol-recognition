import os
import math
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
from PIL import Image
import matplotlib.pyplot as plt


class SimCLRTransform:
    def __init__(self, image_size=96):
        self.base_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __call__(self, x):
        xi = self.base_transform(x)
        xj = self.base_transform(x)
        return xi, xj


class ContrastiveImageFolder(Dataset):
    def __init__(self, root, transform):
        self.dataset = datasets.ImageFolder(root=root)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        path, label = self.dataset.samples[idx]
        image = self.dataset.loader(path)
        xi, xj = self.transform(image)
        return xi, xj, label


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class SimCLRNet(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.encoder = backbone
        self.projector = ProjectionHead(feature_dim, hidden_dim=512, out_dim=out_dim)

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        z = F.normalize(z, dim=1)
        return h, z


def nt_xent_loss(z1, z2, temperature=0.5):
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    similarity = torch.matmul(z, z.T) / temperature

    mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    similarity = similarity.masked_fill(mask, -9e15)

    positive_indices = torch.arange(batch_size, device=z.device)
    positive_indices = torch.cat([positive_indices + batch_size, positive_indices], dim=0)

    loss = F.cross_entropy(similarity, positive_indices)
    return loss


def save_loss_plot(losses, save_path):
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(losses) + 1), losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("SimCLR Training Loss")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    train_dir = Path(args.data_root) / "train"
    assert train_dir.exists(), f"Train directory not found: {train_dir}"

    transform = SimCLRTransform(image_size=args.image_size)
    train_dataset = ContrastiveImageFolder(root=str(train_dir), transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    model = SimCLRNet(out_dim=args.proj_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    epoch_losses = []

    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for step, (xi, xj, _) in enumerate(train_loader):
            xi = xi.to(device, non_blocking=True)
            xj = xj.to(device, non_blocking=True)

            _, zi = model(xi)
            _, zj = model(xj)

            loss = nt_xent_loss(zi, zj, temperature=args.temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (step + 1) % args.log_every == 0:
                print(
                    f"Epoch [{epoch+1}/{args.epochs}] "
                    f"Step [{step+1}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = running_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{args.epochs}] Average Loss: {avg_loss:.4f}")

        checkpoint_path = os.path.join(args.output_dir, f"simclr_epoch_{epoch+1}.pth")
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "args": vars(args),
            },
            checkpoint_path,
        )

    final_ckpt = os.path.join(args.output_dir, "simclr_final.pth")
    torch.save(model.state_dict(), final_ckpt)
    print("Saved final model to:", final_ckpt)

    plot_path = os.path.join(args.output_dir, "train_loss.png")
    save_loss_plot(epoch_losses, plot_path)
    print("Saved loss plot to:", plot_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/minjilee/Downloads/Firefighting Device Detection.v6i.yolov8/cropped_symbols",
    )
    parser.add_argument("--output_dir", type=str, default="./outputs_symbols")
    parser.add_argument("--image_size", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=10)
    args = parser.parse_args()

    train(args)
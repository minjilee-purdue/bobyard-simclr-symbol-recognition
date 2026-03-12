import os
import random
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.manifold import TSNE


class EvalTransform:
    def __init__(self, image_size=96):
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __call__(self, x):
        return self.transform(x)


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
        backbone = models.resnet18(weights=None)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.encoder = backbone
        self.projector = ProjectionHead(feature_dim, hidden_dim=512, out_dim=out_dim)

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        z = F.normalize(z, dim=1)
        return h, z


def build_support_set(imagefolder_dataset, seed=42):
    random.seed(seed)
    class_to_indices = defaultdict(list)

    for idx, (_, label) in enumerate(imagefolder_dataset.samples):
        class_to_indices[label].append(idx)

    support_indices = {}
    for label, indices in class_to_indices.items():
        support_indices[label] = random.choice(indices)

    return support_indices


def load_image(path, transform, device):
    image = Image.open(path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    return tensor


@torch.no_grad()
def extract_embedding(model, image_tensor):
    _, z = model(image_tensor)
    return z.squeeze(0)


@torch.no_grad()
def extract_dataset_embeddings(model, dataset, transform, device):
    embeddings = []
    labels = []
    paths = []

    for path, label in dataset.samples:
        image_tensor = load_image(path, transform, device)
        emb = extract_embedding(model, image_tensor).cpu()
        embeddings.append(emb)
        labels.append(label)
        paths.append(path)

    embeddings = torch.stack(embeddings, dim=0)
    labels = np.array(labels)
    return embeddings, labels, paths


def one_shot_predict(test_embeddings, support_embeddings, support_labels):
    sim = F.cosine_similarity(
        test_embeddings.unsqueeze(1),
        support_embeddings.unsqueeze(0),
        dim=2
    )
    best_idx = sim.argmax(dim=1).cpu().numpy()
    preds = np.array([support_labels[i] for i in best_idx])
    return preds, sim.cpu().numpy()


def save_confusion_matrix(y_true, y_pred, class_names, save_path, max_classes=20):
    unique_labels = sorted(list(set(y_true.tolist()) | set(y_pred.tolist())))

    if len(unique_labels) > max_classes:
        selected = unique_labels[:max_classes]
        keep_mask = np.isin(y_true, selected) & np.isin(y_pred, selected)
        y_true_plot = y_true[keep_mask]
        y_pred_plot = y_pred[keep_mask]
        plot_names = [class_names[i] for i in selected]
        labels_for_cm = selected
    else:
        y_true_plot = y_true
        y_pred_plot = y_pred
        plot_names = [class_names[i] for i in unique_labels]
        labels_for_cm = unique_labels

    cm = confusion_matrix(y_true_plot, y_pred_plot, labels=labels_for_cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=plot_names)

    fig, ax = plt.subplots(figsize=(12, 12))
    disp.plot(ax=ax, xticks_rotation=90, colorbar=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_tsne_plot(embeddings, labels, class_names, save_path, max_points=1000, seed=42):
    np.random.seed(seed)

    X = embeddings.cpu().numpy()
    y = labels

    if len(X) > max_points:
        idx = np.random.choice(len(X), size=max_points, replace=False)
        X = X[idx]
        y = y[idx]

    tsne = TSNE(n_components=2, random_state=seed, perplexity=30)
    X_2d = tsne.fit_transform(X)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, s=10, alpha=0.8, cmap="tab20")
    plt.title("t-SNE of Test Embeddings")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_retrieval_examples(
    test_paths,
    test_labels,
    preds,
    similarity_matrix,
    support_paths,
    support_labels,
    class_names,
    save_dir,
    num_examples=10
):
    os.makedirs(save_dir, exist_ok=True)

    chosen = list(range(min(num_examples, len(test_paths))))

    for i, idx in enumerate(chosen):
        query_path = test_paths[idx]
        true_label = test_labels[idx]
        pred_label = preds[idx]

        sim_row = similarity_matrix[idx]
        best_support_idx = int(np.argmax(sim_row))
        matched_support_path = support_paths[best_support_idx]

        fig, axes = plt.subplots(1, 2, figsize=(6, 3))

        query_img = Image.open(query_path).convert("RGB")
        match_img = Image.open(matched_support_path).convert("RGB")

        axes[0].imshow(query_img)
        axes[0].set_title(f"Query\nTrue: {class_names[true_label]}")
        axes[0].axis("off")

        axes[1].imshow(match_img)
        axes[1].set_title(f"Match\nPred: {class_names[pred_label]}")
        axes[1].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"retrieval_{i:02d}.png"), dpi=200)
        plt.close()


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    data_root = Path(args.data_root)
    train_dir = data_root / "train"
    test_dir = data_root / args.eval_split

    transform = EvalTransform(image_size=args.image_size)

    train_dataset = datasets.ImageFolder(root=str(train_dir))
    test_dataset = datasets.ImageFolder(root=str(test_dir))

    class_names = train_dataset.classes
    print("Number of classes:", len(class_names))
    print("Train samples:", len(train_dataset))
    print(f"{args.eval_split} samples:", len(test_dataset))

    model = SimCLRNet(out_dim=args.proj_dim).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    support_indices = build_support_set(train_dataset, seed=args.seed)

    support_embeddings = []
    support_labels = []
    support_paths = []

    for label, idx in sorted(support_indices.items()):
        path, _ = train_dataset.samples[idx]
        image_tensor = load_image(path, transform, device)
        emb = extract_embedding(model, image_tensor).cpu()
        support_embeddings.append(emb)
        support_labels.append(label)
        support_paths.append(path)

    support_embeddings = torch.stack(support_embeddings, dim=0)

    test_embeddings, test_labels, test_paths = extract_dataset_embeddings(
        model, test_dataset, transform, device
    )

    preds, similarity_matrix = one_shot_predict(
        test_embeddings, support_embeddings, support_labels
    )

    accuracy = (preds == test_labels).mean()
    print(f"One-shot accuracy on {args.eval_split}: {accuracy:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)

    report_path = os.path.join(args.output_dir, f"classification_report_{args.eval_split}.txt")
    with open(report_path, "w") as f:
        f.write(f"One-shot accuracy: {accuracy:.4f}\n\n")
        #f.write(classification_report(test_labels, preds, target_names=class_names, zero_division=0))
        f.write(classification_report(test_labels, preds, labels=range(len(class_names)), target_names=class_names, zero_division=0))
    cm_path = os.path.join(args.output_dir, f"confusion_matrix_{args.eval_split}.png")
    save_confusion_matrix(test_labels, preds, class_names, cm_path)

    tsne_path = os.path.join(args.output_dir, f"tsne_{args.eval_split}.png")
    save_tsne_plot(test_embeddings, test_labels, class_names, tsne_path)

    retrieval_dir = os.path.join(args.output_dir, f"retrieval_examples_{args.eval_split}")
    save_retrieval_examples(
        test_paths=test_paths,
        test_labels=test_labels,
        preds=preds,
        similarity_matrix=similarity_matrix,
        support_paths=support_paths,
        support_labels=support_labels,
        class_names=class_names,
        save_dir=retrieval_dir,
        num_examples=args.num_retrieval_examples,
    )

    print("Saved report to:", report_path)
    print("Saved confusion matrix to:", cm_path)
    print("Saved t-SNE plot to:", tsne_path)
    print("Saved retrieval examples to:", retrieval_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/minjilee/Downloads/Firefighting Device Detection.v6i.yolov8/cropped_symbols",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./outputs_symbols/simclr_final.pth",
    )
    parser.add_argument("--output_dir", type=str, default="./eval_outputs")
    parser.add_argument("--eval_split", type=str, default="test", choices=["valid", "test"])
    parser.add_argument("--image_size", type=int, default=96)
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--num_retrieval_examples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(args)
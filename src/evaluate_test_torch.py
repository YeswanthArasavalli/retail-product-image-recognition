from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

MODEL_PATH = Path("models_torch/best_model_efficientnet_b0.pth")
DATA_DIR = Path("data_cls/test")
OUT_DIR = Path("eval_outputs")
OUT_DIR.mkdir(exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 16


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    ckpt = torch.load(MODEL_PATH, map_location=device)
    class_names = ckpt["class_names"]
    num_classes = len(class_names)

    # rebuild model
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    tfms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    if not DATA_DIR.exists():
        print("Test directory not found:", DATA_DIR)
        return

    test_ds = datasets.ImageFolder(DATA_DIR, transform=tfms)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    all_preds, all_labels, all_files, all_probs = [], [], [], []
    top5_correct = 0
    total = 0

    print("Running inference on test set...")
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(test_loader)):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            confs, preds = torch.max(probs, dim=1)

            # top-5 accuracy
            top5 = torch.topk(probs, k=5, dim=1).indices
            top5_correct += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()

            batch_size = labels.size(0)
            start_idx = batch_idx * BATCH_SIZE
            end_idx = start_idx + batch_size
            batch_files = [p for (p, _) in test_ds.samples[start_idx:end_idx]]

            all_files.extend(batch_files)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(confs.cpu().numpy().tolist())
            total += batch_size

    # numeric metrics
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    top5_acc = top5_correct / total

    print(f"Test accuracy: {acc:.4f}")
    print(f"Test top-5 accuracy: {top5_acc:.4f}")

    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    # save metrics
    metrics = {
        "accuracy": acc,
        "top5_accuracy": top5_acc,
        "n_samples": int(total),
    }
    pd.Series(metrics).to_json(OUT_DIR / "metrics_summary.json", indent=2)
    pd.DataFrame(report).to_json(OUT_DIR / "classification_report.json", indent=2)

    # confusion matrix CSV (for Tableau / analysis)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(OUT_DIR / "confusion_matrix.csv")

    # detailed predictions CSV (perfect for Tableau)
    df_pred = pd.DataFrame({
        "filename": all_files,
        "true_label_idx": all_labels,
        "true_label": [class_names[i] for i in all_labels],
        "pred_label_idx": all_preds,
        "pred_label": [class_names[i] for i in all_preds],
        "confidence": all_probs,
    })
    df_pred["correct"] = df_pred["true_label"] == df_pred["pred_label"]
    df_pred.to_csv(OUT_DIR / "test_predictions_detailed.csv", index=False)

    print("✅ Saved:")
    print("  - metrics_summary.json")
    print("  - classification_report.json")
    print("  - confusion_matrix.csv")
    print("  - test_predictions_detailed.csv  (Tableau-ready)")


if __name__ == "__main__":
    main()

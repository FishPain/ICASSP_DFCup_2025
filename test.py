import os
import tempfile
import torch
import numpy as np
# Set temporary file directory to prevent OOM on the OS drive.
temp_path = os.path.join("/workspace/", "tmp")
os.makedirs(temp_path, exist_ok=True)
os.environ.update({
    "TMPDIR": temp_path,
    "TEMP": temp_path,
    "TORCH_EXTENSIONS_DIR": temp_path,
    "TORCH_HOME": temp_path,
    "HF_HOME": temp_path,
    "TOKENIZERS_PARALLELISM": "True",
    "CUDA_VISIBLE_DEVICES": "0",
})

tempfile.tempdir = temp_path
torch.hub.set_dir(temp_path)

from torch.utils.data import DataLoader
from tqdm import tqdm
# Importing components from training script
from train import CustomModel, ResnetModel, ImagePathDataset, compute_dcf

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Testing function
def test_model(model):
    model.eval()  # Set the model to evaluation mode

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(model.valid_loader, desc="Evaluating Model"):
            images = images
            labels = labels.float()
            outputs = model(images).squeeze()
            outputs = torch.sigmoid(outputs)

            # Apply threshold for binary classification
            predicted = (outputs >= 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            y_true.append(labels.cpu())
            y_pred.append(outputs.cpu())

    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)

    # Calculate and print accuracy
    accuracy = 100 * correct / total
    dcf = compute_dcf(y_true, y_pred)

    print(f'Model Accuracy: {accuracy:.2f}%')
    print(f"Detection Cost Function (DCF): {dcf:.4f}")

    return accuracy, dcf


if __name__ == "__main__":
    # Paths and settings
    model_path = "/workspace/models/best_model.pth"
    batch_size = 64

    # Load model
    print("Loading model...")
    backbone = ResnetModel()
    model = CustomModel(backbone_model=backbone).to(device)
    model.load_state_dict(torch.load(model_path))
    model.load_data(batch_size)

    # Evaluate the model
    print("Starting testing...")
    test_accuracy, test_dcf = test_model(model)

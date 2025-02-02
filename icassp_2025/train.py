import os
import tempfile
import torch

# Set temporary file directory to prevent OOM on the OS drive.
temp_path = os.path.join("/workspace/", "tmp")
os.makedirs(temp_path, exist_ok=True)
os.environ.update(
    {
        "TMPDIR": temp_path,
        "TEMP": temp_path,
        "TORCH_EXTENSIONS_DIR": temp_path,
        "TORCH_HOME": temp_path,
        "HF_HOME": temp_path,
        "TOKENIZERS_PARALLELISM": "True",
        "CUDA_VISIBLE_DEVICES": "0",
    }
)

tempfile.tempdir = temp_path
torch.hub.set_dir(temp_path)
import sys
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from model import *
from utils import set_seed, EarlyStopping
from score import compute_dcf
from dataset import ImagePathDataset

# Include libraries in the parent directory.
module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

# Check if CUDA is available.
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        device_name = torch.cuda.get_device_name(i)
        print(f"GPU {i}: {device_name}")
else:
    print("CUDA is not available. Using CPU.")

device = "cuda" if torch.cuda.is_available() else "cpu"


# Custom model integrating multiple components.
class CustomModel(nn.Module):
    def __init__(self, backbone_model):
        super(CustomModel, self).__init__()
        self.backbone = backbone_model
        self.prompter = QwenExtractor().to(device)
        self.extractor = CLIPExtractor().to(device)
        self.classifier = CustomClassifier().to(device)
        self.output_file = "/workspace/data/embeddings.npy"

    def forward(self, x, epoch, loaded_embeddings, use_custom_embeddings):
        prompt_text, prompted_imgs = self.prompter(
            x, epoch, loaded_embeddings, use_custom_embeddings
        )

        img_emb, text_emb, processed_imgs = self.extractor(prompt_text, prompted_imgs)

        bb_embed = self.backbone(processed_imgs)

        # norm_embed = self.feature_normaliser(all_embed)
        return self.classifier(bb_embed, img_emb, text_emb)

    def load_data(self, batch_size):
        train_dataset = ImagePathDataset("/workspace/data/train")
        valid_dataset = ImagePathDataset("/workspace/data/valid")
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )
        self.valid_loader = DataLoader(
            valid_dataset, batch_size=batch_size, shuffle=False, num_workers=1
        )

    def start_training(
        self,
        criterion,
        optimizer,
        scheduler,
        early_stopping,
        num_epochs,
        use_custom_embeddings=False,
    ):
        train_losses, val_losses = [], []

        for epoch in range(num_epochs):
            self.train()
            # load embeddings from either the embeddings generated in epoch 0 or the cusom embeddings saved
            if epoch > 0 or use_custom_embeddings:
                loaded_embeddings = np.load(self.output_file, allow_pickle=True).item()
            else:
                loaded_embeddings = {}
            running_loss = 0.0
            with tqdm(
                self.train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", unit="batch"
            ) as pbar:
                for images, labels in pbar:
                    labels = labels.float().unsqueeze(1).to(device)
                    outputs = self(
                        images, epoch, loaded_embeddings, use_custom_embeddings
                    )
                    loss = criterion(outputs, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    pbar.set_postfix(
                        {"Train Loss": f"{running_loss / (pbar.n + 1):.4f}"}
                    )

            epoch_train_loss = running_loss / len(self.train_loader)
            train_losses.append(epoch_train_loss)

            self.eval()
            val_running_loss = 0.0
            correct, total = 0, 0
            y_true, y_pred = [], []
            with torch.no_grad():
                with tqdm(
                    self.valid_loader, desc="Validation", unit="batch", leave=False
                ) as val_pbar:
                    for images, labels in val_pbar:
                        labels = labels.float().to(device)
                        outputs = self(
                            images, epoch, loaded_embeddings, use_custom_embeddings
                        ).squeeze()

                        val_loss_batch = criterion(outputs, labels)
                        val_running_loss += val_loss_batch.item()

                        outputs = torch.sigmoid(outputs)
                        predicted = (outputs >= 0.5).float()
                        correct += (predicted == labels).sum().item()
                        total += labels.size(0)

                        y_true.append(labels.cpu())
                        y_pred.append(outputs.cpu())

            epoch_val_loss = val_running_loss / len(self.valid_loader)
            val_losses.append(epoch_val_loss)

            y_true, y_pred = torch.cat(y_true), torch.cat(y_pred)
            dcf = compute_dcf(y_true, y_pred)
            val_accuracy = 100 * correct / total

            # save the embeddings in the first epoch when no custom embeddings are provided
            if epoch == 0 and not use_custom_embeddings:
                np.save(self.output_file, loaded_embeddings)

            scheduler.step(dcf)
            early_stopping(dcf, self)

            print(
                f"Validation DCF: {dcf:.4f} | Acc: {val_accuracy:.2f} | lr: {scheduler._last_lr}\n"
            )

            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        return train_losses, val_losses


# Training configuration.
if __name__ == "__main__":
    batch_size = 64
    num_epochs = 30
    seed = 42
    lr = 1e-4
    lr_patience = 5
    lr_factor = 0.1
    es_patience = 10
    set_seed(seed)

    model = CustomModel(backbone_model=ResnetModel()).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_factor, patience=lr_patience, verbose=True
    )
    early_stopping = EarlyStopping(patience=es_patience, verbose=True)

    model.load_data(batch_size)
    train_losses, val_losses = model.start_training(
        criterion,
        optimizer,
        scheduler,
        early_stopping,
        num_epochs,
        use_custom_embeddings=False,
    )

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid()
    plt.show()

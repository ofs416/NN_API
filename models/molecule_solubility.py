import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem

import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple


class MoleculeDataset(Dataset):
    def __init__(self, csv_file: str, smiles_col="SMILES", target_col="Solubility"):
        self.data = pd.read_csv(csv_file)

        # Pre-compute fingerprints during initialization
        print("Computing molecular fingerprints...")
        self.features = []
        self.targets = []

        for idx in range(len(self.data)):
            smiles = self.data.iloc[idx][smiles_col]
            # Handle ionic compounds
            if "." in smiles:
                parts = smiles.split(".")
                smiles = max(parts, key=len)

            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                # Generate Morgan fingerprint
                morgan_gen = Chem.rdFingerprintGenerator.GetMorganGenerator(radius=2)
                # fp = morgan_gen.GetFingerprint(mol)
                features = morgan_gen.GetFingerprintAsNumPy(mol)
                # features = np.array(list(fp.ToBitString())).astype(np.float32)
                self.features.append(features)
                self.targets.append(self.data.iloc[idx][target_col])
            else:
                print(f"Warning: Could not process SMILES {smiles}")

        self.features = np.array(self.features)
        self.targets = np.array(self.targets)
        print(f"Processed {len(self.features)} molecules successfully")

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        return torch.FloatTensor(self.features[idx]), torch.tensor(
            self.targets[idx]
        ).float()


class SolubilityPredictor(nn.Module):
    def __init__(self, input_size: int = 2048):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x) -> torch.Tensor:
        return self.model(x)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.val_r2_scores: List[float] = []

    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss: float = 0

        for batch_idx, (features, targets) in enumerate(train_loader):
            features, targets = features.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(features).squeeze()
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss: float = 0
        all_preds: List[float] = []
        all_targets: List[float] = []

        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(self.device), targets.to(self.device)
                outputs = self.model(features).squeeze()
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        val_loss = total_loss / len(val_loader)
        r2_score = self.calculate_r2(torch.tensor(all_preds), torch.tensor(all_targets))

        return val_loss, r2_score

    @staticmethod
    def calculate_r2(pred: torch.Tensor, true: torch.Tensor) -> float:
        ss_tot = torch.sum((true - true.mean()) ** 2)
        ss_res = torch.sum((true - pred) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2.item()

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> None:
        best_val_loss: float = float("inf")
        patience: int = 10
        patience_counter: int = 0

        for epoch in tqdm.tqdm(range(epochs)):
            train_loss = self.train_epoch(train_loader)
            val_loss, r2_score = self.validate(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_r2_scores.append(r2_score)

            if scheduler is not None:
                scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), "best_model.pt")
            else:
                patience_counter += 1

            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"R² Score: {r2_score:.4f}")

            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    def plot_training_history(self) -> None:
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training and Validation Loss")

        plt.subplot(1, 2, 2)
        plt.plot(self.val_r2_scores)
        plt.xlabel("Epoch")
        plt.ylabel("R² Score")
        plt.title("Validation R² Score")

        plt.tight_layout()
        plt.show()


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dataset
    dataset = MoleculeDataset("curated-solubility-dataset.csv")

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    # Initialize model, criterion, optimizer
    model = SolubilityPredictor()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    # Create trainer and train
    trainer = Trainer(model, criterion, optimizer, device)
    trainer.train(train_loader, val_loader, epochs=100, scheduler=scheduler)

    # Plot training history
    trainer.plot_training_history()

    model_scripted = torch.jit.script(model)  # Export to TorchScript
    model_scripted.save("model_scripted.pt")  # Save


if __name__ == "__main__":
    main()

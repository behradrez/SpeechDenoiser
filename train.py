from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from model import DenoiserCNN
from preprocessing import preprocess_wav_for_training


class SpeechDenoisingDataset(Dataset):
    def __init__(
        self,
        noisy_dir: Path,
        clean_dir: Path,
        target_sr: int = 16000,
        chunk_seconds: float = 0.5,
        hop_seconds: float | None = 0.25,
        n_fft: int = 512,
        hop_length: int = 128,
    ) -> None:
        self.pairs: list[tuple[torch.Tensor, torch.Tensor]] = []

        noisy_files = sorted(noisy_dir.glob("*.wav"))
        if not noisy_files:
            raise FileNotFoundError(f"No wav files found in {noisy_dir}")

        for noisy_path in noisy_files:
            clean_path = clean_dir / noisy_path.name
            if not clean_path.exists():
                continue

            noisy_feats = preprocess_wav_for_training(
                noisy_path,
                target_sr=target_sr,
                chunk_seconds=chunk_seconds,
                hop_seconds=hop_seconds,
                n_fft=n_fft,
                hop_length=hop_length,
            )
            clean_feats = preprocess_wav_for_training(
                clean_path,
                target_sr=target_sr,
                chunk_seconds=chunk_seconds,
                hop_seconds=hop_seconds,
                n_fft=n_fft,
                hop_length=hop_length,
            )

            num_chunks = min(noisy_feats.size(0), clean_feats.size(0))
            for idx in range(num_chunks):
                self.pairs.append((noisy_feats[idx], clean_feats[idx]))

        if not self.pairs:
            raise ValueError("No paired noisy/clean chunks were loaded.")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        noisy, clean = self.pairs[idx]
        return noisy.unsqueeze(0), clean.unsqueeze(0)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for noisy, clean in loader:
        noisy = noisy.to(device)
        clean = clean.to(device)

        optimizer.zero_grad(set_to_none=True)
        output = model(noisy)
        loss = criterion(output, clean)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * noisy.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for noisy, clean in loader:
            noisy = noisy.to(device)
            clean = clean.to(device)
            output = model(noisy)
            loss = criterion(output, clean)
            total_loss += loss.item() * noisy.size(0)
    return total_loss / len(loader.dataset)


def run_training(
    data_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    test_split: float,
    target_sr: int,
    chunk_seconds: float,
    hop_seconds: float | None,
    n_fft: int,
    hop_length: int,
) -> None:
    noisy_dir = data_dir / "noisy"
    clean_dir = data_dir / "clean"

    dataset = SpeechDenoisingDataset(
        noisy_dir=noisy_dir,
        clean_dir=clean_dir,
        target_sr=target_sr,
        chunk_seconds=chunk_seconds,
        hop_seconds=hop_seconds,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    test_size = max(1, int(len(dataset) * test_split))
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenoiserCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_loss = evaluate(model, test_loader, criterion, device)
        print(f"Epoch {epoch}/{epochs} - train loss: {train_loss:.6f} - test loss: {test_loss:.6f}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and test a speech denoising model.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Path to data directory.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--target-sr", type=int, default=16000)
    parser.add_argument("--chunk-seconds", type=float, default=0.5)
    parser.add_argument("--hop-seconds", type=float, default=0.25)
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop-length", type=int, default=128)
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    run_training(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        test_split=args.test_split,
        target_sr=args.target_sr,
        chunk_seconds=args.chunk_seconds,
        hop_seconds=args.hop_seconds,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
    )


if __name__ == "__main__":
    main()

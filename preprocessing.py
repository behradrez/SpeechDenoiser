from pathlib import Path
import torch
import torchaudio


def load_wav_mono_16k(path: str | Path, target_sr: int = 16000) -> tuple[torch.Tensor, int]:
    """
    Returns:
      wav: Tensor shape (num_samples,) float32 in [-1, 1] (approximately)
      sr:  int sample rate (target_sr)
    """
    wav, sr = torchaudio.load(str(path))  # wav shape: (channels, samples)

    # Convert to mono
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=target_sr)
        sr = target_sr

    wav = wav.squeeze(0)

    # normalize
    peak = wav.abs().max().clamp(min=1e-8)
    wav = wav / peak

    return wav.contiguous(), sr


def chunk_waveform(
    wav: torch.Tensor,
    sr: int,
    chunk_seconds: float = 0.5,
    hop_seconds: float | None = None,
    drop_last: bool = True,
) -> torch.Tensor:
    """
    Splits wav into chunks.

    Returns:
      chunks: Tensor shape (N, chunk_samples)
    """
    chunk_samples = int(round(chunk_seconds * sr))
    if hop_seconds is None:
        hop_samples = chunk_samples  # non-overlapping default
    else:
        hop_samples = int(round(hop_seconds * sr))

    if wav.numel() < chunk_samples:
        # pad short clips
        pad = chunk_samples - wav.numel()
        wav = torch.nn.functional.pad(wav, (0, pad))

    chunks = wav.unfold(dimension=0, size=chunk_samples, step=hop_samples)  # (N, chunk_samples)

    if not drop_last:
        #TODO implement padding for last chunk if not dropped
        pass

    return chunks.contiguous()


def chunks_to_stft_features(
    chunks: torch.Tensor,
    n_fft: int = 512,
    hop_length: int = 128,
    win_length: int | None = None,
    use_log: bool = True,
) -> torch.Tensor:
    """
    Input:
      chunks: (N, L)

    Output:
      feats: (N, F, T) where F = n_fft//2 + 1
    """
    if win_length is None:
        win_length = n_fft

    window = torch.hann_window(win_length, device=chunks.device)

    # stft returns complex tensor: (N, F, T)
    X = torch.stft(
        chunks,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True,
    )

    mag = X.abs()  # (N, F, T)

    if use_log:
        # log compress makes training easier
        mag = torch.log1p(mag)

    return mag.contiguous()

def preprocess_wav_for_training(
    wav_path: str | Path,
    target_sr: int = 16000,
    chunk_seconds: float = 0.5,
    hop_seconds: float | None = None,  # e.g. 0.25 for 50% overlap
    n_fft: int = 512,
    hop_length: int = 128,
) -> torch.Tensor:
    """
    Returns:
      features: (N, F, T) float32
    """
    wav, sr = load_wav_mono_16k(wav_path, target_sr=target_sr)
    chunks = chunk_waveform(wav, sr, chunk_seconds=chunk_seconds, hop_seconds=hop_seconds)
    feats = chunks_to_stft_features(chunks, n_fft=n_fft, hop_length=hop_length)
    return feats


if __name__ == "__main__":
    feats = preprocess_wav_for_training(
        "noisy.wav",
        target_sr=16000,
        chunk_seconds=0.5,
        hop_seconds=0.25,  # helps reduce boundary artifacts; None for no overlap
        n_fft=512,
        hop_length=128,
    )
    print("Features shape:", feats.shape)  # (N, 257, ~63)

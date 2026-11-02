from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
import random
import re
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.linalg import solve
from scipy.signal import istft, resample, resample_poly, stft


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "lab4_data" / "inventory"
UNITS_DIR = DATA_DIR / "units"
MODEL_DIR = ROOT / "models" / "mms-tts-vie"
AUDIO_DIR = ROOT / "audio_output"
FIG_DIR = ROOT / "figures"
RESULT_DIR = ROOT / "results"
SR = 16000
SEED = 457

SENTENCES = {
    "S01": "Mời khách hàng số 23 đến quầy 5.",
    "S02": "Mời khách hàng số 406 đến quầy 7.",
    "S03": "Mời khách hàng số 189 đến quầy 0.",
}

DIGIT = {
    "0": "không", "1": "một", "2": "hai", "3": "ba", "4": "bốn",
    "5": "năm", "6": "sáu", "7": "bảy", "8": "tám", "9": "chín",
}

EXPECTED_UNITS = {
    "mời", "khách", "hàng", "số", "không", "một", "hai", "ba", "bốn",
    "năm", "sáu", "bảy", "tám", "chín", "đến", "quầy",
}


def ensure_dirs() -> None:
    for path in [AUDIO_DIR, FIG_DIR, RESULT_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    for sid in SENTENCES:
        (AUDIO_DIR / sid).mkdir(parents=True, exist_ok=True)
    (AUDIO_DIR / "experiments" / "crossfade").mkdir(parents=True, exist_ok=True)
    (AUDIO_DIR / "experiments" / "prosody").mkdir(parents=True, exist_ok=True)
    (AUDIO_DIR / "experiments" / "hmm").mkdir(parents=True, exist_ok=True)
    (AUDIO_DIR / "blind_listening").mkdir(parents=True, exist_ok=True)


def normalize_text(text: str) -> list[str]:
    """Chuẩn hóa miền hẹp: lowercase, bỏ dấu câu, đọc từng chữ số."""
    text = text.lower().strip()
    text = re.sub(r"[^0-9a-zà-ỹđ\s]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    output: list[str] = []
    for token in text.split():
        if token.isdigit():
            output.extend(DIGIT[digit] for digit in token)
        else:
            output.append(token)
    unknown = sorted(set(output) - EXPECTED_UNITS)
    if unknown:
        raise ValueError(f"Token chưa có trong inventory: {unknown}")
    return output


def spoken_text(text: str) -> str:
    return " ".join(normalize_text(text))


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-12))


def load_wav(path: Path, target_sr: int = SR) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != target_sr:
        x = resample_poly(x, target_sr, sr).astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def save_wav(path: Path, x: np.ndarray, sr: int = SR) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.max(np.abs(x)) + 1e-12)
    if peak > 0.98:
        x = x * (0.98 / peak)
    sf.write(path, np.asarray(x, dtype=np.float32), sr, subtype="PCM_16")


def stft_matrix(x: np.ndarray, n_fft: int = 512, hop_length: int = 128) -> np.ndarray:
    _, _, matrix = stft(
        x, fs=SR, window="hann", nperseg=n_fft,
        noverlap=n_fft - hop_length, nfft=n_fft,
        boundary="zeros", padded=True,
    )
    return matrix


def db_spectrogram(x: np.ndarray, n_fft: int = 512, hop_length: int = 128) -> np.ndarray:
    magnitude = np.abs(stft_matrix(x, n_fft=n_fft, hop_length=hop_length))
    reference = max(float(np.max(magnitude)), 1e-12)
    return 20.0 * np.log10(np.maximum(magnitude, 1e-8) / reference)


def phase_vocoder_time_stretch(x: np.ndarray, duration_scale: float,
                               n_fft: int = 512, hop_length: int = 128) -> np.ndarray:
    """Phase vocoder: duration_scale > 1 tạo audio dài hơn mà gần giữ pitch."""
    if abs(duration_scale - 1.0) < 1e-9:
        return x.copy()
    spectrum = stft_matrix(x, n_fft=n_fft, hop_length=hop_length)
    rate = 1.0 / duration_scale
    time_steps = np.arange(0, spectrum.shape[1] - 1, rate, dtype=float)
    output = np.zeros((spectrum.shape[0], len(time_steps)), dtype=np.complex128)
    phase_advance = 2.0 * np.pi * hop_length * np.arange(spectrum.shape[0]) / n_fft
    phase_acc = np.angle(spectrum[:, 0])
    padded = np.pad(spectrum, ((0, 0), (0, 2)))
    for column, step in enumerate(time_steps):
        left = int(np.floor(step))
        fraction = step - left
        magnitude = (1.0 - fraction) * np.abs(padded[:, left]) + fraction * np.abs(padded[:, left + 1])
        output[:, column] = magnitude * np.exp(1j * phase_acc)
        phase_delta = np.angle(padded[:, left + 1]) - np.angle(padded[:, left]) - phase_advance
        phase_delta -= 2.0 * np.pi * np.round(phase_delta / (2.0 * np.pi))
        phase_acc += phase_advance + phase_delta
    _, y = istft(
        output, fs=SR, window="hann", nperseg=n_fft,
        noverlap=n_fft - hop_length, nfft=n_fft,
        input_onesided=True, boundary=True,
    )
    target_len = max(1, int(round(len(x) * duration_scale)))
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    return np.asarray(y[:target_len], dtype=np.float32)


def pitch_shift_preserve_duration(x: np.ndarray, semitones: float) -> np.ndarray:
    if abs(semitones) < 1e-9:
        return x.copy()
    ratio = 2 ** (semitones / 12.0)
    shifted = resample(x, max(1, int(round(len(x) / ratio))))
    restored = phase_vocoder_time_stretch(np.asarray(shifted, dtype=np.float32), ratio)
    if len(restored) < len(x):
        restored = np.pad(restored, (0, len(x) - len(restored)))
    return np.asarray(restored[:len(x)], dtype=np.float32)


def estimate_f0_track(x: np.ndarray, frame_length: int = 640, hop_length: int = 128,
                      fmin: float = 70.0, fmax: float = 350.0) -> np.ndarray:
    values = []
    min_lag = int(SR / fmax)
    max_lag = int(SR / fmin)
    for start in range(0, max(1, len(x) - frame_length + 1), hop_length):
        frame = x[start:start + frame_length].astype(float)
        if len(frame) < frame_length:
            frame = np.pad(frame, (0, frame_length - len(frame)))
        frame -= np.mean(frame)
        if rms(frame) < 0.008:
            values.append(np.nan)
            continue
        acf = np.correlate(frame, frame, mode="full")[frame_length - 1:]
        if acf[0] <= 1e-12:
            values.append(np.nan)
            continue
        segment = acf[min_lag:max_lag + 1]
        lag = min_lag + int(np.argmax(segment))
        confidence = acf[lag] / acf[0]
        values.append(SR / lag if confidence >= 0.25 else np.nan)
    return np.asarray(values, dtype=float)


@dataclass(frozen=True)
class Unit:
    label: str
    candidate_id: int
    path: Path
    duration: float
    rms_value: float
    samples: np.ndarray


def validate_and_load_inventory() -> tuple[pd.DataFrame, dict[str, list[Unit]]]:
    metadata_path = DATA_DIR / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    metadata = pd.read_csv(metadata_path, encoding="utf-8-sig")
    required = {"label", "candidate_id", "duration_s", "rms", "file_path"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Metadata thiếu cột: {sorted(missing)}")

    inventory: dict[str, list[Unit]] = defaultdict(list)
    validation_rows = []
    for row in metadata.to_dict("records"):
        path = DATA_DIR / str(row["file_path"])
        info = sf.info(path)
        x = load_wav(path)
        issues = []
        if info.samplerate != SR:
            issues.append(f"sample_rate={info.samplerate}")
        if info.channels != 1:
            issues.append(f"channels={info.channels}")
        if info.subtype != "PCM_16":
            issues.append(f"subtype={info.subtype}")
        if len(x) == 0:
            issues.append("empty")
        peak = float(np.max(np.abs(x))) if len(x) else 0.0
        dc = float(np.mean(x)) if len(x) else 0.0
        if peak >= 0.999:
            issues.append("clipping")
        inventory[str(row["label"])].append(Unit(
            label=str(row["label"]),
            candidate_id=int(row["candidate_id"]),
            path=path,
            duration=len(x) / SR,
            rms_value=rms(x),
            samples=x,
        ))
        validation_rows.append({
            "label": row["label"], "candidate_id": int(row["candidate_id"]),
            "file": path.name, "sample_rate": info.samplerate,
            "channels": info.channels, "subtype": info.subtype,
            "duration_s": len(x) / SR, "rms": rms(x), "peak": peak,
            "dc_offset": dc, "status": "PASS" if not issues else "FAIL",
            "issues": ";".join(issues),
        })

    if set(inventory) != EXPECTED_UNITS:
        raise ValueError(f"Inventory labels không đúng: {sorted(inventory)}")
    if any(len(candidates) < 2 for candidates in inventory.values()):
        raise ValueError("Mỗi unit cần ít nhất 2 candidate")
    if any(row["status"] != "PASS" for row in validation_rows):
        raise ValueError("Có file inventory không đạt; xem inventory_validation.csv")
    for candidates in inventory.values():
        candidates.sort(key=lambda item: item.candidate_id)

    validation = pd.DataFrame(validation_rows)
    validation.to_csv(RESULT_DIR / "inventory_validation.csv", index=False, encoding="utf-8-sig")
    plot_inventory_stats(validation)
    return metadata, dict(inventory)


def plot_inventory_stats(validation: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].hist(validation["duration_s"], bins=10, color="#2563eb", edgecolor="white")
    axes[0].set(title="Phân bố thời lượng inventory", xlabel="Giây", ylabel="Số file")
    axes[1].hist(validation["rms"], bins=10, color="#16a34a", edgecolor="white")
    axes[1].set(title="Phân bố RMS inventory", xlabel="RMS", ylabel="Số file")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_inventory_stats.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def fixed_candidate_sequence(tokens: list[str], inventory: dict[str, list[Unit]]) -> list[Unit]:
    return [inventory[token][0] for token in tokens]


def raw_concat(sequence: list[Unit]) -> tuple[np.ndarray, list[int]]:
    boundaries: list[int] = []
    pieces = []
    cursor = 0
    for index, unit in enumerate(sequence):
        if index:
            boundaries.append(cursor)
        pieces.append(unit.samples)
        cursor += len(unit.samples)
    return np.concatenate(pieces), boundaries


def crossfade_concat(sequence: list[Unit], xfade_ms: float) -> tuple[np.ndarray, list[int]]:
    y = sequence[0].samples.copy()
    boundaries: list[int] = []
    requested = int(round(SR * xfade_ms / 1000.0))
    for unit in sequence[1:]:
        b = unit.samples
        n = min(requested, len(y) // 3, len(b) // 3)
        if n <= 0:
            boundaries.append(len(y))
            y = np.concatenate([y, b])
            continue
        alpha = np.linspace(0.0, 1.0, n, dtype=np.float32)
        overlap = (1.0 - alpha) * y[-n:] + alpha * b[:n]
        boundary = len(y) - n // 2
        boundaries.append(boundary)
        y = np.concatenate([y[:-n], overlap, b[n:]])
    return y.astype(np.float32), boundaries


def edge_log_spectrum(x: np.ndarray, side: str, ms: float = 25.0, n_fft: int = 512) -> np.ndarray:
    n = max(32, int(SR * ms / 1000.0))
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    frame = x[-n:] if side == "tail" else x[:n]
    frame = frame * np.hanning(n)
    magnitude = np.abs(np.fft.rfft(frame, n=n_fft))
    return 20.0 * np.log10(np.maximum(magnitude, 1e-7))


def raw_join_features(left: Unit, right: Unit) -> tuple[float, float]:
    spec = float(np.mean((edge_log_spectrum(left.samples, "tail") -
                          edge_log_spectrum(right.samples, "head")) ** 2))
    n = int(SR * 0.025)
    rms_gap = abs(rms(left.samples[-n:]) - rms(right.samples[:n]))
    return spec, rms_gap


def target_cost(unit: Unit, target_duration: float, target_rms: float = 0.08) -> float:
    duration_scale = max(0.02, 0.15 * target_duration)
    rms_scale = 0.01
    return ((unit.duration - target_duration) / duration_scale) ** 2 + \
        ((unit.rms_value - target_rms) / rms_scale) ** 2


def select_units_dp(tokens: list[str], inventory: dict[str, list[Unit]]) -> tuple[list[Unit], pd.DataFrame, float]:
    candidates = [inventory[token] for token in tokens]
    target_durations = {label: float(np.median([u.duration for u in values]))
                        for label, values in inventory.items()}

    all_edges = []
    for position in range(len(tokens) - 1):
        for left in candidates[position]:
            for right in candidates[position + 1]:
                all_edges.append(raw_join_features(left, right))
    spec_scale = max(1e-6, float(np.median([value[0] for value in all_edges])))
    rms_scale = max(1e-6, float(np.median([value[1] for value in all_edges])))

    def join(left: Unit, right: Unit) -> float:
        spec, gap = raw_join_features(left, right)
        return spec / spec_scale + 0.35 * gap / rms_scale

    costs: list[np.ndarray] = []
    backpointers: list[np.ndarray] = []
    first_cost = np.array([
        target_cost(unit, target_durations[tokens[0]]) for unit in candidates[0]
    ], dtype=float)
    costs.append(first_cost)
    backpointers.append(np.full(len(candidates[0]), -1, dtype=int))

    for position in range(1, len(tokens)):
        current = np.full(len(candidates[position]), np.inf)
        back = np.full(len(candidates[position]), -1, dtype=int)
        for j, unit in enumerate(candidates[position]):
            t_cost = target_cost(unit, target_durations[tokens[position]])
            transition = np.array([
                costs[-1][i] + join(prev, unit)
                for i, prev in enumerate(candidates[position - 1])
            ])
            best_prev = int(np.argmin(transition))
            current[j] = t_cost + transition[best_prev]
            back[j] = best_prev
        costs.append(current)
        backpointers.append(back)

    index = int(np.argmin(costs[-1]))
    chosen_indices = [index]
    for position in range(len(tokens) - 1, 0, -1):
        index = int(backpointers[position][index])
        chosen_indices.append(index)
    chosen_indices.reverse()
    sequence = [candidates[p][idx] for p, idx in enumerate(chosen_indices)]

    rows = []
    cumulative = 0.0
    for p, unit in enumerate(sequence):
        t_cost = target_cost(unit, target_durations[tokens[p]])
        j_cost = 0.0 if p == 0 else join(sequence[p - 1], unit)
        cumulative += t_cost + j_cost
        rows.append({
            "position": p, "token": tokens[p], "candidate_id": unit.candidate_id,
            "duration_s": unit.duration, "rms": unit.rms_value,
            "target_cost": t_cost, "join_cost": j_cost,
            "cumulative_cost": cumulative, "file": unit.path.name,
        })
    return sequence, pd.DataFrame(rows), float(np.min(costs[-1]))


def sequence_total_cost(sequence: list[Unit], tokens: list[str], inventory: dict[str, list[Unit]]) -> float:
    # Dùng cùng thang đo theo từng câu để so sánh candidate baseline và DP.
    target_durations = {label: float(np.median([u.duration for u in values]))
                        for label, values in inventory.items()}
    edge_values = []
    for position in range(len(tokens) - 1):
        for left in inventory[tokens[position]]:
            for right in inventory[tokens[position + 1]]:
                edge_values.append(raw_join_features(left, right))
    spec_scale = max(1e-6, float(np.median([v[0] for v in edge_values])))
    rms_scale = max(1e-6, float(np.median([v[1] for v in edge_values])))
    total = 0.0
    for p, unit in enumerate(sequence):
        total += target_cost(unit, target_durations[tokens[p]])
        if p:
            spec, gap = raw_join_features(sequence[p - 1], unit)
            total += spec / spec_scale + 0.35 * gap / rms_scale
    return float(total)


def verify_dp_exhaustive(tokens: list[str], inventory: dict[str, list[Unit]], dp_cost: float) -> dict:
    # Chỉ dùng 4 vị trí đầu để phép vét cạn nhỏ, rõ và tái lập nhanh.
    small_tokens = tokens[:4]
    best = math.inf
    best_ids = None
    for sequence in itertools.product(*(inventory[token] for token in small_tokens)):
        value = sequence_total_cost(list(sequence), small_tokens, inventory)
        if value < best:
            best = value
            best_ids = [unit.candidate_id for unit in sequence]
    dp_sequence, _, small_dp = select_units_dp(small_tokens, inventory)
    return {
        "tokens": small_tokens,
        "exhaustive_cost": best,
        "dp_cost": small_dp,
        "absolute_difference": abs(best - small_dp),
        "exhaustive_candidate_ids": best_ids,
        "dp_candidate_ids": [unit.candidate_id for unit in dp_sequence],
        "pass": bool(abs(best - small_dp) < 1e-9),
        "full_sentence_dp_cost": dp_cost,
    }


def boundary_metrics(y: np.ndarray, boundaries: list[int]) -> tuple[float, float]:
    n = int(SR * 0.025)
    offset = int(SR * 0.0025)
    spectral, jumps = [], []
    for boundary in boundaries:
        left_center = boundary - offset
        right_center = boundary + offset
        if left_center - n // 2 < 0 or right_center + n // 2 >= len(y):
            continue
        left = y[left_center - n // 2:left_center + n // 2]
        right = y[right_center - n // 2:right_center + n // 2]
        l_spec = edge_log_spectrum(left, "head")
        r_spec = edge_log_spectrum(right, "head")
        spectral.append(float(np.mean((l_spec - r_spec) ** 2)))
        radius = int(SR * 0.005)
        local_diff = np.abs(np.diff(y[boundary - radius:boundary + radius]))
        jumps.append(float(np.percentile(local_diff, 95)))
    return float(np.mean(spectral)), float(np.mean(jumps))


def plot_waveform_spectrogram(y: np.ndarray, boundaries: list[int], title: str, path: Path) -> None:
    times = np.arange(len(y)) / SR
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(times, y, linewidth=0.6, color="#1d4ed8")
    axes[0].set_ylabel("Biên độ")
    axes[0].set_title(title)
    spec = db_spectrogram(y, n_fft=512, hop_length=128)
    extent = [0, len(y) / SR, 0, SR / 2]
    axes[1].imshow(spec, origin="lower", aspect="auto", extent=extent, cmap="magma", vmin=-80, vmax=0)
    axes[1].set_ylim(0, 5000)
    for boundary in boundaries:
        t = boundary / SR
        axes[0].axvline(t, color="red", alpha=0.55, linewidth=0.8)
        axes[1].axvline(t, color="cyan", alpha=0.65, linewidth=0.8)
    axes[1].set_ylabel("Tần số (Hz)")
    axes[1].set_xlabel("Thời gian (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_boundary_zoom(raw_y: np.ndarray, raw_boundaries: list[int], best_y: np.ndarray,
                       best_boundaries: list[int], path: Path) -> None:
    selected = [0, min(3, len(raw_boundaries) - 1)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    for column, idx in enumerate(selected):
        for row, (signal, boundaries, name) in enumerate([
            (raw_y, raw_boundaries, "Candidate cố định"),
            (best_y, best_boundaries, "Sau unit selection/cross-fade đã chọn")
        ]):
            boundary = boundaries[idx]
            radius = int(0.04 * SR)
            start, end = max(0, boundary - radius), min(len(signal), boundary + radius)
            t = (np.arange(start, end) - boundary) / SR * 1000
            axes[row, column].plot(t, signal[start:end], linewidth=0.8)
            axes[row, column].axvline(0, color="red", linestyle="--")
            axes[row, column].set_title(f"{name} - biên {idx + 1}")
            axes[row, column].set_xlabel("Thời gian quanh biên (ms)")
            axes[row, column].set_ylabel("Biên độ")
            axes[row, column].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_a_to_c(inventory: dict[str, list[Unit]]) -> dict[str, dict]:
    token_rows, cost_rows, comparison_rows, latency_rows = [], [], [], []
    outputs: dict[str, dict] = {}
    for sid, text in SENTENCES.items():
        tokens = normalize_text(text)
        token_rows.append({"sentence_id": sid, "input_text": text,
                           "normalized_tokens": " | ".join(tokens),
                           "spoken_text": " ".join(tokens)})
        baseline_sequence = fixed_candidate_sequence(tokens, inventory)
        raw_y, raw_boundaries = raw_concat(baseline_sequence)
        raw_path = AUDIO_DIR / sid / "raw_concat.wav"
        save_wav(raw_path, raw_y)

        best_sequence, table, dp_cost = select_units_dp(tokens, inventory)
        table.insert(0, "sentence_id", sid)
        cost_rows.extend(table.to_dict("records"))
        baseline_cost = sequence_total_cost(baseline_sequence, tokens, inventory)
        comparison_rows.append({
            "sentence_id": sid, "baseline_total_cost": baseline_cost,
            "dp_total_cost": dp_cost, "cost_reduction_percent":
            100.0 * (baseline_cost - dp_cost) / max(baseline_cost, 1e-12),
        })

        crossfade_rows = []
        variants = {}
        for xfade_ms in [0, 5, 15, 30]:
            y, boundaries = crossfade_concat(best_sequence, xfade_ms)
            spec_disc, jump = boundary_metrics(y, boundaries)
            variants[xfade_ms] = (y, boundaries)
            crossfade_rows.append({
                "sentence_id": sid, "crossfade_ms": xfade_ms,
                "spectral_discontinuity": spec_disc,
                "mean_amplitude_jump": jump, "duration_s": len(y) / SR,
            })
            if sid == "S01":
                save_wav(AUDIO_DIR / "experiments" / "crossfade" / f"S01_xfade_{xfade_ms:02d}ms.wav", y)

        crossfade_df = pd.DataFrame(crossfade_rows)
        # Chọn theo discontinuity + jump; thêm phạt nhỏ cho overlap dài để tránh mờ phụ âm.
        spec_norm = crossfade_df["spectral_discontinuity"] / max(crossfade_df["spectral_discontinuity"].median(), 1e-9)
        jump_norm = crossfade_df["mean_amplitude_jump"] / max(crossfade_df["mean_amplitude_jump"].median(), 1e-9)
        blur_penalty = crossfade_df["crossfade_ms"] / 30.0 * 0.12
        crossfade_df["selection_score"] = spec_norm + 0.5 * jump_norm + blur_penalty
        chosen_ms = int(crossfade_df.loc[crossfade_df["selection_score"].idxmin(), "crossfade_ms"])
        best_y, best_boundaries = variants[chosen_ms]
        save_wav(AUDIO_DIR / sid / "best_concat.wav", best_y)

        if sid == "S01":
            shutil.copy2(raw_path, AUDIO_DIR / "raw_concat.wav")
            shutil.copy2(AUDIO_DIR / sid / "best_concat.wav", AUDIO_DIR / "best_concat.wav")
            crossfade_df.to_csv(RESULT_DIR / "crossfade_results.csv", index=False, encoding="utf-8-sig")
            plot_waveform_spectrogram(raw_y, raw_boundaries,
                                      "Baseline concatenation - S01",
                                      FIG_DIR / "02_baseline_waveform_spectrogram.png")
            plot_waveform_spectrogram(best_y, best_boundaries,
                                      f"Unit selection + cross-fade {chosen_ms} ms - S01",
                                      FIG_DIR / "03_best_concat_waveform_spectrogram.png")
            plot_boundary_zoom(raw_y, raw_boundaries, best_y, best_boundaries,
                               FIG_DIR / "04_boundary_before_after.png")
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(crossfade_df["crossfade_ms"], crossfade_df["spectral_discontinuity"],
                    marker="o", label="Spectral discontinuity")
            ax.set_xlabel("Cross-fade (ms)")
            ax.set_ylabel("Sai khác phổ trung bình")
            ax.grid(alpha=0.25)
            ax2 = ax.twinx()
            ax2.plot(crossfade_df["crossfade_ms"], crossfade_df["mean_amplitude_jump"],
                     marker="s", color="#dc2626", label="Amplitude jump")
            ax2.set_ylabel("Biên độ nhảy trung bình")
            ax.set_title("Thí nghiệm cross-fade 0/5/15/30 ms")
            fig.tight_layout()
            fig.savefig(FIG_DIR / "05_crossfade_experiment.png", dpi=180, bbox_inches="tight")
            plt.close(fig)

        outputs[sid] = {
            "tokens": tokens, "raw": raw_path, "best": AUDIO_DIR / sid / "best_concat.wav",
            "best_sequence": best_sequence, "chosen_crossfade_ms": chosen_ms,
            "dp_cost": dp_cost,
        }

        # Latency text -> chọn unit -> waveform; bỏ phần vẽ/lưu file.
        for run in range(1, 6):
            start = time.perf_counter()
            measured_tokens = normalize_text(text)
            measured_sequence, _, _ = select_units_dp(measured_tokens, inventory)
            _ = crossfade_concat(measured_sequence, chosen_ms)
            latency_rows.append({
                "sentence_id": sid, "run": run,
                "latency_s": time.perf_counter() - start,
                "crossfade_ms": chosen_ms,
            })

    pd.DataFrame(token_rows).to_csv(RESULT_DIR / "normalized_sentences.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cost_rows).to_csv(RESULT_DIR / "candidate_costs.csv", index=False, encoding="utf-8-sig")
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(RESULT_DIR / "unit_selection_comparison.csv", index=False, encoding="utf-8-sig")
    concat_latency = pd.DataFrame(latency_rows)
    concat_latency.to_csv(RESULT_DIR / "concat_latency_runs.csv", index=False, encoding="utf-8-sig")
    concat_latency.groupby("sentence_id", as_index=False).agg(
        latency_mean_s=("latency_s", "mean"), latency_std_s=("latency_s", "std"),
        latency_min_s=("latency_s", "min"), latency_max_s=("latency_s", "max"),
        crossfade_ms=("crossfade_ms", "first"),
    ).to_csv(RESULT_DIR / "concat_latency_summary.csv", index=False, encoding="utf-8-sig")
    verification = verify_dp_exhaustive(outputs["S01"]["tokens"], inventory, outputs["S01"]["dp_cost"])
    (RESULT_DIR / "dp_verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(comparison_df))
    width = 0.36
    ax.bar(x - width / 2, comparison_df["baseline_total_cost"], width, label="Candidate cố định")
    ax.bar(x + width / 2, comparison_df["dp_total_cost"], width, label="Dynamic programming")
    ax.set_xticks(x, comparison_df["sentence_id"])
    ax.set_ylabel("Total cost")
    ax.set_title("So sánh chi phí chọn unit")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_unit_selection_cost.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return outputs


def estimate_period_f0(x: np.ndarray, sr: int = SR) -> tuple[float, int, np.ndarray]:
    frame_length = min(len(x), int(0.16 * sr))
    center = len(x) // 2
    start = max(0, center - frame_length // 2)
    frame = x[start:start + frame_length].astype(float)
    frame -= np.mean(frame)
    acf = np.correlate(frame, frame, mode="full")[len(frame) - 1:]
    min_lag, max_lag = int(sr / 350), int(sr / 70)
    lag = min_lag + int(np.argmax(acf[min_lag:max_lag + 1]))
    return sr / lag, lag, frame


def run_prosody(inventory: dict[str, list[Unit]]) -> None:
    unit = inventory["ba"][1]
    x = unit.samples
    period_f0, period_samples, frame = estimate_period_f0(x)
    f0_track = estimate_f0_track(x, frame_length=640, hop_length=128)
    track_f0 = float(np.nanmedian(f0_track))

    rows = [{
        "experiment": "f0_check", "setting": "original", "duration_s": len(x) / SR,
        "measured_f0_hz": track_f0, "period_samples": period_samples,
        "fs_over_np_hz": period_f0, "semitone_ratio_expected": 1.0,
    }]
    prosody_dir = AUDIO_DIR / "experiments" / "prosody"
    duration_audio = {}
    for scale in [0.8, 1.0, 1.2]:
        y = phase_vocoder_time_stretch(x, duration_scale=scale)
        duration_audio[scale] = y
        save_wav(prosody_dir / f"ba_duration_{scale:.1f}x.wav", y)
        rows.append({
            "experiment": "duration", "setting": scale, "duration_s": len(y) / SR,
            "measured_f0_hz": float(np.nanmedian(estimate_f0_track(y))),
            "period_samples": "", "fs_over_np_hz": "", "semitone_ratio_expected": "",
        })

    pitch_audio = {}
    for semitone in [-2, 0, 2]:
        y = pitch_shift_preserve_duration(x, semitone)
        pitch_audio[semitone] = y
        save_wav(prosody_dir / f"ba_pitch_{semitone:+d}st.wav", y)
        measured = float(np.nanmedian(estimate_f0_track(y, fmin=60, fmax=420)))
        rows.append({
            "experiment": "pitch", "setting": semitone, "duration_s": len(y) / SR,
            "measured_f0_hz": measured, "period_samples": "", "fs_over_np_hz": "",
            "semitone_ratio_expected": 2 ** (semitone / 12),
        })
    pd.DataFrame(rows).to_csv(RESULT_DIR / "prosody_results.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(2, 1, figsize=(11, 6))
    show_n = min(len(frame), max(6 * period_samples, 500))
    axes[0].plot(np.arange(show_n), frame[:show_n], linewidth=0.8)
    for k in range(0, show_n, period_samples):
        axes[0].axvline(k, color="red", alpha=0.4)
    axes[0].set(title=f"Kiểm chứng chu kỳ: Np={period_samples} mẫu, Fs/Np={period_f0:.1f} Hz",
                xlabel="Mẫu", ylabel="Biên độ")
    times = np.arange(len(f0_track)) * 128 / SR
    axes[1].plot(times, f0_track, marker=".", linewidth=1)
    axes[1].axhline(period_f0, color="red", linestyle="--", label="Fs/Np")
    axes[1].set(title=f"F0 bằng autocorrelation theo frame, median={track_f0:.1f} Hz",
                xlabel="Thời gian (s)", ylabel="F0 (Hz)")
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_f0_verification.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    plot_spectrogram_grid(duration_audio, "Duration scale", FIG_DIR / "08_duration_experiment.png")
    plot_spectrogram_grid(pitch_audio, "Pitch shift (semitone)", FIG_DIR / "09_pitch_experiment.png")


def plot_spectrogram_grid(audio_map: dict, prefix: str, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, (setting, y) in zip(axes, audio_map.items()):
        spec = db_spectrogram(y, n_fft=512, hop_length=64)
        ax.imshow(spec, origin="lower", aspect="auto", extent=[0, len(y) / SR, 0, SR / 2],
                  cmap="magma", vmin=-80, vmax=0)
        ax.set_ylim(0, 5000)
        ax.set_title(f"{prefix}: {setting}")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def smooth_trajectory(values: np.ndarray, strength: float = 12.0) -> np.ndarray:
    n = len(values)
    if n < 3:
        return values.copy()
    difference = np.zeros((n - 1, n))
    for i in range(n - 1):
        difference[i, i] = -1.0
        difference[i, i + 1] = 1.0
    system = np.eye(n) + strength * difference.T @ difference
    return solve(system, values, assume_a="pos")


def token_acoustic_means(token: str) -> tuple[float, float, float]:
    tone_f0 = {
        "mời": 145, "khách": 185, "hàng": 145, "số": 180, "không": 155,
        "một": 130, "hai": 155, "ba": 160, "bốn": 180, "năm": 150,
        "sáu": 182, "bảy": 142, "tám": 180, "chín": 178, "đến": 180, "quầy": 142,
    }
    vowel_formants = {
        "mời": (600, 1250), "khách": (760, 1650), "hàng": (720, 1250), "số": (500, 900),
        "không": (520, 950), "một": (550, 1050), "hai": (750, 2100), "ba": (780, 1250),
        "bốn": (500, 950), "năm": (700, 1450), "sáu": (650, 1200), "bảy": (720, 1950),
        "tám": (700, 1500), "chín": (350, 2200), "đến": (520, 1800), "quầy": (650, 1500),
    }
    f1, f2 = vowel_formants[token]
    return tone_f0[token], f1, f2


def generate_hmm_tracks(tokens: list[str], inventory: dict[str, list[Unit]],
                        duration_scale: float = 1.0, f0_scale: float = 1.0,
                        seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    hop_s = 0.010
    track_rows, state_rows = [], []
    frame_index = 0
    state_id = 0
    for token_index, token in enumerate(tokens):
        token_duration = float(np.median([u.duration for u in inventory[token]])) * duration_scale
        total_frames = max(6, int(round(token_duration / hop_s)))
        proportions = np.array([0.25, 0.50, 0.25])
        state_frames = np.maximum(2, np.rint(total_frames * proportions).astype(int))
        state_frames[1] += total_frames - int(np.sum(state_frames))
        base_f0, base_f1, base_f2 = token_acoustic_means(token)
        for local_state, frames in enumerate(state_frames):
            state_id += 1
            state_name = f"{token}_s{local_state + 1}"
            f0_mean = base_f0 * f0_scale * [0.96, 1.00, 1.04][local_state]
            f1_mean = base_f1 * [0.94, 1.00, 1.03][local_state]
            f2_mean = base_f2 * [0.97, 1.00, 1.02][local_state]
            energy_mean = [0.045, 0.075, 0.050][local_state]
            state_rows.append({
                "state_id": state_id, "token_index": token_index, "token": token,
                "state": state_name, "duration_frames": int(frames),
                "duration_ms": int(frames) * 10, "f0_mean": f0_mean, "f0_std": 5.0,
                "f1_mean": f1_mean, "f1_std": 35.0, "f2_mean": f2_mean, "f2_std": 55.0,
                "energy_mean": energy_mean, "energy_std": 0.006,
            })
            for local_frame in range(int(frames)):
                edge_state = local_state != 1
                voiced = not (edge_state and local_frame < max(1, int(frames) // 3))
                track_rows.append({
                    "frame": frame_index, "time_s": frame_index * hop_s,
                    "token_index": token_index, "token": token, "state_id": state_id,
                    "state": state_name, "voiced": int(voiced),
                    "f0_raw": rng.normal(f0_mean, 5.0) if voiced else 0.0,
                    "f1_raw": rng.normal(f1_mean, 35.0),
                    "f2_raw": rng.normal(f2_mean, 55.0),
                    "energy_raw": max(0.01, rng.normal(energy_mean, 0.006)),
                })
                frame_index += 1

    tracks = pd.DataFrame(track_rows)
    voiced = tracks["voiced"].to_numpy(bool)
    f0_seed = tracks["f0_raw"].to_numpy(float)
    if np.any(voiced):
        indices = np.arange(len(f0_seed))
        f0_filled = np.interp(indices, indices[voiced], f0_seed[voiced])
        f0_smooth = smooth_trajectory(f0_filled, 16.0)
        tracks["f0_hz"] = np.where(voiced, f0_smooth, 0.0)
    else:
        tracks["f0_hz"] = 0.0
    tracks["f1_hz"] = smooth_trajectory(tracks["f1_raw"].to_numpy(float), 20.0)
    tracks["f2_hz"] = smooth_trajectory(tracks["f2_raw"].to_numpy(float), 20.0)
    tracks["energy"] = np.maximum(0.005, smooth_trajectory(tracks["energy_raw"].to_numpy(float), 10.0))
    return tracks, pd.DataFrame(state_rows)


def synthesize_hmm_waveform(tracks: pd.DataFrame, sr: int = SR) -> np.ndarray:
    hop = int(0.010 * sr)
    frame_len = int(0.030 * sr)
    window = np.hanning(frame_len).astype(np.float32)
    total = hop * (len(tracks) - 1) + frame_len
    y = np.zeros(total, dtype=np.float64)
    norm = np.zeros(total, dtype=np.float64)
    phase = 0.0
    rng = np.random.default_rng(SEED)
    t_local = np.arange(frame_len) / sr
    for i, row in tracks.iterrows():
        f0 = float(row["f0_hz"])
        energy = float(row["energy"])
        if f0 > 1.0:
            harmonics = np.arange(1, min(35, int((sr / 2) // f0)) + 1)
            frequencies = harmonics * f0
            envelope = (
                0.7 * np.exp(-0.5 * ((frequencies - row["f1_hz"]) / 170.0) ** 2) +
                0.5 * np.exp(-0.5 * ((frequencies - row["f2_hz"]) / 240.0) ** 2) +
                0.08 / np.sqrt(harmonics)
            )
            frame = np.sum((envelope / harmonics)[:, None] *
                           np.sin(2 * np.pi * frequencies[:, None] * t_local + phase), axis=0)
            phase = float((phase + 2 * np.pi * f0 * hop / sr) % (2 * np.pi))
        else:
            frame = rng.normal(0.0, 0.35, frame_len)
        frame = frame / (np.max(np.abs(frame)) + 1e-9) * energy
        start = i * hop
        y[start:start + frame_len] += frame * window
        norm[start:start + frame_len] += window
    y /= np.maximum(norm, 1e-6)
    return y.astype(np.float32)


def run_hmm(inventory: dict[str, list[Unit]]) -> pd.DataFrame:
    # Sinh base waveform cho cả ba câu để đánh giá cùng nội dung.
    latency_rows = []
    for sid, text in SENTENCES.items():
        tokens_sid = normalize_text(text)
        tracks_sid = states_sid = y_sid = None
        for run in range(1, 6):
            start = time.perf_counter()
            tracks_sid, states_sid = generate_hmm_tracks(tokens_sid, inventory)
            y_sid = synthesize_hmm_waveform(tracks_sid)
            latency_rows.append({
                "sentence_id": sid, "run": run,
                "latency_s": time.perf_counter() - start,
            })
        save_wav(AUDIO_DIR / sid / "hmm_tts.wav", y_sid)
        tracks_sid.to_csv(RESULT_DIR / f"hmm_tracks_{sid}.csv", index=False, encoding="utf-8-sig")
        states_sid.to_csv(RESULT_DIR / f"hmm_states_{sid}.csv", index=False, encoding="utf-8-sig")
    shutil.copy2(AUDIO_DIR / "S01" / "hmm_tts.wav", AUDIO_DIR / "hmm_tts.wav")

    tokens = normalize_text(SENTENCES["S01"])
    variants = {
        "base": (1.0, 1.0), "duration_0.8": (0.8, 1.0),
        "duration_1.2": (1.2, 1.0), "f0_0.9": (1.0, 0.9), "f0_1.1": (1.0, 1.1),
    }
    variant_tracks = {}
    hmm_dir = AUDIO_DIR / "experiments" / "hmm"
    for name, (duration_scale, f0_scale) in variants.items():
        tracks, states = generate_hmm_tracks(tokens, inventory, duration_scale, f0_scale)
        y = synthesize_hmm_waveform(tracks)
        save_wav(hmm_dir / f"hmm_{name}.wav", y)
        variant_tracks[name] = tracks
        if name == "base":
            tracks.to_csv(RESULT_DIR / "hmm_tracks.csv", index=False, encoding="utf-8-sig")
            states.to_csv(RESULT_DIR / "hmm_state_sequence.csv", index=False, encoding="utf-8-sig")

    base = variant_tracks["base"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(base["time_s"], base["f0_hz"], color="#2563eb")
    axes[0].set_ylabel("F0 (Hz)")
    axes[1].plot(base["time_s"], base["f1_hz"], label="F1")
    axes[1].plot(base["time_s"], base["f2_hz"], label="F2")
    axes[1].set_ylabel("Tần số (Hz)")
    axes[1].legend()
    axes[2].plot(base["time_s"], base["energy"], color="#16a34a")
    axes[2].set_ylabel("Energy")
    axes[2].set_xlabel("Thời gian (s)")
    boundaries = base.groupby("state_id")["time_s"].min().to_numpy()
    for ax in axes:
        for boundary in boundaries:
            ax.axvline(boundary, color="gray", alpha=0.18, linewidth=0.6)
        ax.grid(alpha=0.15)
    axes[0].set_title("HMM generation: state sequence và trajectory đã làm trơn bằng delta constraint")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "10_hmm_tracks.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for name in ["duration_0.8", "base", "duration_1.2"]:
        tracks = variant_tracks[name]
        axes[0].plot(tracks["time_s"], tracks["f0_hz"], label=name)
    for name in ["f0_0.9", "base", "f0_1.1"]:
        tracks = variant_tracks[name]
        axes[1].plot(tracks["time_s"], tracks["f0_hz"], label=name)
    axes[0].set(title="Thay đổi duration", xlabel="Thời gian (s)", ylabel="F0 (Hz)")
    axes[1].set(title="Thay đổi F0 scale", xlabel="Thời gian (s)", ylabel="F0 (Hz)")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "11_hmm_parameter_scaling.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    latency = pd.DataFrame(latency_rows)
    latency.to_csv(RESULT_DIR / "hmm_latency_runs.csv", index=False, encoding="utf-8-sig")
    summary = latency.groupby("sentence_id", as_index=False).agg(
        latency_mean_s=("latency_s", "mean"), latency_std_s=("latency_s", "std"),
        latency_min_s=("latency_s", "min"), latency_max_s=("latency_s", "max"),
    )
    summary.to_csv(RESULT_DIR / "hmm_latency_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def run_neural_tts() -> pd.DataFrame:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from transformers import VitsModel, VitsTokenizer, set_seed

    if not (MODEL_DIR / "model.safetensors").exists():
        raise FileNotFoundError("Thiếu model offline tại models/mms-tts-vie")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    load_start = time.perf_counter()
    tokenizer = VitsTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = VitsModel.from_pretrained(MODEL_DIR, local_files_only=True).to(device).eval()
    if device == "cuda":
        torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start
    sampling_rate = int(model.config.sampling_rate)

    # Warm-up tách riêng khỏi latency được báo cáo.
    warm = tokenizer(text="xin chào", return_tensors="pt")
    warm = {key: value.to(device) for key, value in warm.items()}
    with torch.no_grad():
        _ = model(**warm)
    if device == "cuda":
        torch.cuda.synchronize()

    latency_rows = []
    for sid, original in SENTENCES.items():
        text = spoken_text(original)
        inputs = tokenizer(text=text, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        last_outputs = None
        for run in range(1, 6):
            set_seed(SEED + run)
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                last_outputs = model(**inputs)
            if device == "cuda":
                torch.cuda.synchronize()
            latency = time.perf_counter() - start
            latency_rows.append({
                "sentence_id": sid, "run": run, "latency_s": latency,
                "device": device, "model_load_s": load_seconds,
                "spoken_text": text,
            })
        waveform = last_outputs.waveform[0].detach().cpu().numpy().astype(np.float32)
        neural_path = AUDIO_DIR / sid / "neural_tts.wav"
        save_wav(neural_path, waveform, sampling_rate)
        if sid == "S01":
            shutil.copy2(neural_path, AUDIO_DIR / "neural_tts.wav")
        mel = last_outputs.spectrogram[0].detach().cpu().numpy()
        fig, ax = plt.subplots(figsize=(10, 4))
        image = ax.imshow(mel, origin="lower", aspect="auto", cmap="magma")
        ax.set(title=f"MMS-VITS acoustic spectrogram - {sid}", xlabel="Frame", ylabel="Mel/channel")
        fig.colorbar(image, ax=ax, label="Log amplitude")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"12_neural_spectrogram_{sid}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    latency_df = pd.DataFrame(latency_rows)
    latency_df.to_csv(RESULT_DIR / "neural_latency_runs.csv", index=False, encoding="utf-8-sig")
    summary = latency_df.groupby(["sentence_id", "device"], as_index=False).agg(
        latency_mean_s=("latency_s", "mean"), latency_std_s=("latency_s", "std"),
        latency_min_s=("latency_s", "min"), latency_max_s=("latency_s", "max"),
        model_load_s=("model_load_s", "first"), spoken_text=("spoken_text", "first"),
    )
    summary.to_csv(RESULT_DIR / "neural_latency_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def build_evaluation_package(latency_summary: pd.DataFrame, hmm_latency_summary: pd.DataFrame) -> None:
    rng = random.Random(SEED)
    systems = ["concat", "hmm", "neural"]
    source_map = {
        "concat": lambda sid: AUDIO_DIR / sid / "best_concat.wav",
        "hmm": lambda sid: AUDIO_DIR / sid / "hmm_tts.wav",
        "neural": lambda sid: AUDIO_DIR / sid / "neural_tts.wav",
    }
    entries = [(sid, system) for sid in SENTENCES for system in systems]
    rng.shuffle(entries)
    manifest_rows, rating_rows = [], []
    for index, (sid, system) in enumerate(entries, start=1):
        clip_id = f"C{index:02d}"
        source = source_map[system](sid)
        available = source.exists()
        if available:
            destination = AUDIO_DIR / "blind_listening" / f"{clip_id}.wav"
            shutil.copy2(source, destination)
        manifest_rows.append({
            "clip_id": clip_id, "sentence_id": sid, "system": system,
            "source_file": str(source.relative_to(ROOT)), "available": available,
        })
        rating_rows.append({
            "listener_id": "", "clip_id": clip_id, "transcription": "",
            "naturalness_1_5": "", "continuity_1_5": "", "notes": "",
        })
    pd.DataFrame(manifest_rows).to_csv(RESULT_DIR / "blind_manifest_private.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rating_rows).to_csv(RESULT_DIR / "ratings_template.csv", index=False, encoding="utf-8-sig")

    latency_map = latency_summary.set_index("sentence_id")["latency_mean_s"].to_dict()
    concat_latency_summary = pd.read_csv(RESULT_DIR / "concat_latency_summary.csv")
    concat_latency_mean = float(concat_latency_summary["latency_mean_s"].mean())
    hmm_latency_mean = float(hmm_latency_summary["latency_mean_s"].mean())
    comparison = pd.DataFrame([
        {
            "method": "Concatenation", "intelligibility_percent": "PENDING_LISTENING_TEST",
            "naturalness_mos": "PENDING_LISTENING_TEST", "continuity_mos": "PENDING_LISTENING_TEST",
            "latency_mean_s": concat_latency_mean,
            "storage_bytes": directory_size(UNITS_DIR),
            "notes": "Best unit selection + cross-fade",
        },
        {
            "method": "HMM/parametric demo", "intelligibility_percent": "PENDING_LISTENING_TEST",
            "naturalness_mos": "PENDING_LISTENING_TEST", "continuity_mos": "PENDING_LISTENING_TEST",
            "latency_mean_s": hmm_latency_mean, "storage_bytes": (RESULT_DIR / "hmm_tracks.csv").stat().st_size,
            "notes": "Guided state/trajectory demo; waveform is illustrative",
        },
        {
            "method": "Neural MMS-VITS", "intelligibility_percent": "PENDING_LISTENING_TEST",
            "naturalness_mos": "PENDING_LISTENING_TEST", "continuity_mos": "PENDING_LISTENING_TEST",
            "latency_mean_s": float(np.mean(list(latency_map.values()))),
            "storage_bytes": directory_size(MODEL_DIR),
            "notes": "facebook/mms-tts-vie, local_files_only=True",
        },
    ])
    comparison.to_csv(RESULT_DIR / "backend_comparison.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = ["Inventory", "HMM tracks", "Neural model"]
    sizes_mb = [directory_size(UNITS_DIR), (RESULT_DIR / "hmm_tracks.csv").stat().st_size,
                directory_size(MODEL_DIR)]
    sizes_mb = [value / (1024 ** 2) for value in sizes_mb]
    axes[0].bar(labels, sizes_mb, color=["#2563eb", "#f59e0b", "#16a34a"])
    axes[0].set(title="Dung lượng dữ liệu/model", ylabel="MB")
    latency_values = [concat_latency_mean, hmm_latency_mean, float(np.mean(list(latency_map.values())))]
    axes[1].bar(["Concatenation", "HMM", "Neural"], latency_values,
                color=["#2563eb", "#f59e0b", "#16a34a"])
    axes[1].set(title="Latency trung bình trên cùng máy", ylabel="Giây")
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "13_backend_objective_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def create_experiment_summary() -> None:
    inventory = pd.read_csv(RESULT_DIR / "inventory_validation.csv")
    selection = pd.read_csv(RESULT_DIR / "unit_selection_comparison.csv")
    crossfade = pd.read_csv(RESULT_DIR / "crossfade_results.csv")
    prosody = pd.read_csv(RESULT_DIR / "prosody_results.csv")
    concat_latency = pd.read_csv(RESULT_DIR / "concat_latency_summary.csv")
    hmm_latency = pd.read_csv(RESULT_DIR / "hmm_latency_summary.csv")
    neural_latency = pd.read_csv(RESULT_DIR / "neural_latency_summary.csv")
    f0_row = prosody[prosody["experiment"] == "f0_check"].iloc[0]
    pitch_rows = prosody[prosody["experiment"] == "pitch"]
    summary = {
        "A_inventory": {
            "files": int(len(inventory)), "labels": int(inventory["label"].nunique()),
            "all_pass": bool((inventory["status"] == "PASS").all()),
            "duration_range_s": [float(inventory["duration_s"].min()), float(inventory["duration_s"].max())],
            "rms_range": [float(inventory["rms"].min()), float(inventory["rms"].max())],
        },
        "C_unit_selection": {
            "mean_cost_reduction_percent": float(selection["cost_reduction_percent"].mean()),
            "dp_exhaustive_match": json.loads((RESULT_DIR / "dp_verification.json").read_text(encoding="utf-8"))["pass"],
            "selected_crossfade_ms": int(crossfade.loc[crossfade["selection_score"].idxmin(), "crossfade_ms"]),
            "crossfade_observation": "Với inventory này, 0 ms có score thấp nhất; overlap dài làm tăng discontinuity do biên đã có silence sạch.",
        },
        "D_prosody": {
            "period_samples": int(f0_row["period_samples"]),
            "fs_over_np_hz": float(f0_row["fs_over_np_hz"]),
            "frame_autocorrelation_f0_hz": float(f0_row["measured_f0_hz"]),
            "pitch_measured_hz": {
                str(row["setting"]): float(row["measured_f0_hz"]) for _, row in pitch_rows.iterrows()
            },
        },
        "E_hmm": {
            "states_S01": int(len(pd.read_csv(RESULT_DIR / "hmm_state_sequence.csv"))),
            "frames_S01": int(len(pd.read_csv(RESULT_DIR / "hmm_tracks.csv"))),
            "latency_mean_s": float(hmm_latency["latency_mean_s"].mean()),
        },
        "F_neural": {
            "model": "facebook/mms-tts-vie", "offline": True,
            "device": str(neural_latency["device"].iloc[0]),
            "latency_mean_s": float(neural_latency["latency_mean_s"].mean()),
            "model_size_mb": directory_size(MODEL_DIR) / (1024 ** 2),
        },
        "G_evaluation": {
            "concat_latency_mean_s": float(concat_latency["latency_mean_s"].mean()),
            "hmm_latency_mean_s": float(hmm_latency["latency_mean_s"].mean()),
            "neural_latency_mean_s": float(neural_latency["latency_mean_s"].mean()),
            "subjective_scores": "PENDING_REAL_LISTENERS",
        },
    }
    proxy_path = RESULT_DIR / "automated_proxy_summary.csv"
    if proxy_path.exists():
        proxy = pd.read_csv(proxy_path)
        summary["G_automated_proxy"] = {
            row["system"]: {
                "intelligibility_asr_percent": float(row["intelligibility_asr_percent"]),
                "naturalness_proxy_1_5": float(row["naturalness_proxy_1_5"]),
                "continuity_proxy_1_5": float(row["continuity_proxy_1_5"]),
            } for _, row in proxy.iterrows()
        }
        summary["G_automated_proxy"]["disclosure"] = "AUTOMATED_PROXY_NOT_HUMAN_MOS"
    (RESULT_DIR / "experiment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def create_run_manifest(start_time: float) -> None:
    files = []
    for base in [AUDIO_DIR, FIG_DIR, RESULT_DIR]:
        for path in base.rglob("*"):
            if path.is_file():
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                files.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": digest})
    manifest = {
        "lab": "CSE457 Lab 4 - Tong hop tieng noi",
        "seed": SEED, "sample_rate": SR,
        "runtime_seconds": time.perf_counter() - start_time,
        "sentences": SENTENCES,
        "outputs": files,
        "limitations": [
            "Inventory is synthetic neural speech and must be disclosed as such.",
            "Subjective MOS/intelligibility require real listeners; template is prepared but scores are not fabricated.",
            "HMM waveform is an educational parametric resynthesis; the required state/trajectory analysis is complete.",
        ],
    }
    (RESULT_DIR / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    np.random.seed(SEED)
    random.seed(SEED)
    ensure_dirs()
    start = time.perf_counter()
    print("[A] Kiểm tra inventory, metadata, histogram")
    _, inventory = validate_and_load_inventory()
    print("[B-C] Baseline, unit selection, cross-fade")
    run_a_to_c(inventory)
    print("[D] F0, duration, pitch, source-filter")
    run_prosody(inventory)
    print("[E] HMM state/trajectory generation")
    hmm_latency_summary = run_hmm(inventory)
    print("[F] Neural TTS pretrained offline")
    latency_summary = run_neural_tts()
    print("[G] Gói đánh giá mù và bảng so sánh")
    build_evaluation_package(latency_summary, hmm_latency_summary)
    create_experiment_summary()
    create_run_manifest(start)
    print(f"Hoàn tất sau {time.perf_counter() - start:.1f} giây")


if __name__ == "__main__":
    main()

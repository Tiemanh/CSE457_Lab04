import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parent
METADATA = ROOT / "metadata.csv"


def main():
    with METADATA.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    labels = Counter(row["label"] for row in rows)
    durations = []
    rms_values = []
    peaks = []
    errors = []

    for row in rows:
        path = ROOT / row["file_path"]
        if not path.exists():
            errors.append(f"Thiếu file: {row['file_path']}")
            continue
        info = sf.info(path)
        x, sr = sf.read(path, dtype="float32", always_2d=False)
        if info.samplerate != 16000:
            errors.append(f"Sai sample rate: {row['file_path']} = {info.samplerate}")
        if info.channels != 1 or x.ndim != 1:
            errors.append(f"Không phải mono: {row['file_path']}")
        if info.subtype != "PCM_16":
            errors.append(f"Sai subtype: {row['file_path']} = {info.subtype}")
        if len(x) == 0:
            errors.append(f"File rỗng: {row['file_path']}")
            continue
        durations.append(len(x) / sr)
        rms_values.append(float(np.sqrt(np.mean(x * x) + 1e-12)))
        peaks.append(float(np.max(np.abs(x))))

    if len(rows) != 48:
        errors.append(f"Số dòng metadata phải là 48, hiện có {len(rows)}")
    if len(labels) != 16:
        errors.append(f"Số nhãn phải là 16, hiện có {len(labels)}")
    for label, count in labels.items():
        if count != 3:
            errors.append(f"Nhãn '{label}' có {count} candidate, cần 3")
    if peaks and max(peaks) >= 0.999:
        errors.append("Phát hiện clipping hoặc peak sát 0 dBFS")

    report = {
        "status": "PASS" if not errors else "FAIL",
        "wav_files": len(durations),
        "unique_labels": len(labels),
        "candidates_per_label": dict(sorted(labels.items())),
        "sample_rate_hz": 16000,
        "channels": 1,
        "subtype": "PCM_16",
        "duration_s": {
            "min": min(durations),
            "mean": float(np.mean(durations)),
            "max": max(durations),
        },
        "rms": {
            "min": min(rms_values),
            "mean": float(np.mean(rms_values)),
            "max": max(rms_values),
        },
        "peak_max": max(peaks),
        "errors": errors,
    }
    (ROOT / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].hist(durations, bins=10, color="#2563eb", edgecolor="white")
    axes[0].set_title("Phân bố thời lượng unit")
    axes[0].set_xlabel("Thời lượng (giây)")
    axes[0].set_ylabel("Số file")
    axes[0].grid(alpha=0.2)
    axes[1].hist(rms_values, bins=10, color="#16a34a", edgecolor="white")
    axes[1].set_title("Phân bố RMS")
    axes[1].set_xlabel("RMS")
    axes[1].set_ylabel("Số file")
    axes[1].grid(alpha=0.2)
    fig.suptitle("Lab 4 - Kiểm tra inventory tiếng Việt (48 WAV)")
    fig.tight_layout()
    fig.savefig(ROOT / "inventory_stats.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

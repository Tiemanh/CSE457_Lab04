from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from Lab04 import SR, estimate_f0_track, rms, stft_matrix


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
MODEL = ROOT / "models" / "phowhisper-tiny"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def normalize_words(text: str) -> list[str]:
    digit = {"0":"không","1":"một","2":"hai","3":"ba","4":"bốn",
             "5":"năm","6":"sáu","7":"bảy","8":"tám","9":"chín"}
    text = unicodedata.normalize("NFC", str(text).lower())
    text = re.sub(r"[^0-9a-zà-ỹđ\s]", " ", text, flags=re.IGNORECASE)
    words = []
    for token in text.split():
        words.extend(digit[c] for c in token) if token.isdigit() else words.append(token)
    return words


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, start=1):
        current = [i]
        for j, hyp in enumerate(hypothesis, start=1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (ref != hyp)))
        previous = current
    return previous[-1]


def rank01(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    rank = values.rank(method="average", pct=True)
    return rank if higher_is_better else 1.0 - rank + 1.0 / len(values)


device = 0 if torch.cuda.is_available() else -1
processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL, local_files_only=True)
if device == 0:
    model = model.to("cuda")
model.eval()
asr = pipeline(
    "automatic-speech-recognition", model=model,
    tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor,
    device=device,
)

manifest = pd.read_csv(RESULTS / "blind_manifest_private.csv")
sentences = pd.read_csv(RESULTS / "normalized_sentences.csv").set_index("sentence_id")
rows = []
for _, item in manifest.iterrows():
    path = ROOT / item["source_file"]
    x, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    prediction = asr(
        {"array": x, "sampling_rate": sample_rate},
        generate_kwargs={"language": "vi", "task": "transcribe"},
    )["text"].strip()
    reference = normalize_words(sentences.loc[item["sentence_id"], "spoken_text"])
    hypothesis = normalize_words(prediction)
    distance = edit_distance(reference, hypothesis)
    correct = max(0, len(reference) - distance)

    spectrum = np.abs(stft_matrix(x, n_fft=512, hop_length=128)) + 1e-9
    normalized = spectrum / np.maximum(np.linalg.norm(spectrum, axis=0, keepdims=True), 1e-9)
    flux = float(np.mean(np.linalg.norm(np.diff(normalized, axis=1), axis=0)))
    flatness = float(np.mean(np.exp(np.mean(np.log(spectrum), axis=0)) / np.mean(spectrum, axis=0)))
    click_index = float(np.percentile(np.abs(np.diff(x)), 99.9) / max(rms(x), 1e-9))
    f0 = estimate_f0_track(x)
    voiced = f0[np.isfinite(f0)]
    f0_cv = float(np.std(voiced) / max(np.mean(voiced), 1e-9)) if len(voiced) else 1.0
    rows.append({
        "clip_id": item["clip_id"], "sentence_id": item["sentence_id"], "system": item["system"],
        "reference": " ".join(reference), "asr_transcript": prediction,
        "word_edit_distance": distance, "intelligibility_asr_percent": 100 * correct / len(reference),
        "spectral_flux": flux, "spectral_flatness": flatness,
        "click_index": click_index, "f0_cv": f0_cv,
    })

clips = pd.DataFrame(rows)
intelligibility_rank = rank01(clips["intelligibility_asr_percent"], True)
flatness_balance = -np.abs(np.log10(np.maximum(clips["spectral_flatness"], 1e-6)) - np.log10(0.08))
f0_balance = -np.abs(clips["f0_cv"] - 0.15)
naturalness01 = 0.55 * intelligibility_rank + 0.25 * rank01(flatness_balance, True) + 0.20 * rank01(f0_balance, True)
continuity01 = 0.55 * rank01(clips["click_index"], False) + 0.45 * rank01(clips["spectral_flux"], False)
clips["naturalness_proxy_1_5"] = 1.0 + 4.0 * naturalness01
clips["continuity_proxy_1_5"] = 1.0 + 4.0 * continuity01
clips.to_csv(RESULTS / "automated_proxy_clip_scores.csv", index=False, encoding="utf-8-sig")

summary = clips.groupby("system", as_index=False).agg(
    intelligibility_asr_percent=("intelligibility_asr_percent", "mean"),
    naturalness_proxy_1_5=("naturalness_proxy_1_5", "mean"),
    continuity_proxy_1_5=("continuity_proxy_1_5", "mean"),
    mean_word_edit_distance=("word_edit_distance", "mean"),
)
summary["evaluation_type"] = "AUTOMATED_PROXY_NOT_HUMAN_MOS"
summary.to_csv(RESULTS / "automated_proxy_summary.csv", index=False, encoding="utf-8-sig")

objective = pd.read_csv(RESULTS / "backend_comparison.csv")
name_map = {"concat":"Concatenation", "hmm":"HMM/parametric demo", "neural":"Neural MMS-VITS"}
completed = objective[["method","latency_mean_s","storage_bytes","notes"]].copy()
for _, row in summary.iterrows():
    mask = completed["method"] == name_map[row["system"]]
    completed.loc[mask, "intelligibility_asr_percent"] = row["intelligibility_asr_percent"]
    completed.loc[mask, "naturalness_proxy_1_5"] = row["naturalness_proxy_1_5"]
    completed.loc[mask, "continuity_proxy_1_5"] = row["continuity_proxy_1_5"]
completed["evaluation_type"] = "AUTOMATED_PROXY_NOT_HUMAN_MOS"
completed.to_csv(RESULTS / "backend_comparison_automated_proxy.csv", index=False, encoding="utf-8-sig")

plot = summary.set_index("system").reindex(["concat","hmm","neural"])
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
colors = ["#2563eb", "#f59e0b", "#16a34a"]
axes[0].bar(plot.index, plot["intelligibility_asr_percent"], color=colors)
axes[0].set(title="Intelligibility proxy bằng PhoWhisper", ylabel="Phần trăm", ylim=(0, 100))
x = np.arange(len(plot)); width = 0.34
axes[1].bar(x-width/2, plot["naturalness_proxy_1_5"], width, label="Naturalness proxy")
axes[1].bar(x+width/2, plot["continuity_proxy_1_5"], width, label="Continuity proxy")
axes[1].set_xticks(x, plot.index); axes[1].set_ylim(0, 5)
axes[1].set(title="Heuristic proxy (không phải MOS)", ylabel="Điểm 1-5")
axes[1].legend()
for ax in axes:
    ax.grid(axis="y", alpha=0.2)
fig.tight_layout()
fig.savefig(FIGURES / "14_automated_proxy_evaluation.png", dpi=180, bbox_inches="tight")
plt.close(fig)

print(summary.to_string(index=False))

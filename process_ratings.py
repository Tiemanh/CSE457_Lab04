from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
INPUT = RESULTS / "ratings_completed"
FIGURES = ROOT / "figures"


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


files = sorted(INPUT.glob("*.csv"))
if not files:
    raise SystemExit(f"Chưa có ratings CSV trong {INPUT}")

ratings = pd.concat([pd.read_csv(path, encoding="utf-8-sig") for path in files], ignore_index=True)
required = {"listener_id","clip_id","transcription","naturalness_1_5","continuity_1_5"}
missing = required - set(ratings.columns)
if missing:
    raise ValueError(f"Ratings thiếu cột: {sorted(missing)}")

manifest = pd.read_csv(RESULTS / "blind_manifest_private.csv")
sentences = pd.read_csv(RESULTS / "normalized_sentences.csv").set_index("sentence_id")
ratings = ratings.merge(manifest[["clip_id","sentence_id","system"]], on="clip_id", how="left", validate="many_to_one")
if ratings[["sentence_id","system"]].isna().any().any():
    raise ValueError("Có clip_id không tồn tại trong manifest")

def score_row(row):
    reference = normalize_words(sentences.loc[row["sentence_id"], "spoken_text"])
    hypothesis = normalize_words(row["transcription"])
    distance = edit_distance(reference, hypothesis)
    correct = max(0, len(reference) - distance)
    return pd.Series({"reference_words": len(reference), "edit_distance": distance,
                      "correct_words": correct, "intelligibility_percent": 100*correct/len(reference)})

ratings = pd.concat([ratings, ratings.apply(score_row, axis=1)], axis=1)
ratings["naturalness_1_5"] = pd.to_numeric(ratings["naturalness_1_5"], errors="raise")
ratings["continuity_1_5"] = pd.to_numeric(ratings["continuity_1_5"], errors="raise")
if not ratings["naturalness_1_5"].between(1,5).all() or not ratings["continuity_1_5"].between(1,5).all():
    raise ValueError("Điểm MOS phải trong khoảng 1-5")

ratings.to_csv(RESULTS / "ratings.csv", index=False, encoding="utf-8-sig")
summary = ratings.groupby("system", as_index=False).agg(
    listeners=("listener_id", "nunique"), clips=("clip_id", "count"),
    intelligibility_percent=("intelligibility_percent", "mean"),
    intelligibility_std=("intelligibility_percent", "std"),
    naturalness_mos=("naturalness_1_5", "mean"), naturalness_std=("naturalness_1_5", "std"),
    continuity_mos=("continuity_1_5", "mean"), continuity_std=("continuity_1_5", "std"),
)
summary.to_csv(RESULTS / "listening_summary.csv", index=False, encoding="utf-8-sig")

comparison = pd.read_csv(RESULTS / "backend_comparison.csv")
for column in ["intelligibility_percent", "naturalness_mos", "continuity_mos"]:
    comparison[column] = pd.to_numeric(comparison[column], errors="coerce")
name_map = {"concat":"Concatenation", "hmm":"HMM/parametric demo", "neural":"Neural MMS-VITS"}
for _, row in summary.iterrows():
    mask = comparison["method"] == name_map[row["system"]]
    comparison.loc[mask, "intelligibility_percent"] = row["intelligibility_percent"]
    comparison.loc[mask, "naturalness_mos"] = row["naturalness_mos"]
    comparison.loc[mask, "continuity_mos"] = row["continuity_mos"]
comparison.to_csv(RESULTS / "backend_comparison_completed.csv", index=False, encoding="utf-8-sig")

plot = summary.set_index("system").reindex(["concat","hmm","neural"])
fig, axes = plt.subplots(1,2,figsize=(11,4.5))
axes[0].bar(plot.index, plot["intelligibility_percent"], color=["#2563eb","#f59e0b","#16a34a"])
axes[0].set(title="Intelligibility", ylabel="Phần trăm", ylim=(0,100))
x=np.arange(len(plot));w=0.34
axes[1].bar(x-w/2,plot["naturalness_mos"],w,label="Naturalness")
axes[1].bar(x+w/2,plot["continuity_mos"],w,label="Continuity")
axes[1].set_xticks(x,plot.index);axes[1].set(title="MOS mini",ylabel="Điểm 1-5",ylim=(0,5));axes[1].legend()
for ax in axes: ax.grid(axis="y",alpha=.2)
fig.tight_layout();fig.savefig(FIGURES/"14_subjective_evaluation.png",dpi=180,bbox_inches="tight");plt.close(fig)

listeners = int(ratings["listener_id"].nunique())
status = "COMPLETE" if listeners >= 5 else "NEED_MORE_LISTENERS"
print(f"{status}: {listeners} người nghe, {len(ratings)} lượt đánh giá")
print(summary.to_string(index=False))

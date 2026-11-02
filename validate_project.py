import json
from pathlib import Path

import pandas as pd
import soundfile as sf


ROOT = Path(__file__).resolve().parent
errors = []


def require(path: Path):
    if not path.exists():
        errors.append(f"Thiếu: {path.relative_to(ROOT)}")


units = list((ROOT / "lab4_data" / "inventory" / "units").glob("*.wav"))
if len(units) != 48:
    errors.append(f"Cần 48 unit WAV, hiện có {len(units)}")
for path in units:
    info = sf.info(path)
    if (info.samplerate, info.channels, info.subtype) != (16000, 1, "PCM_16"):
        errors.append(f"Sai format: {path.name}")

for sid in ["S01", "S02", "S03"]:
    for filename in ["raw_concat.wav", "best_concat.wav", "hmm_tts.wav", "neural_tts.wav"]:
        require(ROOT / "audio_output" / sid / filename)

for filename in ["raw_concat.wav", "best_concat.wav", "hmm_tts.wav", "neural_tts.wav"]:
    require(ROOT / "audio_output" / filename)

for path in (ROOT / "audio_output").rglob("*.wav"):
    info = sf.info(path)
    if info.frames <= 0:
        errors.append(f"Audio rỗng: {path.relative_to(ROOT)}")
    if info.channels != 1 or info.subtype != "PCM_16":
        errors.append(f"Sai format output: {path.relative_to(ROOT)}")

figures = list((ROOT / "figures").glob("*.png"))
if len(figures) < 5:
    errors.append(f"Cần ít nhất 5 hình, hiện có {len(figures)}")

required_results = [
    "inventory_validation.csv", "normalized_sentences.csv", "candidate_costs.csv",
    "unit_selection_comparison.csv", "crossfade_results.csv", "prosody_results.csv",
    "hmm_state_sequence.csv", "hmm_tracks.csv", "neural_latency_summary.csv",
    "backend_comparison.csv", "ratings_template.csv", "run_manifest.json",
    "concat_latency_summary.csv", "hmm_latency_summary.csv", "experiment_summary.json",
]
for filename in required_results:
    require(ROOT / "results" / filename)

require(ROOT / "models" / "mms-tts-vie" / "model.safetensors")
require(ROOT / "models" / "mms-tts-vie" / "vocab.json")
require(ROOT / "models" / "phowhisper-tiny" / "pytorch_model.bin")
require(ROOT / "listening_test.html")
require(ROOT / "process_ratings.py")
require(ROOT / "automated_proxy_evaluation.py")
require(ROOT / "results" / "automated_proxy_summary.csv")
require(ROOT / "results" / "backend_comparison_automated_proxy.csv")
require(ROOT / "figures" / "14_automated_proxy_evaluation.png")

dp_path = ROOT / "results" / "dp_verification.json"
require(dp_path)
if dp_path.exists() and not json.loads(dp_path.read_text(encoding="utf-8"))["pass"]:
    errors.append("Dynamic programming không khớp exhaustive search")

validation_path = ROOT / "results" / "inventory_validation.csv"
if validation_path.exists():
    table = pd.read_csv(validation_path)
    if not (table["status"] == "PASS").all():
        errors.append("Inventory có file FAIL")

result = {
    "status": "PASS" if not errors else "FAIL",
    "unit_wav_count": len(units),
    "figure_count": len(figures),
    "blind_clip_count": len(list((ROOT / "audio_output" / "blind_listening").glob("*.wav"))),
    "errors": errors,
}
(ROOT / "results" / "project_validation.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(result, ensure_ascii=False, indent=2))
raise SystemExit(0 if not errors else 1)

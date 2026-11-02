# Danh mục nội dung đã chuẩn bị cho báo cáo

## Hình

1. `figures/01_inventory_stats.png` - Histogram duration và RMS, dùng cho mục A.
2. `figures/02_baseline_waveform_spectrogram.png` - Baseline có đánh dấu biên, dùng cho mục B.
3. `figures/03_best_concat_waveform_spectrogram.png` - Kết quả unit selection và smoothing, dùng cho mục C.
4. `figures/04_boundary_before_after.png` - Phóng to hai biên trước/sau smoothing.
5. `figures/05_crossfade_experiment.png` - So sánh 0/5/15/30 ms.
6. `figures/06_unit_selection_cost.png` - Total cost baseline và dynamic programming.
7. `figures/07_f0_verification.png` - Kiểm chứng F0 xấp xỉ Fs/Np.
8. `figures/08_duration_experiment.png` - Spectrogram duration 0.8/1.0/1.2.
9. `figures/09_pitch_experiment.png` - Spectrogram pitch -2/0/+2 semitone.
10. `figures/10_hmm_tracks.png` - F0, F1, F2 và energy theo state.
11. `figures/11_hmm_parameter_scaling.png` - HMM duration/F0 scaling.
12-14. `figures/12_neural_spectrogram_S01..S03.png` - Acoustic representation của neural model.
15. `figures/13_backend_objective_comparison.png` - Latency và dung lượng.
16. `figures/14_automated_proxy_evaluation.png` - Đánh giá tự động thay thế khi không có người nghe; phải ghi rõ không phải MOS.

PDF chỉ yêu cầu ít nhất 5 hình. Nên chọn khoảng 7-9 hình quan trọng để báo cáo không quá dày.

## Bảng

- `results/inventory_validation.csv`: kiểm tra 48 WAV.
- `results/normalized_sentences.csv`: input và token sau normalization.
- `results/candidate_costs.csv`: target/join/cumulative cost.
- `results/unit_selection_comparison.csv`: baseline so với DP.
- `results/crossfade_results.csv`: thí nghiệm 0/5/15/30 ms.
- `results/prosody_results.csv`: F0, duration và pitch.
- `results/hmm_state_sequence.csv`: state duration và Gaussian parameters.
- `results/hmm_tracks.csv`: trajectory theo frame.
- `results/neural_latency_summary.csv`: latency trung bình và độ lệch chuẩn.
- `results/backend_comparison.csv`: khung so sánh ba phương pháp.

## Audio

- `audio_output/raw_concat.wav` và `best_concat.wav`.
- `audio_output/hmm_tts.wav` và `neural_tts.wav`.
- `audio_output/experiments/crossfade/`: bốn mức cross-fade.
- `audio_output/experiments/prosody/`: sáu biến thể duration/pitch.
- `audio_output/experiments/hmm/`: năm biến thể HMM.
- `audio_output/blind_listening/`: chín clip đã đổi tên mù.

## Lựa chọn đánh giá

Phương án đúng theo gợi ý của đề là mời 5-10 người nghe. Nếu không tổ chức người nghe, dự án đã có `results/backend_comparison_automated_proxy.csv` và hình 14. Trong báo cáo phải gọi đây là đánh giá tự động/proxy, không gọi là MOS hoặc listening test của con người. Sau đó chỉ còn viết báo cáo 5-8 trang và trả lời 10 câu hỏi lý thuyết.

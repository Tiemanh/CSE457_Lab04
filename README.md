# CSE457_Lab04 - Tổng hợp tiếng nói

Báo cáo chính thức được đặt tại `Lab04.docx` và `Lab04.pdf`. Mọi audio, bảng và hình dùng trong báo cáo được sinh và kiểm chứng từ mã nguồn trong repository.

> Lưu ý: các điểm tự nhiên/liên tục trong kết quả là chỉ số proxy tự động, không phải MOS.

## Chạy toàn bộ bài

```powershell
python -X utf8 Lab04.py
```

Chương trình chạy lần lượt:

1. Kiểm tra 48 WAV, metadata, format, clipping, duration, RMS và DC offset.
2. Chuẩn hóa ba câu test và tạo baseline concatenation.
3. Tính target cost, join cost, chọn unit bằng dynamic programming và kiểm chứng với exhaustive search.
4. Thử cross-fade 0/5/15/30 ms và tạo `best_concat.wav`.
5. Đo F0 bằng chu kỳ và autocorrelation; thử duration 0.8/1.0/1.2 và pitch -2/0/+2 semitone.
6. Chạy HMM demo ở chế độ generation, xuất state sequence, F0/F1/F2/energy trajectory và waveform minh họa.
7. Chạy model neural TTS tiếng Việt MMS-VITS từ file cục bộ với `local_files_only=True`, sinh ba câu và đo latency 5 lần/câu.
8. Tạo bảng so sánh khách quan và đánh giá proxy tự động bằng PhoWhisper cùng các chỉ số tín hiệu.

## Môi trường

- Python 3.10 trở lên
- `pip install -r requirements.txt`
- GPU không bắt buộc; nếu có CUDA, model neural tự sử dụng GPU.
- Model đã nằm trong `models/mms-tts-vie`, vì vậy inference không cần mạng.

## Ba câu test

- S01: Mời khách hàng số 23 đến quầy 5.
- S02: Mời khách hàng số 406 đến quầy 7.
- S03: Mời khách hàng số 189 đến quầy 0.

Chuỗi số được đọc từng chữ số theo yêu cầu của miền hẹp.

## Đầu ra chính

- `audio_output/raw_concat.wav`
- `audio_output/best_concat.wav`
- `audio_output/hmm_tts.wav`
- `audio_output/neural_tts.wav`
- `audio_output/S01..S03/`: kết quả ba backend cho cùng ba câu
- `audio_output/experiments/`: cross-fade, pitch, duration và HMM scaling
- `figures/`: 15 hình đã sẵn sàng cho báo cáo
- `results/`: cost, latency, state/trajectory, validation và kết quả đánh giá proxy tự động

## Giới hạn cần công bố

Inventory 48 word-unit là giọng tổng hợp `vi-VN-HoaiMyNeural`, không phải giọng người tự thu. Dữ liệu đạt định dạng kỹ thuật của bài và phù hợp để thực hiện thuật toán, nhưng cần công bố đúng nguồn. Nếu giảng viên bắt buộc dữ liệu người thật, giữ nguyên chương trình và thay các WAV bằng bản thu có sự đồng ý.

HMM backend là mô hình tham số có hướng dẫn cho mục quan sát state/trajectory. Waveform HMM là resynthesis minh họa từ F0, formant và energy, không phải hệ HTS đã huấn luyện từ corpus lớn.

Chạy `python -X utf8 automated_proxy_evaluation.py` để đánh giá. Kết quả dùng PhoWhisper cho độ hiểu ASR và heuristic tín hiệu cho naturalness/continuity. Các cột được ghi rõ là proxy tự động, không phải MOS.

## Kiểm chứng

- Dynamic programming khớp exhaustive search: xem `results/dp_verification.json`.
- Model neural chỉ đọc từ thư mục local: xem hàm `run_neural_tts()`.
- Mỗi đầu ra được ghi checksum SHA-256 trong `results/run_manifest.json`.
- Toàn bộ inventory được kiểm tra trong `results/inventory_validation.csv`.

## Bản quyền và sử dụng

Model `facebook/mms-tts-vie` thuộc dự án MMS của Meta và dùng giấy phép CC BY-NC 4.0. Chỉ dùng dự án này cho mục đích học tập/phi thương mại và trích dẫn nguồn trong báo cáo.

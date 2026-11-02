# Bộ dữ liệu mẫu Lab 4 - Tổng hợp tiếng nói

Bộ dữ liệu này gồm 16 từ tiếng Việt trong miền hẹp thông báo xếp hàng, mỗi từ có 3 candidate, tổng cộng 48 file WAV.

## Nội dung

Các unit: `mời, khách, hàng, số, không, một, hai, ba, bốn, năm, sáu, bảy, tám, chín, đến, quầy`.

Định dạng đầu ra:

- WAV mono
- 16 kHz
- PCM 16-bit
- Đã cắt khoảng lặng thừa và cân RMS có kiểm soát

Ba candidate sử dụng cùng giọng `vi-VN-HoaiMyNeural`, với biến thiên nhẹ về tốc độ và RMS để phục vụ thí nghiệm unit selection. Trường `pitch_setting` trong metadata được giữ ở `+0Hz`; pitch shift của mục D cần được thực hiện riêng trên audio sau khi tạo inventory.

## Công bố nguồn dữ liệu

Đây là **giọng tổng hợp thần kinh**, được tạo bằng dịch vụ Microsoft Edge TTS qua thư viện `edge-tts`. Đây không phải bản ghi trực tiếp của một người nói và không được mô tả là dữ liệu tự thu.

Bộ dữ liệu phù hợp để:

- phát triển và kiểm thử text normalization;
- baseline concatenation;
- unit selection và join cost;
- cross-fade;
- thử nghiệm pitch/duration ở mức minh họa.

Để tối đa điểm phần thu dữ liệu, nên dùng cùng cấu trúc và thay các file này bằng bản thu của chính sinh viên hoặc người đã đồng ý thu âm. Không nên trích hoặc bắt chước giọng của người trong video khi chưa có sự cho phép.

## Cấu trúc

```text
lab4_vietnamese_dataset/
  units/                      # 48 WAV
  metadata.csv                # thông tin kỹ thuật và checksum
  generation_manifest.json   # cấu hình sinh dữ liệu
  README.md
```

## Trích dẫn trong báo cáo

Có thể ghi: “Inventory mẫu được tạo bằng giọng tổng hợp tiếng Việt `vi-VN-HoaiMyNeural`, sau đó chuyển về mono 16 kHz, PCM 16-bit, cắt silence và chuẩn hóa RMS có kiểm soát. Dữ liệu được dùng cho mục đích học thuật và được công bố là dữ liệu tổng hợp.”

## Nguồn dữ liệu người thật thay thế

Nếu cần dữ liệu giọng người thật có giấy phép rõ ràng, có thể dùng VIVOS (CC BY-NC-SA 4.0) hoặc Mozilla Common Voice. Hai bộ này chứa audio cấp câu và nhiều người nói, nên cần chọn một speaker, tách word và kiểm tra thủ công; chúng không sẵn ở dạng 16 unit x 3 candidate của bài lab.

# Hướng dẫn listening test mù

1. Mở `listening_test.html` bằng Chrome hoặc Edge.
2. Không cho người nghe xem `results/blind_manifest_private.csv`.
3. Người nghe nhập mã L01, L02... và hoàn thành đủ chín clip.
4. Nhấn “Kiểm tra và tải CSV”.
5. Chép file tải về vào `results/ratings_completed/`.
6. Lặp lại cho 5-10 người, dùng cùng tai nghe và mức âm lượng.
7. Chạy `python -X utf8 process_ratings.py`.

Naturalness và continuity là hai câu hỏi riêng. Intelligibility được tính từ số từ nghe đúng hoặc số câu đúng; phải ghi rõ quy tắc chấm trong báo cáo.

Không chỉnh sửa `blind_manifest_private.csv` và không đưa file này cho người tham gia.

Chương trình sẽ tự tạo `ratings.csv`, `listening_summary.csv`, `backend_comparison_completed.csv` và `figures/14_subjective_evaluation.png`. Nếu có dưới năm mã người nghe duy nhất, chương trình vẫn tính kết quả nhưng báo `NEED_MORE_LISTENERS`.

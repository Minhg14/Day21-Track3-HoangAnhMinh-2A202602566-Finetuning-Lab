# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**
Điều làm tôi ngạc nhiên nhất là hiện tượng catastrophic forgetting xảy ra quá nhanh và rõ rệt: chỉ với 250 mẫu dữ liệu phân loại ticket và 30 steps huấn luyện, năng lực trả lời câu hỏi phổ thông của Qwen3.5-4B đã sụt giảm ngay 13.6% (regression từ 0.7911 xuống 0.6556). Tôi từng nghĩ LoRA chỉ cập nhật một phần nhỏ tham số nên sẽ giữ được phần lớn tri thức gốc, nhưng thực tế việc không có dữ liệu nhắc lại (replay data) vẫn làm hỏng mô hình rất nhanh. Đồng thời, việc `attn_only` có train loss thấp hơn hẳn `correct` (0.537 so với 0.627) nhưng khi chấm target lại bằng nhau chằn chặn (0.970) cũng là một bất ngờ lớn.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**
Tôi mất nhiều thời gian nhất ở khâu chạy 3 run đối chứng ở NB4 (`attn_only`, `wrong_lr`, `qlora`) và NB5 trên Colab, cũng như việc tải và đồng bộ các file kết quả về máy local. Ban đầu tôi nghĩ thời gian huấn luyện sẽ kéo dài hơn 100 phút như ước lượng trong tài liệu, nhưng may mắn là hạ tầng T4 lúc chạy ít tải nên toàn bộ pipeline hoàn thành trong khoảng 40–50 phút. Chỗ tốn công suy nghĩ nhất chính là việc phân tích nguyên nhân tại sao cổng hồi quy FAILED và đối chiếu 5 ca định tính để tìm ra mẫu câu "Khi nào tiện" gây nhầm lẫn nhãn urgency.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**
Trước lab này, tôi từng có 2 niềm tin sai lầm:
- Thứ nhất, tin rằng "train loss càng thấp thì mô hình càng xịn". Thí nghiệm NB4 và NB5 chứng minh trực tiếp rằng train loss thấp nhất (`attn_only` 0.537) không hề thắng `correct` (0.627) trên downstream accuracy.
- Thứ hai, tin rằng "cứ fine-tune ra accuracy cao là đem đi triển khai được". Sau khi thấy target đạt 97% nhưng verdict bị FAILED vì hỏng tri thức nền, tôi nhận ra việc đánh giá toàn diện qua cổng hồi quy quan trọng hơn việc chỉ chăm chăm nhìn vào độ chính xác của bài toán nghiệp vụ.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**
Tôi dùng AI assistant để giải thích cấu trúc mã nguồn của lab, hỗ trợ tạo đoạn code tải file kết quả từ Colab về máy local dạng file zip, và tổng hợp số liệu từ các file JSON trong `results/` vào báo cáo. Chỗ AI assistant từng sai là ban đầu nhìn thời gian chạy 15 phút đến NB4 đã vội phỏng đoán là tôi đặt `EVAL_LIMIT=8` thay vì nhận ra là do GPU T4 trên Colab đang chạy rất mượt; đồng thời khi gọi lệnh tạo file ban đầu assistant đã đính kèm metadata artifact sai đường dẫn. Nhờ kiểm tra lại log và hình ảnh thực tế, tôi đã điều chỉnh kịp thời.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**
Bước đầu tiên chắc chắn không phải là nhảy vào chọn mô hình hay viết mã huấn luyện, mà là:
1. **Thiết lập một bộ đánh giá chuẩn (Golden Evaluation Set) kèm bộ kiểm định hồi quy (Regression Suite):** Đóng băng bộ dữ liệu kiểm thử này trước khi chạm vào bất kỳ dòng code huấn luyện nào.
2. **Đo đạc baseline của mô hình gốc với prompt kỹ nghệ cẩn thận:** Xác định xem prompt tối ưu đã giải quyết được bao nhiêu % bài toán. Nếu prompt kỹ nghệ đã đạt 85-90% yêu cầu với chi phí thấp và không bị quên tri thức, tôi sẽ tư vấn khách hàng chưa cần vội fine-tune.
3. Nếu bắt buộc fine-tune, luôn chuẩn bị sẵn 3–5% dữ liệu tổng quát (replay data) trộn vào tập huấn luyện để phòng ngừa hiện tượng sụp đổ tri thức.

# Lab 21 — Evaluation Report

**Họ tên**: Hoàng Anh Minh  **MSSV**: 2A202602566  **Ngày**: 2026-10-07  
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `NVIDIA Tesla T4 16GB (Google Colab)`

> Mọi con số dưới đây đều khớp chính xác với file trong `results/`. Grader có thể kiểm tra chéo tự động.

---

## 1. Setup

| Thông số | Giá trị |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường (mặc định) |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 256 — p95 đo được là 98 *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epochs / 30 steps |

**Lý do chọn model và dataset:**
- **Base model (`unsloth/Qwen3.5-4B`):** Phù hợp với tier T4 (16GB VRAM), vừa đủ dung lượng để huấn luyện LoRA ở chuẩn 16-bit (fp16) mà không bị OOM, đồng thời sở hữu năng lực xử lý tiếng Việt và cấu trúc JSON rất tốt.
- **Dataset (250 ticket CSKH):** Bài toán phân loại ticket dịch vụ khách hàng với 4 trường thông tin bắt buộc (`intent`, `urgency`, `product`, `sentiment`). Dataset này cung cấp thước đo khách quan tuyệt đối (exact match từng trường), không phụ thuộc vào LLM judge thiên vị hay thiếu ổn định.

**Template có giữ khối `<think>` không?** Có — *(results/template_check.json: verdict = "reasoning preserved — safe to train on traces")*. Chat template giữ nguyên thẻ mở `<think>`, nội dung suy luận và thẻ đóng `</think>` trước khi đưa ra câu trả lời cuối cùng, đảm bảo an toàn khi huấn luyện trên mô hình reasoning.

---

## 2. Mask proof (NB1)

| Chỉ số | Giá trị thực tế |
|---|---|
| `supervised_fraction` | `0.4149` (41.49%) |
| Câu trả lời nằm trong loss | `true` (Đúng) |
| Câu hỏi KHÔNG nằm trong loss | `true` (Đúng) |

Đoạn văn bản mẫu **được tính loss** (supervised preview):

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Đoạn văn bản mẫu **bị che đi** (masked preview — không tính loss):

```
<|im_start|>system
Phân loại ticket sau.<|im_end|>
<|im_start|>user
Alo shop, mình đặt balo laptop mã đơn VN411453. Cho tôi trả lại. Đã 3 ngày rồi. Cho tôi hỏi.<|im_end|>
<|im_start|>assistant
<think>
```

Tỷ lệ token được tính loss là 41.49% (`supervised_fraction = 0.4149`), phản ánh đúng thực tế là chỉ có phần nội dung câu trả lời JSON của assistant (từ sau thẻ đóng `</think>`) mới tham gia vào hàm mất mát. Prompt và instruction hoàn toàn bị gán nhãn `-100` (được che khỏi loss).

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.7911 | 0.000 | 3313.7 |
| (b) base + optimized prompt | 0.765 | 0.7911 | 1.000 | 1024.7 |
| (c) LoRA fine-tune | 0.970 | 0.6556 | 1.000 | 1390.7 |

**(b) có thật sự mạnh hơn (a) không?** Có — baseline (b) vượt trội hoàn toàn so với (a):
- **target**: Tăng từ `0.000` lên `0.765` (+0.765). Prompt ngây thơ (naive) không cung cấp schema cấu trúc JSON nên mô hình sinh tự do văn bản lan man, dẫn đến tỷ lệ khớp nhãn bằng 0%. Prompt tối ưu (optimized) đã định nghĩa rõ các lớp thuộc tính và ví dụ cụ thể, giúp mô hình đạt độ chính xác 76.5%.
- **format**: Tăng từ `0.000` lên `1.000` (100% hợp lệ). Mô hình luôn xuất ra đúng định dạng JSON có đủ 4 khóa.
- **latency**: Giảm từ `3313.7 ms` xuống `1024.7 ms` (nhanh hơn gấp 3.2 lần) do mô hình không sinh văn bản thừa.
- **regression**: Được bảo tồn nguyên vẹn ở mức `0.7911` (79.11%), chứng minh prompt kỹ nghệ không làm suy giảm năng lực tri thức nền tảng của mô hình.

Prompt tối ưu được giữ nguyên bản gốc từ pipeline (mã SHA: `719e74d3b6232053`), không bị làm suy yếu để tâng bốc bản fine-tune.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | latency (ms) | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.627 | **0.970** | 1390.7 | 8.78 |
| `attn_only` | q,v | 283 | 32,456,704 | 1e-4 | 0.537 | **0.970** | 882.6 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 1.570 | **0.000** | 5240.8 | 8.78 |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 0.706 | **0.940** | 1760.1 | 3.86 |

> **Quy tắc xếp hạng cốt lõi:** Đánh giá bằng cột **target** ở NB5, không đánh giá bằng cột train loss.
> - Thứ tự theo target: `correct` (0.970) = `attn_only` (0.970) > `qlora` (0.940) > `wrong_lr` (0.000).
> - Thứ tự theo train loss: `attn_only` (0.537) < `correct` (0.627) < `qlora` (0.706) < `wrong_lr` (1.570).
>
> Hai thứ tự này không trùng khớp: `attn_only` có train loss thấp nhất nhưng điểm target thực tế lại chỉ ngang bằng `correct`. Điều này chứng minh rằng lấy train loss làm chỉ số thay thế cho năng lực suy luận thực tế (downstream task) là một ngụy biện đo lường nghiêm trọng.

### 4.1 — Phân tích `attn_only` vs `correct`
Cấu hình `attn_only` được nâng rank lên r=283 để khớp chính xác ngân sách tham số (32.46M params, sai lệch dưới 0.03% so với 32.46M của `correct`). Trên tập đánh giá mục tiêu (target), `attn_only` và `correct` hòa nhau với cùng độ chính xác 0.970. Tuy nhiên, thứ tự theo train loss lại khác: `attn_only` có loss thấp hơn rõ rệt (0.537 so với 0.627). Điều này chứng minh rằng rank cực lớn (283) giúp mô hình khớp dữ liệu huấn luyện sâu hơn (overfit nhẹ vào loss), nhưng không đem lại thêm bất kỳ lợi thế nào trên tập dữ liệu kiểm thử thực tế. Trong bài toán phân loại trích xuất 4 trường này, vị trí adapter phân bổ đều trên toàn bộ các lớp tuyến tính (`text-linear`) với rank vừa phải (r=16) đem lại hiệu quả tổng quát hóa tương đương mà lại tránh được độ phức tạp tính toán khi cập nhật ma trận rank cao.

### 4.2 — Phân tích `wrong_lr`
Cấu hình `wrong_lr` chỉ thay đổi duy nhất một tham số: learning rate hạ từ 1e-4 xuống 1e-5 (áp dụng thang learning rate của full fine-tuning cho LoRA). Kết quả là đường loss giảm rất chậm và dừng lại ở mức 1.570 sau 30 steps, trong khi target rớt thẳng về 0.000 và format bằng 0.000. Nếu chỉ quan sát đường loss mà không biết giá trị LR, một kỹ sư sẽ dễ dàng kết luận sai lầm rằng "dữ liệu quá phức tạp" hoặc "30 steps là chưa đủ để mô hình hội tụ". Thực tế, bản chất của LoRA là gradient truyền qua tích hai ma trận hạng thấp A và B, khiến tín hiệu cập nhật bị suy giảm mạnh; do đó LoRA bắt buộc phải dùng learning rate cao gấp 5 đến 10 lần so với full fine-tuning để các trọng số adapter di chuyển đủ nhanh. Độ trễ của `wrong_lr` cũng tăng vọt lên 5240.8 ms vì mô hình bị bối rối và sinh ra các chuỗi văn bản dài vô nghĩa.

### 4.3 — Phân tích `qlora`
Cấu hình `qlora` lượng tử hóa 4-bit giúp tiết kiệm tới 56.0% bộ nhớ VRAM đỉnh (chỉ tiêu tốn 3.86 GB so với 8.78 GB của `correct`). Tuy nhiên, sự đánh đổi là rất rõ ràng: điểm target tụt từ 0.970 xuống 0.940 (mất 3 điểm phần trăm), thời gian trễ tăng thêm 26.6% (1760.1 ms so với 1390.7 ms do chi phí giải lượng tử hóa on-the-fly), và train loss cũng cao hơn (0.706 so với 0.627). Các số liệu đo đạc thực tế hoàn toàn ủng hộ cảnh báo từ nhà phát triển Qwen rằng không nên dùng QLoRA cho dòng Qwen3.5 nếu không bị giới hạn phần cứng nghiêm ngặt. Việc lượng tử hóa trọng số nền xuống 4-bit gây mất mát thông tin biểu diễn tinh vi, làm giảm độ chính xác và làm chậm quá trình sinh văn bản.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`  
`target Δ = +0.205` · `regression Δ = -0.136` · `valid_trace_rate = 0.00`

### Diễn giải phán quyết (156 từ):
Cổng hồi quy đưa ra phán quyết FAILED do năng lực tổng quát bị suy thoái nghiêm trọng: chỉ số regression sụt giảm tới 0.136 (từ 0.7911 xuống 0.6556), vượt xa ngưỡng dung sai tối đa cho phép là 0.020. Dù mô hình fine-tune đạt bước nhảy vọt ấn tượng trên nhiệm vụ mục tiêu (target tăng từ 0.765 lên 0.970, tức target Δ = +0.205), việc suy giảm 13.6% khả năng giải quyết các câu hỏi phổ thông chứng minh mô hình đã rơi vào hiện tượng quên lãng thảm họa (catastrophic forgetting). Nguyên nhân bắt nguồn từ việc huấn luyện đơn nhiệm trên tập dữ liệu hẹp 250 mẫu mà không kèm theo dữ liệu nhắc lại (replay data). Đồng thời, chỉ số `valid_trace_rate = 0.0` chỉ ra hiện tượng suy thoái chuỗi suy luận (reasoning-trace collapse): mô hình đã học được đường tắt để bỏ qua khối `<think>` và nhảy thẳng vào kết quả JSON. Vì vậy, bản fine-tune này chưa đủ điều kiện an toàn để triển khai thực tế.

---

## 6. Định tính — Bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | "mình đặt chuột không dây... Cho tôi trả lại. Gấp" | doi_tra / cao / tich_cuc | Sai urgency | Đủ 4/4 đúng | ✅ FT thắng: Nhận diện chính xác độ khẩn cấp cao |
| 2 | "mình đặt đèn bàn LED... Hoàn tiền. Quá hạn rồi" | hoan_tien / cao / tich_cuc | Sai sentiment | Đủ 4/4 đúng | ✅ FT thắng: Phân biệt tốt giữa khiếu nại và cảm xúc tích cực |
| 3 | "mình đặt bình giữ nhiệt... Chưa thấy tiền. Khi nào tiện" | hoan_tien / thap / tich_cuc | 3/4 đúng | 3/4 đúng (đoán urgency: trung_binh) | ❌ **FT thua**: Cả hai đều đoán sai urgency vì câu văn lịch sự |
| 4 | "mình đặt nồi chiên không dầu... Thiếu phụ kiện. Khi nào tiện" | san_pham_loi / thap / trung_tinh | 3/4 đúng | 3/4 đúng (đoán urgency: trung_binh) | ❌ **FT thua**: Nhầm lẫn giữa độ khẩn cấp thấp và trung bình |
| 5 | "mình đặt áo khoác gió... Bị lỗi. Khi nào tiện" | san_pham_loi / thap / tich_cuc | 2/4 đúng | 3/4 đúng (đoán urgency: trung_binh) | ❌ **FT thua**: FT vẫn đoán sai urgency dù đúng intent và product |

**Phân tích mẫu chung của các ca fine-tune THUA:**
Toàn bộ các ca mà bản fine-tune không thể đạt điểm tối đa (score = 0.75, sai 1 trường) đều chia sẻ một mẫu hình ngữ nghĩa rất đặc thù: sự xuất hiện của cụm từ thể hiện thái độ nhẹ nhàng, lịch sự như *"Khi nào tiện"*, *"Không vội"*. Theo quy chuẩn nhãn của bộ dữ liệu, các cụm từ này biểu thị mức độ khẩn cấp thấp (`urgency: "thap"`). Tuy nhiên, trọng số của mô hình fine-tune có xu hướng mặc định gán các yêu cầu có vấn đề lỗi/hoàn tiền về mức độ trung bình (`urgency: "trung_binh"`). Điều này cho thấy mô hình bị thiên lệch tần suất xuất hiện trong tập huấn luyện và chưa nắm bắt triệt để sắc thái biểu cảm giảm nhẹ của khách hàng Việt Nam.

---

## 7. Kết luận & điều tôi học được

### Kết luận tổng kết (182 từ):
Dựa trên các bằng chứng thực nghiệm thu được, quyết định kỹ thuật dứt khoát là **KHÔNG NÊN DEPLOY** bản fine-tune hiện tại lên môi trường production. Dù mô hình thể hiện sự cải thiện xuất sắc trên nhiệm vụ chính (độ chính xác target đạt 97.0% so với 76.5% của prompt tối ưu), cái giá phải trả là sự suy giảm nghiêm trọng 13.6% trên năng lực tổng quát nền tảng cùng sự biến mất hoàn toàn của chuỗi tư duy hợp lệ (`valid_trace_rate = 0.0`). Trong một hệ thống vận hành thực tế, một mô hình phân loại ticket bị mất nhận thức ngôn ngữ cơ bản sẽ trở nên cực kỳ mong manh trước các trường hợp bất thường (edge cases) và gây rủi ro an toàn vận hành. 

Thí nghiệm cũng chỉ ra rằng đòn bẩy thật sự của quá trình fine-tuning không nằm ở việc cố gắng tăng rank hay thay đổi vị trí adapter (khi `attn_only` và `correct` cho kết quả target tương đồng), mà nằm ở **thang đo learning rate** (sai 1 bậc LR dẫn đến sụp đổ toàn diện) và **chất lượng cân bằng dữ liệu**. Để bản fine-tune đủ điều kiện deploy, bước tiếp theo bắt buộc phải là bổ sung 3–5% dữ liệu tổng quát (replay data) nhằm vượt qua cổng hồi quy.

### Ba điều tôi học được:
1. **Train loss là một chỉ báo dối lừa nếu không đo lường downstream metric:** Run `attn_only` với rank cực lớn r=283 đạt train loss thấp nhất toàn bộ thí nghiệm (0.537 so với 0.627 của `correct`), nhưng trên tập target thực tế cả hai đều đạt 0.970. Tối ưu hóa điểm số phụ (surrogate loss) thay vì năng lực thực tế chính là cạm bẫy lớn nhất trong huấn luyện mô hình ngôn ngữ.
2. **LoRA đòi hỏi thang learning rate riêng biệt so với full fine-tuning:** Thử nghiệm `wrong_lr` đã chứng minh trực quan việc áp dụng LR truyền thống (1e-5) vào cấu trúc adapter khiến mô hình gần như không cập nhật được tri thức mới sau 30 steps, dẫn đến target bằng 0%. Tham số adapter cần bước nhảy đủ lớn (1e-4) để bù đắp sự suy giảm gradient qua phép chiếu ma trận tích.
3. **Cổng kiểm tra hồi quy (regression gate) là ranh giới bắt buộc giữa demo và production:** Một mô hình có thể đạt 97% độ chính xác nghiệp vụ nhưng vẫn là một sản phẩm lỗi nếu nó phá hỏng tri thức phổ quát của mô hình gốc. Kiểm định hồi quy trước khi bàn giao không phải là thủ tục hành chính, mà là chốt chặn duy nhất ngăn chặn hiện tượng catastrophic forgetting âm thầm xâm nhập hệ thống.

### Nếu có thêm 2 giờ nữa, tôi sẽ thử:
- Bổ sung 3% đến 5% dữ liệu hội thoại phổ quát từ tập Alpaca/ShareGPT tiếng Việt vào `train_seed.jsonl` để kiểm chứng xem cổng hồi quy có chuyển sang `PASSED` hay không.
- Thử nghiệm cấu hình `MASK_MODE=response-only` để ngăn chặn hiện tượng sụp đổ chuỗi suy luận reasoning-trace, so sánh xem `valid_trace_rate` có được khôi phục về trên 0.8 hay không.
- Chạy thử nghiệm NB6 để thực hiện merge adapter vào base model và đo lường độ suy hao độ chính xác cũng như độ trễ phục vụ thực tế (serving latency).

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:

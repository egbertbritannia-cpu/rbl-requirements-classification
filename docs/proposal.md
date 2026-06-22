# Legacy Draft Notice

This proposal is an earlier draft and is not the active repository scope. The current experiment protocol uses only `data/exp/promise_exp.csv` and focuses on reproducing/adapting baseline methodology on the PROMISE expanded dataset. See `README.md` and `docs/research_questions.md` for the current scope.

# Proposal

**"Beyond Accuracy: Enhancing the Robustness of Software Requirements Classification against Paraphrasing using Paradigm-Complementary Hybrid Transformers"** *(Vượt xa Độ chính xác: Nâng cao độ bền vững của phân loại yêu cầu phần mềm trước nhiễu loạn hành văn bằng kiến trúc Transformer lai bổ trợ nguyên lý)*

### **1. PROBLEM STATEMENT**

- **Thực trạng:** Các mô hình AI hiện tại đạt độ chính xác cao trên dữ liệu tĩnh. Tuy nhiên, trong môi trường Agile, khách hàng liên tục thay đổi cách diễn đạt (paraphrase).
- **Vấn đề:** Khi cấu trúc câu thay đổi, các AI truyền thống dễ bị "đánh lừa" dẫn đến phân loại sai giữa FR và NFR. Đáng chú ý, ngay cả các mô hình ngôn ngữ lớn (LLMs) hiện đại nhất cũng vấp phải tình trạng "ảo giác" (hallucination) và thiếu tính nhất quán khi đối mặt với dữ liệu bị paraphrase liên tục.
- **Lỗ hổng:** Thiếu vắng một khung đo lường độ bền vững (Robustness) và một kiến trúc AI có khả năng duy trì độ chính xác mà không phải phụ thuộc vào chi phí API đắt đỏ của các LLMs khổng lồ. Ngoài ra, các hệ thống hiện tại vẫn chưa làm rõ tính transparency và explainability trong quá trình phân loại FR/NFR

### **2. RESEARCH QUESTIONS**

**Research Question:** AI phân loại FR và NFR chính xác đến mức nào?

- **sub - Research Question 1:** Mô hình nào đạt hiệu năng cao nhất (So sánh Hybrid đề xuất với các Baselines truyền thống và cả State-of-the-art LLMs)?
- **sub - Research Question 2 (core):** Độ bền vững (Robustness) của các mô hình thay đổi ra sao khi yêu cầu phần mềm bị paraphrase?
- **sub - Research Question 3:** Đặc trưng ngôn ngữ nào khiến các mô hình hộp đen (bao gồm cả LLMs) dự đoán sai?

### **3. Methodology**

**3.1. core design: Hybrid ModernBERT+ ELECTRA**

Tận dụng sự **bổ trợ về cơ sở cấu trúc học (Paradigm Complementarity)**:

- ModernBERT (Cơ chế MLM): Xử lý ngữ cảnh dài, "bình thường hóa" cấu trúc ngữ pháp bị đảo lộn.
- ELECTRA (Cơ chế RTD): Đóng vai trò làm radar rà soát từ khóa (token-level), bắt chặt các ràng buộc bất di bất dịch của NFR (*encrypted, latency*).

> ModernBERT và ELECTRA được thiết kế dựa trên các pretraining objectives khác nhau. ModernBERT nổi bật ở khả năng biểu diễn ngữ nghĩa và xử lý ngữ cảnh dài, trong khi ELECTRA nổi bật ở khả năng học tín hiệu ngôn ngữ hiệu quả thông qua Replaced Token Detection. Do đó, nghiên cứu giả thuyết rằng hai encoder tạo ra các biểu diễn bổ sung (complementary representations). Thay vì sử dụng Naive Concatenation như các nghiên cứu trước, một cơ chế Gated Fusion được đề xuất nhằm học động mức độ đóng góp của từng encoder cho mỗi software requirement.
> 

```
ModernBERT + ELECTRA + Adaptive Gated Fusion
```

  ****

![image.png](image.png)

**3.2. Robustness**

**3.3. XAI +** 

1. Bước 1: Trích xuất các ca thất bại (Error Logging Pipeline)
    
    Đầu tiên, hệ thống cần một luồng xử lý đầu ra rành mạch. Sau khi chạy đánh giá trên tập test, viết một module để so sánh nhãn thực tế (`GroundTruth`) và nhãn dự đoán (`Prediction`).
    Tất cả các trường hợp sai lệch (ví dụ: thực tế là NFR nhưng máy đoán là FR) sẽ được tự động đóng gói thành các đối tượng JSON và lưu vào một tệp log riêng (vd: `mispredictions_log.json`). Đối tượng này nên chứa: câu gốc, câu paraphrase, nhãn đúng, nhãn sai, và điểm tự tin (confidence score) của mô hình.
    
2. Bước 2: Bóc tách cấu trúc bằng Cây cú pháp (Dependency Parsing)
    
    Đây là trái tim của phần phân tích. Một câu yêu cầu phần mềm không chỉ là một chuỗi từ vựng, mà là một cấu trúc cây phụ thuộc (Dependency Tree). Bạn có thể sử dụng thư viện NLP mã nguồn mở như **spaCy** (Python) để chuyển đổi mỗi câu bị lỗi thành một cấu trúc đồ thị.
    
    Ví dụ câu: *"The system shall encrypt data within 2 seconds"*
    spaCy sẽ dựng một cây với `encrypt` (Động từ) làm gốc, chĩa nhánh ra `data` (Tân ngữ) và `within 2 seconds` (Trạng ngữ chỉ thời gian).
    

Tại đây, bạn áp dụng trực tiếp các thuật toán duyệt đồ thị cơ bản. Bằng cách thiết lập một thuật toán **Duyệt theo chiều sâu (DFS)** xuất phát từ node gốc (Động từ chính), bạn có thể quét dọc xuống các nhánh để kiểm tra:

- Có sự xuất hiện của các từ bổ nghĩa (Modifiers) bị ẩn sâu ở nhánh lá không? (Ví dụ: các từ như *securely, fast*).
- Cấu trúc câu có bị đảo ngược (Bị động) khiến DFS quét trúng tân ngữ trước khi quét trúng động từ không? Việc dùng DFS duyệt qua cây cú pháp giúp bạn lập trình tự động đếm xem có bao nhiêu câu sai bị vướng vào cấu trúc bị động, hoặc bao nhiêu câu sai do động từ hành động mạnh lấn át cụm từ chỉ hiệu năng.
1. Bước 3: Phân cụm lỗi bằng Mô hình hóa (Error Taxonomy)
    
    Sau khi thu thập được các đặc trưng ngữ pháp từ bước 2, bạn cần hệ thống hóa chúng. Để dễ hình dung và đưa vào bài báo, hãy thiết kế một cấu trúc phân cấp tương tự như khi vẽ **Class Diagram (Biểu đồ Lớp)** cho một hệ thống phần mềm.
    
    Bạn tạo ra một "Lớp cha" là `Misprediction_Reason`, từ đó kế thừa ra các "Lớp con" đại diện cho các nhóm lỗi ngôn ngữ:
    
    - **`Lexical_Overlap` (Chồng chéo từ vựng):** Xảy ra khi câu NFR chứa quá nhiều từ vựng thuộc về chức năng nghiệp vụ (ví dụ: *user, cart, login*).
    - **`Syntactic_Ambiguity` (Mơ hồ cú pháp):** Xảy ra khi câu dùng cấu trúc bị động, khiến AI không xác định được chủ thể chịu trách nhiệm cho yêu cầu.
    - **`Implicit_Constraint` (Ràng buộc ẩn):** Yêu cầu không có từ khóa NFR rõ ràng (như *fast, secure*) mà dùng ngữ cảnh (ví dụ: *"phải tuân thủ chuẩn ISO 27001"*).
2. Bước 4: Dùng LLM làm "Trợ lý Phân tích" (Automated Explainability)
    
    Để tiết kiệm sức lực, thay vì bạn tự đọc kết quả DFS ở Bước 2, bạn có thể đẩy tệp `mispredictions_log.json` vào luồng MCP để gọi API của GPT-4o hoặc Llama-3.
    Bạn thiết lập Prompt: *"Dưới đây là 50 câu yêu cầu bảo mật (Security NFR) mà AI của tôi dự đoán nhầm thành FR. Hãy phân tích cấu trúc ngôn ngữ (như loại động từ, cấu trúc chủ/bị động) để tìm ra mẫu số chung khiến AI bị nhầm lẫn."*
    
    **Tóm lại:** Việc hiện thực hóa RQ3.3 thực chất là bài toán dựng một mini-backend nhận đầu vào là các câu đoán sai, dùng công cụ NLP tạo cây cú pháp, dùng thuật toán duyệt đồ thị bóc tách cấu trúc, và cuối cùng phân loại chúng thành các nhóm lỗi logic. Báo cáo được quy trình này trong paper sẽ khiến hội đồng phản biện cực kỳ ấn tượng vì tính hệ thống và minh bạch của nó.
    

vậy dùng SHAP khi nào, để làm gì?

**SHAP (SHapley Additive exPlanations)** chính là "vũ khí tối thượng" để bạn giải quyết trọn vẹn **RQ3.3** (Tại sao mô hình dự đoán sai) và biến bài báo của mình thành một công trình chuẩn mực về **Explainable AI (Trí tuệ nhân tạo có thể giải thích)**.

Dưới đây là lời giải thích chi tiết về việc dùng SHAP khi nào, để làm gì, và cách nó khớp hoàn hảo vào khung nghiên cứu của bạn.

### 1. SHAP là gì và Để làm gì?

Các mô hình Deep Learning như ModernBERT hay ELECTRA được gọi là "hộp đen" (black-box). Bạn đưa cho nó một câu yêu cầu, nó trả về kết quả là NFR (95% tự tin). Nhưng nếu Reviewer hỏi: *"Tại sao nó lại tự tin như vậy? Nó dựa vào chữ nào trong câu?"* thì bản thân mô hình không trả lời được.

**SHAP sinh ra để "mở nắp hộp đen" đó.**

- Dựa trên lý thuyết trò chơi (Game Theory), SHAP tính toán **mức độ đóng góp (contribution score)** của TỪNG TỪ (token) trong một câu đối với kết quả cuối cùng.
- Nó sẽ gán cho mỗi từ một giá trị âm hoặc dương.
    - Ví dụ: Từ nào đẩy mô hình dự đoán là **NFR** sẽ được bôi **màu đỏ** (giá trị dương).
    - Từ nào kéo mô hình về hướng **FR** sẽ được bôi **màu xanh** (giá trị âm).

### 2. Dùng SHAP khi nào trong bài báo của bạn?

Bạn KHÔNG dùng SHAP trong quá trình huấn luyện (training). Bạn chỉ dùng SHAP ở **Giai đoạn Đánh giá (Evaluation & Error Analysis) - tức là Tuần 9-10 trong kế hoạch của bạn**, sau khi mô hình đã train xong và bắt đầu dự đoán trên tập test.

Cụ thể, bạn dùng SHAP trong 2 trường hợp "ăn tiền" sau:

**Trường hợp 1: Phân tích các ca đoán sai (Error Analysis - Giải quyết RQ3.3)**
Thay vì bạn phải dùng thuật toán duyệt cây cú pháp (DFS) phức tạp như đã bàn ở trên, bạn có thể chạy thẳng SHAP vào các câu AI đoán sai.

- *Ví dụ một câu Paraphrase bị lỗi:* "User information must be protected using cryptography."
- Thực tế đây là Security NFR, nhưng AI đoán nhầm là FR.
- *Bạn chạy SHAP và phát hiện ra:* SHAP bôi xanh lè (FR) cụm từ "User information" với lực kéo rất mạnh, trong khi cụm từ "protected" và "cryptography" lại có điểm SHAP rất mờ nhạt.
- **Kết luận rút ra đưa vào paper:** *"Mô hình AI bị 'thiên kiến' (bias) quá nặng với các danh từ chủ thể (User, System) mà phớt lờ các trạng từ/động từ chỉ tính chất (protected). Đây là nguyên nhân cốt lõi khiến Paraphrasing làm AI vấp ngã."*

**Trường hợp 2: Chứng minh sự vượt trội của ELECTRA so với BERT**
Bạn đã lập luận rằng ELECTRA đóng vai trò soi từ khóa (token-level precision) cực tốt. Lời nói gió bay, nhưng SHAP sẽ biến nó thành bằng chứng thép.

- Bạn đưa cùng một câu cho Baseline (chỉ dùng BERT) và mô hình Hybrid của bạn (ModernBERT + ELECTRA).
- Trích xuất biểu đồ SHAP của cả hai.
- Biểu đồ sẽ cho thấy: Cùng một từ khóa "encrypted", ở mô hình BERT nó chỉ có điểm SHAP là +0.2, nhưng ở mô hình Hybrid của bạn, nó sáng rực lên với điểm SHAP +0.8.
- **Lập luận ghi vào paper:** *"Phân tích từ SHAP đã chứng minh thực nghiệm rằng cơ chế Replaced Token Detection của ELECTRA thực sự giúp hệ thống khuếch đại sự chú ý vào các ràng buộc phi chức năng (NFR constraints) đúng như thiết kế."*

### 3. Cách triển khai thực tế (Rất dễ dàng)

Việc code SHAP cho các mô hình Transformer hiện nay cực kỳ nhàn vì đã có thư viện mã nguồn mở hỗ trợ tận răng.
Bạn chỉ cần cài đặt thư viện `shap` trong Python. Thư viện này có một module chuyên dụng cho text tên là `shap.Explainer`.

### **4. 4 CONTRIBUTIONS**

1. **Dataset:** Công bố benchmark mới **PROMISE-Robust** (Dữ liệu gốc + 3 biến thể paraphrase do LLM sinh ra).
2. **Metric:** Tiên phong áp dụng **Robustness Score (Độ nhất quán)** trong Requirements Engineering.
3. **Architecture:** Khẳng định sức mạnh của Hybrid Transformer (MLM + RTD) chống lại nhiễu loạn ngôn ngữ.
4. **LLM Benchmarking & Explainability:** Đánh giá giới hạn của Few-shot Prompting trên LLMs hiện đại trong bài toán Paraphrase, từ đó giải thích lý do tại sao các mô hình Fine-tuned chuyên biệt vẫn là giải pháp thực tiễn nhất cho ngành phần mềm.

### **5. EXPERIMENTAL DESIGN**

Đánh giá mô hình qua 2 Giai đoạn: **Benchmark (Dữ liệu gốc)** và **Robustness Test (Dữ liệu Paraphrase)**.

**Baselines system:**

1. *Ablation:* ModernBERT (đơn), ELECTRA (đơn).
2. *base 2:* BERT + RoBERTa (tái hiện lại bài báo “”)
3. *standard base:* BERT, RoBERTa. (tái hiện lại bài báo “”)
4. *floor base:* SVM + TF-IDF. (tái hiện bài báo “”)
5. **SOTA Few-Shot LLM : GPT-4o-mini / Llama-3-8B:** Tái hiện chính xác System Prompt và Few-shot examples từ bài báo *"On the Effectiveness of Zero-Shot and Few-Shot Pretrained Language Models for Software Requirement Classification" (Shafikuzzaman et al., 2025)*.

### **6. Research Plan**

| week | task | expected results |
| --- | --- | --- |
| 1-5 |  Literature Review, chuẩn hóa dataset PROMISE. Đọc kỹ thiết lập thực nghiệm của các bài báo. |  |
| 6 |  Dùng LLM sinh bộ dữ liệu PROMISE-Robust. 
• Code các baseline truyền thống (SVM, BERT).
• [MỚI] Xây dựng Pipeline (Java/Python) gọi API tự động để đánh giá Baseline 7 (xử lý rate-limit, JSON parsing, batch processing). |  |
| 7 | Thiết kế và Fine-tune kiến trúc custom Hybrid (ModernBERT + ELECTRA). Chạy thực nghiệm song song. |  |
| 8 | Trích xuất kết quả (Accuracy, F1, Robustness Score). Thực hiện Error Analysis bóc tách lỗi giữa Hybrid model và LLMs. |  |
| 9 | Viết bài luận. Đóng gói Source code, Prompt Templates, và Dataset đẩy lên GitHub để tăng tính minh bạch (Reproducibility), sẵn sàng Submit. |  |

````markdown
# 🤖 RBL: AI Classification of Software Requirements (FR vs NFR)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/rbl-requirements-classification/blob/main/notebooks/01_EDA_and_preprocessing.ipynb)

## 📌 Tổng Quan Đề Tài

**Câu hỏi nghiên cứu**: *"How accurately can AI classify software requirements into functional and non-functional categories?"*

Dự án này so sánh khả năng phân loại yêu cầu phần mềm (FR/NFR) của các phương pháp AI:
- Classical ML: Naive Bayes, SVM, Logistic Regression
- Transformer: BERT, RoBERTa
- LLMs: GPT-4o, Llama 3 (zero-shot & few-shot prompting)

## 👥 Thành Viên Nhóm

| Tên | MSSV | Vai trò |
|-----|------|---------|
| Nguyễn Văn A | 2xxxxxxx | Data preprocessing, Baseline models |
| Trần Thị B   | 2xxxxxxx | BERT fine-tuning |
| Lê Văn C     | 2xxxxxxx | LLM experiments, Report |

## 📁 Cấu Trúc Dự Án

```
notebooks/01_EDA_and_preprocessing.ipynb   ← Khám phá & làm sạch dữ liệu
notebooks/02_baseline_models.ipynb         ← SVM, Naive Bayes
notebooks/03_bert_finetuning.ipynb         ← BERT / RoBERTa
notebooks/04_llm_prompting.ipynb           ← GPT-4o, Llama 3
notebooks/05_results_analysis.ipynb        ← So sánh tổng hợp
```

## 🚀 Cách Chạy (Google Colab)

1. Click vào badge "Open in Colab" bên trên
2. Đăng nhập Google Account
3. Chọn: **Runtime → Change runtime type → GPU (T4)**
4. Chạy cell đầu tiên để setup môi trường

## 📊 Kết Quả (cập nhật liên tục)

| Model | Weighted F1 | Trạng thái |
|-------|------------|-----------|
| SVM (TF-IDF) | - | ⏳ Đang thực hiện |
| BERT-base | - | ⏳ Chờ |
| GPT-4o (0-shot) | - | ⏳ Chờ |

## 📦 Dataset

- **PROMISE NFR Dataset**: 625 software requirements, 12 classes
- Nguồn: `openscience.us/repo/requirements/nfr.html`

## 🔗 Tài Liệu Liên Quan

- [Cẩm Nang Nghiên Cứu](docs/research_questions.md)
- [Ghi Chú Papers](papers/paper_notes.md)
````

> ⚠️ **Thay `YOUR_USERNAME`** bằng tên GitHub thật của bạn ở badge Colab
 
# ⚖️ Legal RAG Assistant

A Retrieval-Augmented Generation (RAG) system that answers legal questions by retrieving relevant information from legal documents and generating context-aware responses with a Large Language Model (LLM). The current implementation is designed for **Persian legal question answering**.

## Features

- 📄 Retrieval-Augmented Generation (RAG)
- 🔍 Semantic search over legal documents
- 🤖 LLM-powered question answering
- 💬 Supports Persian legal question answering
- 📚 Vector-based document retrieval using LanceDB
- ⚡ Modular architecture built with LangChain

## Tech Stack

- Python
- LangChain
- LanceDB
- Hugging Face Transformers
- Hugging Face Embeddings
- OpenAI Compatible LLM

## Pipeline

```text
Legal Documents
        │
        ▼
  Text Chunking
        │
        ▼
 Embedding Model
        │
        ▼
 LanceDB Vector Store
        │
        ▼
   User Question
        │
        ▼
 Semantic Retrieval
        │
        ▼
 Retrieved Context
        │
        ▼
 Large Language Model
        │
        ▼
 Answer in Persian
```

## System Architecture

<p align="center">
  <img width="301" height="778" alt="image" src="https://github.com/user-attachments/assets/4b753639-91e4-4cfe-ae91-0db52f8b2cff" />
</p>

## Example

**Question**

 جریمه قانونی مقرر برای چک برگشتی طبق قانون جدید چیست؟

**Answer**

<div dir="rtl">

طبق مستندات ارائه‌شده، جریمه قانونی برای چک برگشتی به شرح زیر است:

1. **مجازات کیفری برای صادرکننده چک بلامحل (ماده 7):**
   - **الف)** اگر مبلغ چک کمتر از **80,000,000** ریال باشد، صادرکننده به حبس تا حداکثر شش ماه محکوم می‌شود.
   - **ب)** اگر مبلغ چک بین **780,000,000** ریال تا **3,900,000,000** ریال باشد، صادرکننده به حبس از شش ماه تا یک سال محکوم می‌شود.
   - **ج)** اگر مبلغ چک بیش از **3,900,000,000** ریال باشد، صادرکننده به حبس از یک سال تا دو سال و ممنوعیت از داشتن دسته‌چک به مدت دو سال محکوم می‌شود.

2. **مجازات‌های بانکی و محدودیت‌ها (ماده 5 مکرر):**
   - **الف)** عدم افتتاح حساب جدید و عدم صدور کارت بانکی جدید.
   - **ب)** مسدود شدن وجوه تمامی حساب‌ها و کارت‌های بانکی صادرکننده، به میزان کسری مبلغ چک.
   - **ج)** عدم پرداخت تسهیلات بانکی یا صدور ضمانت‌نامه‌های ارزی یا ریالی.

3. **محرومیت از حساب جاری (ماده 21):**
   - اگر شخص بیش از یک بار چک بلامحل صادر کند و تعقیب وی منجر به صدور کیفرخواست شود، بانک‌ها مکلف‌اند حساب‌های جاری او را مسدود کرده و تا سه سال به نام او حساب جاری جدیدی افتتاح نکنند.

4. **رفع سوءاثر از چک (ماده 5 مکرر، تبصره 3):**
   - واریز کسری مبلغ چک به حساب و درخواست مسدودی.
   - ارائه لاشه چک یا رضایت‌نامه رسمی دارنده چک.
   - ارائه حکم قضایی مبنی بر برائت ذمه.
   - سپری شدن سه سال از تاریخ صدور گواهینامه عدم پرداخت، مشروط بر اینکه دعوای حقوقی یا کیفری مطرح نشده باشد.

**تبصره مهم (ماده 7):** این مجازات‌ها شامل مواردی که چک بلامحل بابت معاملات نامشروع یا بهره‌برداری برای مقاصد رباخواری صادر شده باشد، نمی‌شود.

</div>

## Future Improvements

- Hybrid Search (Dense + BM25)
- Cross-Encoder Reranking
- Citation-aware responses
- Multi-turn conversation support
- Support for additional Persian legal resources
- Web interface using Streamlit or Gradio

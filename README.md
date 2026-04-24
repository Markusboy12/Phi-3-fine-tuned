# Phi-3 Medical Assistant: Fine-Tuning with Unsloth

This project demonstrates the fine-tuning of Microsoft's **Phi-3-mini** model on a specialized healthcare dataset using **Unsloth** for optimized performance. The goal is to create a lightweight, efficient Large Language Model (LLM) capable of providing preliminary medical guidance and answering health-related questions in a conversational format.

## 🚀 Key Features

- **Model**: Microsoft Phi-3-mini-4k-instruct (quantized to 4-bit).
- **Optimization**: Uses **Unsloth** for 2x faster training and reduced memory usage.
- **Technique**: Low-Rank Adaptation (**LoRA**) for parameter-efficient fine-tuning.
- **Domain**: Healthcare/Medical Q&A.
- **Hardware**: Optimized for single GPU environments (e.g., Google Colab T4).

## 🛠️ Tech Stack

- **Python**
- **Unsloth**: For fast LLM patching and training.
- **Hugging Face Transformers & TRL**: For model loading and Supervised Fine-Tuning (SFT).
- **PEFT**: For LoRA implementation.
- **Datasets**: For data processing.
- **PyTorch**: Deep learning framework.

## 📂 Project Structure

```text
.
├── Phi_3_fine_tuning_with_Unsloth.ipynb  # Main Jupyter Notebook with full pipeline
├── healthcare_data2.json                 # Dataset containing medical Q&A pairs
├── requirements.txt                      # Dependencies (if running locally)
└── README.md                             # This file
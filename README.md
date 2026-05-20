
***

# 🇷🇼 Tiny-Aya-Kinyarwanda: Continual Pretraining for Low-Resource Languages

[![Model: Tiny-Aya](https://img.shields.io/badge/Model-Tiny--Aya--3.35B-green)](https://huggingface.co/CohereForAI/tiny-aya-base)
[![Hardware: NVIDIA A100](https://img.shields.io/badge/Hardware-1x_NVIDIA_A100-orange)](https://lambdalabs.com/)
[![License: CC-BY-NC-4.0](https://img.shields.io/badge/License-CC--BY--NC--4.0-lightgrey)](https://creativecommons.org/licenses/by-nc/4.0/)

This project demonstrates the **Language-Specific Continual Pretraining** of the **Tiny Aya Base** (3.35B parameter) model on **Kinyarwanda**, a Bantu language spoken by ~12 million people. 

> **The Problem:** Most LLMs are "born" knowing 70-100 languages. But what happens when you need it to speak Language #101? 
> **The Solution:** Instead of starting from scratch (which costs millions), we "continually pretrain" an existing model on a new language for just **3.5 hours**, transforming it from gibberish to fluent Kinyarwanda.

---

## 📖 Table of Contents
1. [The Model & Architecture](#the-model--architecture)
2. [Data Pipeline (Cleaning & Engineering)](#data-pipeline)
3. [Training Challenges & Memory Solutions](#training-challenges)
4. [The 70/20/10 Data Recipe](#the-data-recipe)
5. [Evaluation & Results](#evaluation)
6. [How to Reproduce](#how-to-reproduce)

---

## 🧠 The Model & Architecture

We used **Tiny Aya Base** by Cohere Labs. It is a "dense decoder-only transformer"—essentially a highly efficient text-prediction engine.

| Feature | Specification | Tech Simplified |
| :--- | :--- | :--- |
| **Parameters** | 3.35 Billion | The number of "neurons" in the model's brain. |
| **Vocabulary** | 262,144 tokens | The size of the dictionary the model uses. |
| **Context Length** | 8192 tokens | How much text the model can "read" at one time. |
| **Attention** | GQA (Grouped Query Attention) | A faster way for the model to look back at previous words. |
| **Activation** | SwiGLU | The math function that helps neurons "fire." |

---

## 🏗 Data Pipeline

Finding high-quality Kinyarwanda text is difficult. Many standard AI datasets (like mc4 or CC-100) are currently broken or deprecated. 

### 1. Data Sources
We successfully gathered **~610,000 raw documents** from:
*   **FineWeb-2 (Kinyarwanda subset):** 200k docs (~400MB)
*   **Glot500:** 402k docs (~200MB)
*   **Wikipedia:** 8k docs (~13MB)

### 2. The "Filter" (Making the data clean)
If you train on junk, you get a junk model. We used a strict 4-step cleaning process:

*   **Language Identification (LID) Fix:** Standard tools like FastText misidentified Kinyarwanda as Tagalog or English. We bypassed this by trusting source-level metadata from Wikipedia and FineWeb.
*   **Heuristic Filtering:** 
    *   *Length:* Removed anything under 50 characters.
    *   *Alpha Ratio (≥ 0.6):* The text must be at least 60% letters. This removes math code, gibberish, and random symbols.
    *   *Kinyarwanda Stopword Check:* Every document **must** contain at least 3 common Kinyarwanda words (e.g., `ni`, `mu`, `na`, `ya`, `cy`).
*   **Exact Deduplication:** Used SHA-256 hashing to delete 100% identical copies of files.
*   **Near-Deduplication (MinHash LSH):** Using a "Jaccard threshold of 0.7." If two articles were 70% similar, we deleted one to avoid the model getting "stuck" on repetitive info.

---

## ⚖️ The Data Recipe (Anti-Forgetting)

A common mistake in AI is training **only** on the new language. This causes **Catastrophic Forgetting**—the model learns Kinyarwanda but "breaks" its knowledge of English. 

We used a **70/20/10 mix**:
1.  **70% Kinyarwanda:** The primary target for adaptation.
2.  **20% English Replay:** To maintain general reasoning and English fluency.
3.  **10% Bantu Replay:** We included Swahili (`sw`), Shona (`sn`), and Zulu (`zu`) because they are "cousin" languages to Kinyarwanda. This helps with **cross-lingual transfer**.

---

## ⚡ Training Challenges & Memory Solutions

Training a 3.35B model on a single 40GB GPU is like trying to fit a giant into a small car. We used four "engineering hacks" to make it work:

*   **Problem 1: Optimizer Memory.** Standard optimizers require ~28GB of RAM. 
    *   **Solution:** Used `adamw_bnb_8bit` (8-bit quantization), which cut memory usage in half.
*   **Problem 2: Sequence Length OOM.** At full length (8k tokens), the model crashed.
    *   **Solution:** Split sequences into 4 chunks of 2048 tokens each.
*   **Problem 3: DeepSpeed Failure.** JIT compilation failed due to missing headers.
    *   **Solution:** Abandoned DeepSpeed in favor of the 8-bit optimizer approach.
*   **Problem 4: Attention Mechanism.** 
    *   **Solution:** Installed `flash-attn` (v2.8.3) for memory-efficient attention.

---

## 📊 Evaluation & Results

### 1. Perplexity (The "Confusion Score")
Perplexity measures how "surprised" a model is by a language. **Lower is better.**

| Language | Baseline (Original) | Fine-tuned (Our Model) | Reduction |
| :--- | :--- | :--- | :--- |
| **Kinyarwanda** | **111.65** | **15.31** | **86.3%** |
| **English** | 9.96 | 6.54 | 34.3% |
| **Swahili** | 13.05 | 9.00 | 31.0% |

**Key takeaway:** We improved Kinyarwanda understanding by **7.3x** while actually *improving* English and Swahili performance.

### 2. Qualitative Leap (Generative Samples)
*   **Prompt:** *"Describe the country of Rwanda."*
*   **Baseline Output:** "Iyitaboza cy'uburungi. Umurima cy'abuburungi..." (Repetitive nonsense)
*   **Fine-tuned Output:** "U Rwanda ni igihugu cy'u Burayi... 13,650,000 (2017)." (Coherent text describing location and population).

---

## 📁 Project Structure

```text
tiny-aya-kinyarwanda/
├── data/
│   ├── download_data.py      # Fetches raw data from HF
│   ├── filter_and_dedup.py   # Heuristics + MinHash deduplication
│   ├── tokenize_and_pack.py  # Tokenization with Tiny Aya vocab
│   └── mix_data.py           # Mixing Kinyarwanda/English/Bantu
├── training/
│   ├── config.yaml           # Training hyperparameters
│   └── train.py              # Main training script (HuggingFace Trainer)
├── eval/
│   ├── eval_perplexity.py    # Metric calculation
│   └── eval_generation.py    # Side-by-side prompt testing
└── requirements.txt          # torch, transformers, bitsandbytes, flash-attn
```

---

## 🚀 How to Reproduce

1.  **Environment:** 
    ```bash
    pip install torch transformers datasets bitsandbytes flash-attn
    ```
2.  **Data Prep:**
    ```bash
    python data/download_data.py
    python data/filter_and_dedup.py --skip_lid # Skip LID as per report
    ```
3.  **Train:**
    ```bash
    python training/train.py --config ./training/config.yaml
    ```

---

## 🌟 Future Directions
*   **Instruction Tuning:** Convert this base model into a "Chat" model using SFT (Supervised Fine-Tuning).
*   **Longer Sequences:** Utilize DeepSpeed ZeRO-3 to train on the full 8192 context length.
*   **Vocabulary Analysis:** Investigate if adding 5,000 specific Kinyarwanda tokens would further improve character-per-token efficiency.

---
*Developed as a research project to demonstrate high-efficiency language adaptation of Large Language Models.*

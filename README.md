# 📸 Telegram Vision Bot

A lightweight Telegram bot that uses a local AI model (BLIP) to generate
captions and tags for uploaded images. Runs entirely on your machine —
no paid APIs needed.

## ✨ Features

- 🖼️ **Image Captioning** — AI-generated descriptions for photos
- 🏷️ **Auto-Tagging** — Smart hashtags from the caption
- 💾 **Caching** — Skips reprocessing duplicate images
- 📜 **User History** — Tracks last 3 interactions per user
- ⚡ **Local AI** — Uses open-source BLIP model (no API costs)

## 🛠️ Tech Stack

| Component      | Technology                                  |
|---------------|---------------------------------------------|
| Bot Framework | python-telegram-bot 21.0                    |
| Vision Model  | BLIP (Salesforce/blip-image-captioning-base)|
| ML Framework  | PyTorch + Hugging Face Transformers         |
| Language      | Python 3.11                                 |
| Platform      | Windows 10/11                               |

## 📐 Architecture

```text
User (Telegram App)
       │
       ▼
┌──────────────┐
│ Telegram API │
└──────┬───────┘
       │
       ▼
┌──────────────┐     ┌─────────────┐
│   bot.py     │────▶│ BLIP Model  │
│  (Handlers)  │◀────│ (Local AI)  │
└──────┬───────┘     └─────────────┘
       │
       ▼
┌──────────────┐
│ Cache +      │
│ History      │
└──────────────┘
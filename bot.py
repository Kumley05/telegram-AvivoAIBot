"""
Telegram Avivo Bot (Windows Compatible)
Receives images and generates AI captions + tags using local BLIP model.
"""

import os
import logging
from io import BytesIO
from dotenv import load_dotenv

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# ──────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────

load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────
# LOAD AI MODEL (loads once at startup)
# ──────────────────────────────────────

MODEL_NAME = "Salesforce/blip-image-captioning-base"

logger.info("=" * 50)
logger.info("Loading BLIP Avivo Model...")
logger.info("First run will download ~900MB model.")
logger.info("Please wait...")
logger.info("=" * 50)

processor = BlipProcessor.from_pretrained(MODEL_NAME)
model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)

logger.info("=" * 50)
logger.info("Model loaded successfully!")
logger.info("=" * 50)

# ──────────────────────────────────────
# CACHE AND HISTORY
# ──────────────────────────────────────

image_cache = {}       # Stores processed image results
user_history = {}      # Stores last 3 interactions per user


def save_to_history(user_id: int, entry: str) -> None:
    """Save interaction to user history. Keep only last 3."""
    if user_id not in user_history:
        user_history[user_id] = []
    user_history[user_id].append(entry)
    # Keep only last 3 entries
    if len(user_history[user_id]) > 3:
        user_history[user_id] = user_history[user_id][-3:]


def get_caption(image: Image.Image) -> str:
    """Generate a caption for the given image using BLIP."""
    inputs = processor(image, return_tensors="pt")
    output_ids = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(output_ids[0], skip_special_tokens=True)
    return caption


def get_tags(caption: str) -> str:
    """Extract meaningful hashtags from caption."""
    stop_words = {
        "a", "an", "the", "is", "in", "on", "at", "of",
        "and", "with", "to", "for", "it", "by", "that",
        "this", "are", "was", "were", "been", "being",
        "have", "has", "had", "do", "does", "did",
        "there", "their", "they", "them", "then",
        "from", "into", "some", "very", "just",
    }
    words = caption.lower().split()
    tags = []
    seen = set()

    for word in words:
        # Remove punctuation
        clean = word.strip(".,!?;:'\"")
        if (
            clean not in stop_words
            and len(clean) > 3
            and clean not in seen
        ):
            tags.append(f"#{clean}")
            seen.add(clean)
        if len(tags) >= 3:
            break

    return " ".join(tags) if tags else "#image"


# ──────────────────────────────────────
# COMMAND HANDLERS
# ──────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
    text = (
        "🤖 *Welcome to the Avivo Bot!*\n\n"
        "I use AI to describe your images.\n\n"
        "*How to use:*\n"
        "📸 Just send me any photo!\n\n"
        "*Commands:*\n"
        "/help — Show instructions\n"
        "/image — How to upload a photo\n"
        "/history — Your last 3 interactions\n"
    )
    await update.message.reply_text(text, parse_mode="Markdown")


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command."""
    text = (
        "📖 *Bot Instructions*\n\n"
        "1️⃣ Send me any photo or image\n"
        "2️⃣ I'll analyze it using AI\n"
        "3️⃣ You'll get a caption + hashtags\n\n"
        "*Commands:*\n"
        "/start — Welcome message\n"
        "/help — This help message\n"
        "/image — Upload instructions\n"
        "/history — Recent interactions\n"
    )
    await update.message.reply_text(text, parse_mode="Markdown")


async def cmd_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /image command."""
    text = (
        "📸 *How to send an image:*\n\n"
        "1️⃣ Click the 📎 paperclip icon below\n"
        "2️⃣ Choose a photo from your gallery\n"
        "3️⃣ Send it to this chat\n"
        "4️⃣ Wait 5-10 seconds for AI analysis\n\n"
        "💡 *Tip:* Clear, bright photos work best!"
    )
    await update.message.reply_text(text, parse_mode="Markdown")


async def cmd_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /history command."""
    user_id = update.message.from_user.id
    history = user_history.get(user_id, [])

    if not history:
        await update.message.reply_text(
            "📜 No history yet!\nSend me a photo to get started."
        )
        return

    text = "📜 *Your Recent Interactions:*\n\n"
    for i, entry in enumerate(history, 1):
        text += f"*{i}.* {entry}\n\n"

    await update.message.reply_text(text, parse_mode="Markdown")


# ──────────────────────────────────────
# IMAGE HANDLER
# ──────────────────────────────────────

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process uploaded photos."""
    user_id = update.message.from_user.id

    # Send processing message
    wait_msg = await update.message.reply_text(
        "🔍 Analyzing your image...\n⏳ This may take 5-10 seconds."
    )

    try:
        # Get highest resolution photo
        photo = update.message.photo[-1]
        file_id = photo.file_id

        # Check if we already processed this exact image
        if file_id in image_cache:
            caption = image_cache[file_id]["caption"]
            tags = image_cache[file_id]["tags"]
            logger.info(f"Cache hit for image: {file_id[:20]}...")
        else:
            # Download image into memory
            photo_file = await photo.get_file()
            buffer = BytesIO()
            await photo_file.download_to_memory(buffer)
            buffer.seek(0)

            # Open with PIL
            raw_image = Image.open(buffer).convert("RGB")

            # Generate caption and tags
            caption = get_caption(raw_image)
            tags = get_tags(caption)

            # Save to cache
            image_cache[file_id] = {
                "caption": caption,
                "tags": tags,
            }
            logger.info(f"Processed new image: {file_id[:20]}...")

        # Build response
        response = (
            f"✅ *Analysis Complete!*\n\n"
            f"📝 *Caption:*\n{caption.capitalize()}\n\n"
            f"🏷️ *Tags:*\n{tags}\n\n"
            f"─────────────────────\n"
            f"🤖 _Powered by BLIP (Salesforce)_"
        )

        await wait_msg.edit_text(response, parse_mode="Markdown")

        # Save to user history
        short = caption[:40] + "..." if len(caption) > 40 else caption
        save_to_history(user_id, f"🖼️ {short}")

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        await wait_msg.edit_text(
            "❌ Sorry, something went wrong.\n"
            "Please try again with a different image."
        )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle non-command text messages."""
    await update.message.reply_text(
        "🤔 I can only analyze images!\n\n"
        "📸 Send me a photo, or type /help"
    )


# ──────────────────────────────────────
# MAIN
# ──────────────────────────────────────

def main() -> None:
    """Start the bot."""

    # Check token
    if not TOKEN:
        print("")
        print("=" * 50)
        print("ERROR: No bot token found!")
        print("")
        print("Fix: Open the .env file and add:")
        print("TELEGRAM_BOT_TOKEN=your_token_here")
        print("=" * 50)
        print("")
        return

    print("")
    print("=" * 50)
    print("   Telegram Avivo Bot")
    print("   Starting up...")
    print("=" * 50)
    print("")

    # Build application
    app = Application.builder().token(TOKEN).build()

    # Register commands
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("image", cmd_image))
    app.add_handler(CommandHandler("history", cmd_history))

    # Register message handlers
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    # Start
    print("Bot is running!")
    print("Open Telegram and message your bot.")
    print("Press Ctrl+C to stop.")
    print("")

    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()

import os
import httpx
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
API_URL = os.getenv("API_URL", "http://api:8000")

async def start(update: Update, context):
    await update.message.reply_text("Send me a website URL to index first!")

async def handle_message(update: Update, context):
    user_id = update.message.from_user.id
    text = update.message.text
    
    try:
        if text.startswith("/process_url"):
            url = text[len("/process_url"):].strip()
            async with httpx.AsyncClient(timeout = 240) as client:
                await update.message.reply_text("Indexing website...")
                # Send as JSON body with "url" key
                response = await client.post(
                    f"{API_URL}/process_url",
                    json={"url": url}  # Wrap in JSON object
                )
            await update.message.reply_text("Indexing finished! Now ask questions")
            context.user_data["current_url"] = url
        else:
            url = context.user_data.get("current_url")
            if not url:
                await update.message.reply_text("Please send a URL first!")
                return
                
            async with httpx.AsyncClient(timeout = 300) as client:
                await update.message.reply_text("Processing query...")
                # Send all parameters in JSON body
                response = await client.post(
                    f"{API_URL}/ask",
                    json={
                        "url": url,
                        "query": text,
                        "user_id": str(user_id)
                    }
                )
                await update.message.reply_text(response.json()["response"])
    except Exception as e:
        await update.message.reply_text(f"Error: {str(e)}")

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT, handle_message))
    app.run_polling()

if __name__ == "__main__":
    main()
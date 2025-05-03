import json
import asyncio
import os
import logging
from utils.config import MODELS,MAX_TOKEN,TEMPERATURE,TOP_P
from typing import Optional
from groq import Groq

logger = logging.getLogger(__name__)

groq_ai = Groq(api_key=os.getenv("GROQ_KEY"))

def build_message(system_message, user_info=None):
    sys_parts = [system_message.strip()]
    system_messages = [{"role": "system", "content": "\n\n".join(sys_parts)}]
    if user_info and user_info.strip():
        system_messages.append({"role": "system", "content": user_info.strip()})

    return system_messages

async def get_chat_response(
    system_message: str,
    user_message: str,
    user_info: Optional[str] = None,
    user_id: Optional[str] = None
    ):
    model = MODELS

    conversation = build_message(system_message, user_info)
    conversation.append({"role": "user", "content": user_message.strip()})

    try:
        logger.info(f"\n=== [MODEL: {model}] ===")
        logger.info("Payload messages yang dikirim ke LLM:")
        logger.info(json.dumps(conversation, indent=2, ensure_ascii=False))

        response = groq_ai.chat.completions.create(
            messages=conversation,
            model=model,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKEN,
            frequency_penalty=0,
            top_p=TOP_P
        )

        content = response.choices[0].message.content
        return content

    except Exception as e:
        logging.error(f"[ERROR - {model}]: {e}", exc_info=True)
        await asyncio.sleep(1)
        return "Terjadi kesalahan saat memproses permintaan ke LLM."

import logging

from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel
from app.groq_handler import get_chat_response
from utils.logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI()
logger.info('-------------API Groq [Start]--------------')

class GroqRequest(BaseModel):
    system_message: str
    user_message: str
    user_info: Optional[str] = None
    user_id: Optional[str] = None
      
@app.post("/chat")
async def endpoint_chat(request: GroqRequest):
    logger.info(f"-------------[GETTING_RESPONSE]-------------")
    try:
        response = await get_chat_response(
            request.system_message,
            request.user_message,
            request.user_info,
            request.user_id
        )
        logger.info(f"Response {request.user_id}: {response}")
        logger.info(f"-------------[END_RESPONSE]-------------")
        return {"response": response}
    except Exception as e:
        logger.error(f"Error dalam memproses chat request untuk user_id {request.user_id}: {str(e)}", exc_info=True)
        return {"error": "Terjadi kesalahan dalam memproses permintaan"}

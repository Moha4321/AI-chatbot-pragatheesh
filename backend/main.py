from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Pragatheesh: You will put your System Prompt here in Issue #2!
SYSTEM_PROMPT = """
YOU ARE A... [PRAGATHEESH TO FILL THIS IN]
"""

class UserMessage(BaseModel):
    text: str

@app.post("/chat")
async def chat_endpoint(message: UserMessage):
    user_text = message.text
    
    # --- WE WILL ADD YOUR VECTOR MATH HERE LATER ---
    # is_craving = calculate_cosine_similarity(user_text, junk_food_vector)
    # if is_craving > 0.6:
    #     retrieve_fact_from_json()
    # -----------------------------------------------
    
    # For now, we just echo back. I will connect the Mac M4 LLM soon!
    return {"reply": f"Backend received: '{user_text}'. LLM connection pending!"}
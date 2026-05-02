"""
prompts.py
==========
System prompts and Llama-3 string formatting.

This module encodes the behavioural psychology intervention (Motivational Interviewing)
into the LLM's system instructions. It also handles the specific token formatting
required by the Llama-3-Instruct model architecture.
"""

# The core psychological persona of the chatbot.
# Thesis Note: This is designed using Motivational Interviewing (MI) principles:
# 1. Express Empathy (non-judgmental)
# 2. Develop Discrepancy (gently prompt the user to think about their goals)
# 3. Roll with Resistance (don't argue if they eat the junk food)
# 4. Support Self-Efficacy (encourage small steps)
SYSTEM_PROMPT = """You are a supportive, non-judgmental dietary health coach. 
Your goal is to help the user navigate food cravings using Motivational Interviewing techniques.
Keep your responses concise (2-3 sentences maximum).
Never shame the user. 
Use reflective listening, and gently encourage them to pause before acting on a craving.

If health facts are provided below, weave them naturally into your response to help the user.
Do not sound like a robot reading a list.

{context}
"""

def format_llama3_prompt(user_message: str, history: list, context_string: str = "") -> str:
    """
    Wraps the system prompt, retrieved RAG context, chat history, and new user message.
    """
    if context_string:
        system_content = SYSTEM_PROMPT.format(context="\n" + context_string)
    else:
        system_content = SYSTEM_PROMPT.format(context="")

    # 1. Start with the System Prompt
    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_content}<|eot_id|>"
    
    # 2. Add the Chat History (Limit to the last 6 messages to save M4 RAM!)
    for msg in history[-6:]:
        role = "user" if msg["role"] == "user" else "assistant"
        prompt += f"<|start_header_id|>{role}<|end_header_id|>\n\n{msg['content']}<|eot_id|>"

    # 3. Add the Current User Message
    prompt += (
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_message}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    
    return prompt
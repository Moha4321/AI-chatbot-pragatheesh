"""
study_logger.py
===============
Telemetry and data collection for the N=32 pilot study.

Logs every chat interaction to a CSV file for post-study quantitative analysis.
"""

import csv
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# The log file will be saved in the research_data folder at the project root
LOG_DIR = Path(__file__).parent.parent / "research_data"
LOG_FILE = LOG_DIR / "chat_telemetry.csv"

# Define the columns for our dataset
HEADERS =[
    "timestamp",
    "session_id",
    "user_message",
    "rag_triggered",
    "intent_score",
    "bot_response"
]

def init_logger():
    """Create the CSV file and write headers if it doesn't exist."""
    LOG_DIR.mkdir(exist_ok=True)
    if not LOG_FILE.exists():
        with open(LOG_FILE, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(HEADERS)
        logger.info(f"Initialized new telemetry file at {LOG_FILE}")

def log_interaction(
    session_id: str, 
    user_message: str, 
    rag_triggered: bool, 
    intent_score: float, 
    bot_response: str
):
    """
    Append a single chat interaction to the CSV dataset.
    """
    try:
        with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.utcnow().isoformat(),
                session_id,
                user_message,
                rag_triggered,
                round(intent_score, 4),
                bot_response
            ])
    except Exception as e:
        logger.error(f"Failed to write to telemetry CSV: {e}")

# Initialize the file when the module is imported
init_logger()

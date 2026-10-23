import os
from pathlib import Path
import tweepy
import nltk
import sklearn
from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parent / ".env")

CONSUMER_KEY = os.getenv("X_CONSUMER_KEY")
SECRET_KEY = os.getenv("X_SECRET_KEY")
BEARER_TOKEN = os.getenv("X_BEARER_TOKEN")


client = tweepy.Client(bearer_token=BEARER_TOKEN, wait_on_rate_limit=True)

print("TA1.py is running successfully.")

"""Tweepy wrapper for posting tweets and threads to X."""

import os

import tweepy
from dotenv import load_dotenv


def get_client() -> tweepy.Client:
    """Create an authenticated tweepy Client from .env credentials."""
    load_dotenv()

    return tweepy.Client(
        consumer_key=os.environ["X_API_KEY"],
        consumer_secret=os.environ["X_API_SECRET"],
        access_token=os.environ["X_ACCESS_TOKEN"],
        access_token_secret=os.environ["X_ACCESS_TOKEN_SECRET"],
    )


def post_thread(client: tweepy.Client, tweets: list[str]) -> None:
    """Post a list of tweets as a thread (each replying to the previous)."""
    previous_id = None
    for i, tweet_text in enumerate(tweets):
        response = client.create_tweet(
            text=tweet_text,
            in_reply_to_tweet_id=previous_id,
        )
        previous_id = response.data["id"]
        print(f"  Posted tweet {i + 1}/{len(tweets)}: {tweet_text[:60]}...")

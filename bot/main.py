"""CLI entry point for Corporate Bathroom Chronicles bot."""

import argparse
import json
import sys
import time
from pathlib import Path

from bot.stories import STORIES, split_into_tweets

STATE_FILE = Path(__file__).resolve().parent.parent / "state.json"


def load_state() -> int:
    """Return the index of the next story to post."""
    if STATE_FILE.exists():
        data = json.loads(STATE_FILE.read_text())
        return data.get("next_index", 0)
    return 0


def save_state(next_index: int) -> None:
    """Save the index of the next story to post."""
    STATE_FILE.write_text(json.dumps({"next_index": next_index}))


def dry_run(index: int) -> None:
    """Print the next story without posting."""
    story = STORIES[index]
    tweets = split_into_tweets(story)
    print(f"=== Story {index + 1}/{len(STORIES)} ===\n")
    for i, tweet in enumerate(tweets):
        print(f"--- Tweet {i + 1}/{len(tweets)} ({len(tweet)} chars) ---")
        print(tweet)
        print()


def post_once() -> None:
    """Post the next unposted story to X."""
    index = load_state()
    if index >= len(STORIES):
        print("All 25 stories have been posted!")
        return

    # Import here so --dry-run works without credentials
    from bot.twitter_client import get_client, post_thread

    story = STORIES[index]
    tweets = split_into_tweets(story)

    print(f"Posting story {index + 1}/{len(STORIES)} ({len(tweets)} tweets)...")
    client = get_client()
    post_thread(client, tweets)

    save_state(index + 1)
    print(f"Done! {len(STORIES) - index - 1} stories remaining.")


def post_all(delay_minutes: int = 60) -> None:
    """Post all remaining stories with a delay between each."""
    index = load_state()
    if index >= len(STORIES):
        print("All 25 stories have been posted!")
        return

    from bot.twitter_client import get_client, post_thread

    client = get_client()
    remaining = len(STORIES) - index

    print(f"{remaining} stories remaining. Posting with {delay_minutes}min delay...")

    while index < len(STORIES):
        story = STORIES[index]
        tweets = split_into_tweets(story)

        print(f"\nPosting story {index + 1}/{len(STORIES)} ({len(tweets)} tweets)...")
        post_thread(client, tweets)
        save_state(index + 1)

        index += 1
        if index < len(STORIES):
            print(f"Waiting {delay_minutes} minutes before next post...")
            time.sleep(delay_minutes * 60)

    print("\nAll 25 stories have been posted!")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Corporate Bathroom Chronicles - X Bot"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the next story without posting",
    )
    parser.add_argument(
        "--post-once",
        action="store_true",
        help="Post the next story to X",
    )
    parser.add_argument(
        "--post-all",
        action="store_true",
        help="Post all remaining stories with delay",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=60,
        help="Minutes between posts when using --post-all (default: 60)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset state to start from the first story",
    )

    args = parser.parse_args()

    if args.reset:
        save_state(0)
        print("State reset. Will start from story 1.")
        return

    if args.dry_run:
        index = load_state()
        if index >= len(STORIES):
            print("All 25 stories have been posted! Use --reset to start over.")
            return
        dry_run(index)
    elif args.post_once:
        post_once()
    elif args.post_all:
        post_all(args.delay)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

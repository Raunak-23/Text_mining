"""
Reddit data collector using PRAW.

Fetches high-interaction threads for a given product query, walks the full
comment tree, and returns a structured JSON-serialisable dict.

Output schema (one file per subreddit):
{
  "product": "iPhone 15 Pro",
  "subreddit": "apple",
  "fetched_at": "2024-...",
  "posts": [
    {
      "id": "...",
      "title": "...",
      "selftext": "...",
      "score": 412,
      "upvote_ratio": 0.97,
      "num_comments": 134,
      "url": "...",
      "created_utc": 1234567890,
      "comments": [
        {
          "id": "...",
          "body": "...",
          "score": 28,
          "depth": 0,
          "replies": [...]          # full recursive tree
        }
      ]
    }
  ]
}
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import praw
import praw.models

from absa.utils.config import settings
from absa.utils.display import console, print_info, print_success, print_warning


# ---------------------------------------------------------------------------
# PRAW client factory
# ---------------------------------------------------------------------------

def _make_reddit() -> praw.Reddit:
    return praw.Reddit(
        client_id=settings.reddit_client_id,
        client_secret=settings.reddit_client_secret,
        user_agent=settings.reddit_user_agent,
        ratelimit_seconds=5,
    )


# ---------------------------------------------------------------------------
# Subreddit discovery
# ---------------------------------------------------------------------------

_CATEGORY_SUBREDDITS: dict[str, list[str]] = {
    "smartphone": ["smartphones", "android", "iphone", "apple", "samsung", "OnePlus", "pixel"],
    "laptop":     ["laptops", "SuggestALaptop", "thinkpad", "Dell", "surface"],
    "headphone":  ["headphones", "audiophile", "HeadphoneAdvice", "SonyHeadphones"],
    "tv":         ["4kTV", "hometheater", "Televisions"],
    "camera":     ["photography", "Cameras", "mirrorless"],
}

def _guess_subreddits(product: str) -> list[str]:
    """Return product-category subreddits + generic review subs."""
    q = product.lower()
    chosen: list[str] = []
    for category, subs in _CATEGORY_SUBREDDITS.items():
        if category in q or any(kw in q for kw in category.split()):
            chosen.extend(subs)
    if not chosen:
        chosen = settings.default_subreddits
    # Always append generic high-signal subs
    for generic in ("reviews", "gadgets", "technology"):
        if generic not in chosen:
            chosen.append(generic)
    return chosen


# ---------------------------------------------------------------------------
# Comment tree walker
# ---------------------------------------------------------------------------

def _walk_comment(
    comment: praw.models.Comment,
    depth: int = 0,
    limit: int = 200,
    collected: list[int] | None = None,
) -> dict[str, Any] | None:
    """Recursively serialize a comment and its replies."""
    if collected is None:
        collected = [0]
    if collected[0] >= limit:
        return None
    if not isinstance(comment, praw.models.Comment):
        return None  # skip MoreComments objects

    body = comment.body.strip()
    # Skip deleted / removed / bot-typical empty comments
    if body in {"[deleted]", "[removed]", ""} or len(body) < 10:
        return None

    score = getattr(comment, "score", 0) or 0
    node: dict[str, Any] = {
        "id":    comment.id,
        "body":  body,
        "score": score,
        "depth": depth,
        "replies": [],
    }
    collected[0] += 1

    # Walk replies (PRAW already replaced MoreComments with limit=0 expand)
    if hasattr(comment, "replies"):
        for reply in comment.replies:
            if collected[0] >= limit:
                break
            child = _walk_comment(reply, depth + 1, limit, collected)
            if child is not None:
                node["replies"].append(child)

    return node


# ---------------------------------------------------------------------------
# Post fetcher
# ---------------------------------------------------------------------------

def _fetch_posts_from_subreddit(
    reddit: praw.Reddit,
    subreddit_name: str,
    product: str,
    post_limit: int,
    comment_limit: int,
    time_filter: str,
    min_score: int,
    min_comments: int,
) -> list[dict[str, Any]]:
    try:
        sub = reddit.subreddit(subreddit_name)
        # Search within the subreddit for the product name
        results = sub.search(
            query=product,
            sort="relevance",
            time_filter=time_filter,
            limit=post_limit * 2,  # over-fetch so we can filter
        )
    except Exception as exc:
        print_warning(f"r/{subreddit_name}: search failed — {exc}")
        return []

    posts: list[dict[str, Any]] = []
    for submission in results:
        if len(posts) >= post_limit:
            break
        if submission.score < min_score:
            continue
        if submission.num_comments < min_comments:
            continue

        # Expand comment forest (replace MoreComments shallowly)
        try:
            submission.comments.replace_more(limit=0)
        except Exception:
            pass

        comments: list[dict[str, Any]] = []
        collected = [0]
        for top_comment in submission.comments:
            if collected[0] >= comment_limit:
                break
            node = _walk_comment(top_comment, depth=0, limit=comment_limit, collected=collected)
            if node:
                comments.append(node)

        posts.append(
            {
                "id":           submission.id,
                "title":        submission.title,
                "selftext":     submission.selftext,
                "score":        submission.score,
                "upvote_ratio": submission.upvote_ratio,
                "num_comments": submission.num_comments,
                "url":          submission.url,
                "created_utc":  int(submission.created_utc),
                "comments":     comments,
            }
        )
        time.sleep(0.3)  # polite pacing to stay within rate limits

    return posts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch(
    product: str,
    subreddits: list[str] | None = None,
    post_limit: int | None = None,
    comment_limit: int | None = None,
    time_filter: str | None = None,
    out_dir: Path | None = None,
    force: bool = False,
) -> dict[str, Path]:
    """
    Fetch Reddit data for *product* across *subreddits*.

    Returns a mapping of  subreddit_name -> path_to_json_file.
    Files are written under out_dir/<slug>/<subreddit>.json.
    Skips already-cached files unless force=True.
    """
    reddit = _make_reddit()

    subreddits = subreddits or _guess_subreddits(product)
    post_limit    = post_limit    or settings.fetch_post_limit
    comment_limit = comment_limit or settings.fetch_comment_limit
    time_filter   = time_filter   or settings.fetch_time_filter
    min_score     = settings.fetch_min_score
    min_comments  = settings.fetch_min_comments

    slug = _slugify(product)
    out_dir = out_dir or (settings.raw_dir / slug)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Path] = {}

    with console.status(f"[cyan]Fetching Reddit data for '{product}'…") as status:
        for sub_name in subreddits:
            out_file = out_dir / f"{sub_name}.json"
            if out_file.exists() and not force:
                print_info(f"r/{sub_name}: cache hit → {out_file.name}")
                results[sub_name] = out_file
                continue

            status.update(f"[cyan]Searching r/{sub_name} for '{product}'…")
            posts = _fetch_posts_from_subreddit(
                reddit,
                sub_name,
                product,
                post_limit,
                comment_limit,
                time_filter,
                min_score,
                min_comments,
            )

            if not posts:
                print_warning(f"r/{sub_name}: no usable posts found, skipping.")
                continue

            payload = {
                "product":    product,
                "subreddit":  sub_name,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "posts":      posts,
            }
            out_file.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print_success(
                f"r/{sub_name}: {len(posts)} posts -> {out_file.relative_to(settings.raw_dir.parent)}"
            )
            results[sub_name] = out_file

    total_posts = 0
    for p in results.values():
        try:
            total_posts += len(json.loads(p.read_text(encoding="utf-8"))["posts"])
        except Exception:
            pass
    print_success(f"Fetch complete: {total_posts} posts across {len(results)} subreddits.")
    return results


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"[\s_-]+", "-", text)

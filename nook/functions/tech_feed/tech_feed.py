# nook/functions/tech_feed/tech_feed.py
import inspect
import os
import time
import traceback
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

import feedparser
import requests
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib
from bs4 import BeautifulSoup
from ..common.python.gemini_client import create_client

_MARKDOWN_FORMAT = """
# {title}

[View on {feed_name}]({url})

{summary}
"""

class Config:
    tech_feed_max_entries_per_day = 10
    summary_index_s3_key_format = "tech_feed/{date}.md"
    threshold_days = 1

    @classmethod
    def load_feeds(cls) -> dict[str, str]:
        feed_toml_path = os.path.join(os.path.dirname(__file__), "feed.toml")
        with open(feed_toml_path, "rb") as f:
            feed_data = tomllib.load(f)
        return {feed["name"]: feed["url"] for feed in feed_data.get("feeds", [])}

@dataclass
class Article:
    feed_name: str
    title: str
    url: str
    text: str
    soup: BeautifulSoup
    category: str | None = field(default=None)
    summary: list[str] = field(init=False)

class TechFeed:
    def __init__(self) -> None:
        self._client = create_client()
        self._tech_feed_urls = Config.load_feeds()
        self._threshold = datetime.now() - timedelta(days=Config.threshold_days)

    def __call__(self) -> None:
        markdowns = []
        for feed_name, feed_url in self._tech_feed_urls.items():
            print(f"Processing feed: {feed_name} ({feed_url})")
            feed_parser: feedparser.FeedParserDict = feedparser.parse(feed_url)
            print(f"Feed entries count: {len(feed_parser['entries'])}")
            entries = self._filter_entries(feed_parser)
            print(f"Filtered entries count: {len(entries)}")
            if len(entries) > Config.tech_feed_max_entries_per_day:
                entries = entries[:Config.tech_feed_max_entries_per_day]
            for entry in entries:
                try:
                    article = self._retrieve_article(entry, feed_name=feed_name)
                    print(f"Retrieved article: {article.title}")
                    article.summary = self._summarize_article(article)
                    print(f"Summarized article: {article.title}")
                    markdowns.append(self._stylize_article(article))
                except Exception as e:
                    print(f"Error processing article {entry.get('link', 'unknown')}: {e}")
                    traceback.print_exc()
                    continue
            time.sleep(2)  # APIリクエストの制限を避ける
        print(f"Total markdowns generated: {len(markdowns)}")
        self._store_summaries(markdowns)

    def _filter_entries(self, feed_parser: feedparser.FeedParserDict) -> list[dict[str, Any]]:
        filtered_entries = []
        for entry in feed_parser["entries"]:
            date_ = entry.get("date_parsed") or entry.get("published_parsed")
            if not date_:
                print(f"date_ is None for entry: {entry.get('link', 'unknown')}")
                continue
            try:
                published_dt = datetime.fromtimestamp(time.mktime(date_))
                if published_dt > self._threshold:
                    filtered_entries.append(entry)
                else:
                    print(f"Entry too old: {entry.get('title', 'unknown')} ({published_dt})")
            except Exception as e:
                print(f"Error converting date for {entry.get('link', 'unknown')}: {e}")
                traceback.print_exc()
                continue
        return filtered_entries

    def _retrieve_article(self, entry: dict[str, Any], feed_name: str) -> Article:
        try:
            response = requests.get(entry.link)
            soup = BeautifulSoup(response.text, "html.parser")
            text = "\n".join(
                [p.get_text() for p in soup.find_all(["p", "code", "ul", "h1", "h2", "h3", "h4", "h5", "h6"])]
            )
            return Article(
                feed_name=feed_name,
                title=entry.title,
                url=entry.link,
                text=text,
                soup=soup,
            )
        except Exception as e:
            raise Exception(f"Error raised while retrieving article: {e}") from e

    def _store_summaries(self, summaries: list[str]) -> None:
        date_str = date.today().strftime("%Y-%m-%d")
        key = Config.summary_index_s3_key_format.format(date=date_str)
        output_dir = os.environ.get("OUTPUT_DIR", "./output")
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, key)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n---\n".join(summaries))
        print(f"Saved summaries to {file_path}")

    def _stylize_article(self, article: Article) -> str:
        return _MARKDOWN_FORMAT.format(
            title=article.title,
            feed_name=article.feed_name,
            url=article.url,
            summary=article.summary,
        )

    def _summarize_article(self, article: Article) -> str:
        return self._client.generate_content(
            contents=self._contents_format.format(title=article.title, text=article.text),
            system_instruction=self._system_instruction,
        )

    @property
    def _system_instruction(self) -> str:
        return inspect.cleandoc(
            """
            ユーザーから記事のタイトルと文章が与えられるので、内容をよく読み、日本語で詳細な要約を作成してください。
            与えられる文章はHTMLから抽出された文章なので、一部情報が欠落していたり、数式、コード、不必要な文章などが含まれている場合があります。
            要約以外の出力は不要です。
            """
        )

    @property
    def _contents_format(self) -> str:
        return inspect.cleandoc(
            """
            {title}

            本文:
            {text}
            """
        )

if __name__ == "__main__":
    # テスト用コード
    tech_feed = TechFeed()
    tech_feed()
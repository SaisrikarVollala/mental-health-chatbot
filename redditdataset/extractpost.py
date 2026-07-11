import io
import zstandard as zstd
import orjson
import pandas as pd

INPUT_FILE = "Dyslexia_submissions.zst"
OUTPUT_FILE = "dyslexia_posts.csv"

rows = []

with open(INPUT_FILE, "rb") as fh:
    dctx = zstd.ZstdDecompressor()

    with dctx.stream_reader(fh) as reader:
        stream = io.TextIOWrapper(reader, encoding="utf-8")

        for line in stream:
            post = orjson.loads(line)

            rows.append({
                "id": post.get("id"),
                "title": post.get("title"),
                "body": post.get("selftext"),
                "author": post.get("author"),
                "score": post.get("score"),
                "upvote_ratio": post.get("upvote_ratio"),
                "num_comments": post.get("num_comments"),
                "created_utc": post.get("created_utc"),
                "permalink": post.get("permalink"),
                "url": post.get("url")
            })

df = pd.DataFrame(rows)

df.to_csv(OUTPUT_FILE, index=False)

print(f"Extracted {len(df)} posts.")
print(df.head())
import io
import zstandard as zstd
import orjson
import pandas as pd

INPUT_FILE = "Dyslexia_comments.zst"
OUTPUT_FILE = "dyslexia_comments.csv"

rows = []

with open(INPUT_FILE, "rb") as fh:
    dctx = zstd.ZstdDecompressor()

    with dctx.stream_reader(fh) as reader:
        stream = io.TextIOWrapper(reader, encoding="utf-8")

        for line in stream:
            comment = orjson.loads(line)

            rows.append({
                "id": comment.get("id"),
                "body": comment.get("body"),
                "author": comment.get("author"),
                "score": comment.get("score"),
                "created_utc": comment.get("created_utc"),
                "link_id": comment.get("link_id"),
                "parent_id": comment.get("parent_id")
            })

df = pd.DataFrame(rows)

df.to_csv(OUTPUT_FILE, index=False)

print(f"Extracted {len(df)} comments.")
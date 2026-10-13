"""Acquire a small checksummed public-source corpus; never private user data.

Store short book-body extracts, not the branded ebook/header/license text.
Source acknowledgements and upstream checksums remain in the manifest.
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import urllib.request

from .provenance import digest


def acquire(config: dict, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("Preserve existing corpus; choose a new output or reuse its verified manifest")
    output.mkdir(parents=True)
    rows = []
    for doc in config["documents"]:
        url = f"https://www.gutenberg.org/cache/epub/{doc['ebook']}/pg{doc['ebook']}.txt"
        request = urllib.request.Request(url, headers={"User-Agent": "HTIS-Context-public-research/0.1"})
        with urllib.request.urlopen(request, timeout=45) as response:
            raw = response.read()
            metadata = {"etag": response.headers.get("ETag"), "last_modified": response.headers.get("Last-Modified")}
        text = raw.decode("utf-8-sig")
        if "posted with permission of the copyright holder" in text[:10000].lower():
            raise ValueError("Copyright-restricted source requires independent review")
        start = re.search(r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .*?\*\*\*", text, re.IGNORECASE)
        end = re.search(r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", text, re.IGNORECASE)
        if start is None or end is None:
            raise ValueError(f"Cannot identify ebook body: {url}")
        body = text[start.end():end.start()]
        # Long paragraphs exclude tables of contents and most decorative headings.
        paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", body)
                      if len(p.split()) >= 65 and "gutenberg" not in p.lower()]
        if len(paragraphs) < 6:
            raise ValueError(f"Insufficient body prose: {url}")
        extract = "\n\n".join(paragraphs[3:7])
        row = {**doc, "url": url, "catalog": f"https://www.gutenberg.org/ebooks/{doc['ebook']}",
               "rights": "Public-domain book text in the United States; stripped body extract, source acknowledgement only",
               "license_reference": "https://www.gutenberg.org/policy/license.html",
               "upstream_sha256": hashlib.sha256(raw).hexdigest(), "extract_sha256": digest(extract),
               "extraction": "body between START/END markers; prose paragraphs with >=65 words; paragraphs 3:7",
               "text": extract, **metadata}
        rows.append(row)
        print(f"Acquired {doc['id']}: {len(extract)} extract characters", flush=True)
    manifest = {"schema": "htis-context-public-corpus-v1", "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
                "configuration_sha256": digest(config), "documents": rows}
    manifest["documents_sha256"] = digest(rows)
    (output / "corpus.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def read_corpus(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result["schema"] != "htis-context-public-corpus-v1" or digest(result["documents"]) != result["documents_sha256"]:
        raise ValueError("Corpus manifest checksum/schema mismatch")
    seen = set()
    for doc in result["documents"]:
        if doc["id"] in seen or digest(doc["text"]) != doc["extract_sha256"]:
            raise ValueError("Duplicate document or changed public extract")
        if doc["split"] not in ("train", "validation", "test"):
            raise ValueError("Unknown document split")
        seen.add(doc["id"])
    return result

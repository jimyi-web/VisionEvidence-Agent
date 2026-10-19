"""Download bounded official metadata only, pinning a commit; never fetch images."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen
from urllib.error import HTTPError

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True, help="New local directory")
selector = parser.add_mutually_exclusive_group()
selector.add_argument("--revision", help="Exact official commit; avoids GitHub API rate limits")
selector.add_argument("--ref", choices=("master",), help="Explicit branch snapshot, pinned by downloaded content hashes instead of a commit")
args = parser.parse_args()


def fetch(url):
    with urlopen(Request(url, headers={"User-Agent": "VisionEvidence-Agent-research-metadata"}), timeout=30) as response:
        content = response.read(2 * 1024 * 1024 + 1)
    if len(content) > 2 * 1024 * 1024:
        raise ValueError("metadata response exceeds 2 MiB cap")
    return content


commit_url = "https://api.github.com/repos/stevejpapad/image-text-verification/commits/master"
if args.ref:
    revision = args.ref
elif args.revision:
    revision = args.revision
else:
    try:
        revision = json.loads(fetch(commit_url))["sha"]
    except HTTPError as exc:
        if exc.code != 403:
            raise
        # Public API rate limits are common on shared networks. Read only the
        # repository page's explicit branch OID; never guess an upstream version.
        html = fetch("https://github.com/stevejpapad/image-text-verification").decode("utf-8")
        match = re.search(r'"currentOid":"([0-9a-f]{40})"', html)
        if match is None:
            raise ValueError("GitHub API rate-limited and branch OID unavailable; pass --revision") from exc
        revision = match.group(1)
if not args.ref and not re.fullmatch(r"[0-9a-f]{40}", revision):
    parser.error("revision must be a full 40-character commit SHA")
args.output.mkdir(parents=True, exist_ok=False)
files = {}
for source, local in (("VERITE/VERITE.csv", "VERITE.csv"),
                      ("VERITE/VERITE_articles.csv", "VERITE_articles.csv"),
                      ("LICENSE", "LICENSE"), ("README.md", "README.md")):
    url = f"https://raw.githubusercontent.com/stevejpapad/image-text-verification/{revision}/{source}"
    content = fetch(url)
    (args.output / local).write_bytes(content)
    files[local] = {"url": url, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
snapshot = hashlib.sha256(((args.output / "VERITE.csv").read_bytes() +
                           (args.output / "VERITE_articles.csv").read_bytes())).hexdigest()
manifest = {"revision": f"snapshot-sha256:{snapshot}" if args.ref else revision,
            "upstream_commit": None if args.ref else revision, "source_ref": revision,
            "retrieved_utc": datetime.now(timezone.utc).isoformat(),
            "files": files, "images_downloaded": 0, "usage": "research purposes only; see official README"}
(args.output / "source.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(json.dumps(manifest, indent=2))

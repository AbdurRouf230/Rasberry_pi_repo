#!/usr/bin/env python3
"""Download ~100 Wikimedia Commons photos into samples/ (no simulation images)."""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SAMPLES = HERE / "samples"
UA = "IntelleSwarmSampleFetch/1.0 (YOLO bench; https://commons.wikimedia.org)"
API = "https://commons.wikimedia.org/w/api.php"
THUMB_W = 800

BOX_CATEGORIES = [
    "Category:Moving boxes",
    "Category:Corrugated boxes",
    "Category:Corrugated fiberboard cartons",
    "Category:Cardboard boxes",
    "Category:Paperboard boxes",
]
BOX_NAME = re.compile(
    r"box|carton|cardboard|umzug|karton|corrugat|moving.?box|packing",
    re.I,
)
MISS_CATEGORIES = [
    "Category:Lawns",
    "Category:Blue sky",
    "Category:Soils",
    "Category:Meadows",
    "Category:Asphalt",
]


def api(params: dict) -> dict:
    q = urllib.parse.urlencode({**params, "format": "json"})
    req = urllib.request.Request(f"{API}?{q}", headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def category_files(title: str, limit: int) -> list[str]:
    names: list[str] = []
    cont = None
    while len(names) < limit:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": title,
            "cmtype": "file",
            "cmlimit": "100",
        }
        if cont:
            params["cmcontinue"] = cont
        data = api(params)
        for row in data.get("query", {}).get("categorymembers", []):
            name = row.get("title") or ""
            if name.startswith("File:"):
                names.append(name)
        cont = data.get("continue", {}).get("cmcontinue")
        if not cont:
            break
        time.sleep(0.15)
    return names[:limit]


def image_thumb(file_title: str) -> tuple[str, str] | None:
    data = api({
        "action": "query",
        "titles": file_title,
        "prop": "imageinfo",
        "iiprop": "url|mime|size",
        "iiurlwidth": str(THUMB_W),
    })
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        info = (page.get("imageinfo") or [None])[0]
        if not info:
            return None
        mime = str(info.get("mime") or "")
        if mime not in {"image/jpeg", "image/png"}:
            return None
        url = info.get("thumburl") or info.get("url")
        if not url:
            return None
        ext = ".jpg" if "jpeg" in mime else ".png"
        return url, ext
    return None


def slug(file_title: str) -> str:
    raw = file_title.replace("File:", "", 1)
    raw = Path(raw).stem
    raw = urllib.parse.unquote(raw)
    raw = re.sub(r"[^A-Za-z0-9]+", "_", raw).strip("_").lower()
    return raw[:60] or "img"


def download(url: str, dest: Path) -> bool:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
    except Exception as exc:
        print(f"  fail {dest.name}: {exc}")
        return False
    if len(data) < 8000:
        print(f"  tiny {dest.name} {len(data)}")
        return False
    dest.write_bytes(data)
    return True


def fill(prefix: str, categories: list[str], want: int, seen: set[str]) -> int:
    got = 0
    SAMPLES.mkdir(exist_ok=True)
    for cat in categories:
        if got >= want:
            break
        print(f"list {cat}")
        files = category_files(cat, limit=max(250, want * 4))
        for title in files:
            if got >= want:
                break
            key = title.lower()
            if key in seen:
                continue
            if prefix.startswith("hit_") and not BOX_NAME.search(title):
                continue
            seen.add(key)
            info = image_thumb(title)
            time.sleep(0.12)
            if info is None:
                continue
            url, ext = info
            name = f"{prefix}{slug(title)}{ext}"
            dest = SAMPLES / name
            n = 2
            while dest.exists():
                dest = SAMPLES / f"{prefix}{slug(title)}_{n}{ext}"
                n += 1
            if download(url, dest):
                got += 1
                print(f"  {got}/{want} {dest.name}")
            else:
                if dest.exists():
                    dest.unlink()
    return got


def cleanup_samples() -> None:
    """Keep cardboard-like hit names; drop obvious off-topic Commons files."""
    hit_ok = re.compile(
        r"box|carton|cardboard|umzug|pack|corrugat|karton|koli|moving|parcel",
        re.I,
    )
    miss_drop = re.compile(
        r"kibbutz|home_in_|squadron|tallet|boerderij|einhashlosha|grinter|"
        r"digging|dirtylens|loosing_prime",
        re.I,
    )
    for path in SAMPLES.iterdir():
        if not path.is_file():
            continue
        name = path.name
        if name.startswith("hit_") and not hit_ok.search(name):
            path.unlink()
            continue
        if re.search(r"_2\.(jpg|png)$", name, re.I):
            path.unlink()
            continue
        if name.startswith("miss_") and miss_drop.search(name):
            path.unlink()


def main() -> None:
    cleanup_samples()
    existing = list(SAMPLES.glob("hit_*.jpg")) + list(SAMPLES.glob("hit_*.png"))
    existing += list(SAMPLES.glob("miss_*.jpg")) + list(SAMPLES.glob("miss_*.png"))
    have = len(existing)
    target = 100
    print(f"already {have} images, target {target}")
    seen: set[str] = set()
    hits_have = len(list(SAMPLES.glob("hit_*")))
    miss_have = len(list(SAMPLES.glob("miss_*")))
    need_hit = max(0, 80 - hits_have)
    need_miss = max(0, 20 - miss_have)
    extra = max(0, target - have - need_hit - need_miss)
    need_hit += extra
    print(f"download hit={need_hit} miss={need_miss}")
    h = fill("hit_online_", BOX_CATEGORIES, need_hit, seen)
    m = fill("miss_online_", MISS_CATEGORIES, need_miss, seen)
    total = len(list(SAMPLES.glob("hit_*"))) + len(list(SAMPLES.glob("miss_*")))
    print(f"added hit={h} miss={m} folder total={total}")


if __name__ == "__main__":
    main()

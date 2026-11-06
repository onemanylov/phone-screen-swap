#!/usr/bin/env python3
"""Find new phone photos on Pinterest, no login needed. Photos go to photos/<name>/, with a numbered contact sheet to pick from.

  python3 find_photos.py search "taking photo of sneakers with iphone" --limit 60
  python3 find_photos.py related https://www.pinterest.com/pin/1234567890/ --limit 60
  python3 find_photos.py sheet photos/some-folder

Pinterest's "more like this" on a photo you liked finds real snapshots best. Search works for the first seeds: queries about
taking a photo beat queries about a hand. "related" needs a pin with a number in its link.
Photos belong to their authors: check you may use one before you publish it.
"""
import argparse, re, shutil, subprocess, sys, urllib.parse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent / "photos"
OK = (".jpg", ".jpeg", ".png", ".webp")


def fetch(url, name, limit):
    if not shutil.which("gallery-dl") and subprocess.run([sys.executable, "-m", "gallery_dl", "--version"], capture_output=True).returncode:
        sys.exit("gallery-dl is missing: pip install -r requirements.txt")
    out = ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "gallery_dl"] if not shutil.which("gallery-dl") else ["gallery-dl"]
    subprocess.run(cmd + ["--range", f"1-{limit}", "-D", str(out), "-f", "{id}.{extension}", "--filter", "extension in ('jpg', 'jpeg', 'png', 'webp')", url], stdout=subprocess.DEVNULL)
    sheet(out)


def sheet(folder, cols=8, h=340):
    folder = Path(folder)
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in OK and p.name != "_sheet.jpg")
    tiles, kept = [], []
    w = int(h * 9 / 16)
    for p in files:
        im = cv2.imread(str(p))
        if im is None:
            continue
        s = h / im.shape[0]
        t = cv2.resize(im, (max(1, int(im.shape[1] * s)), h), interpolation=cv2.INTER_AREA)[:, :w]
        t = cv2.copyMakeBorder(t, 0, 0, 0, w - t.shape[1], cv2.BORDER_CONSTANT)
        cv2.putText(t, str(len(kept)), (6, 28), 0, .8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(t, str(len(kept)), (6, 28), 0, .8, (0, 255, 255), 2, cv2.LINE_AA)
        tiles.append(t)
        kept.append(p)
    if not tiles:
        sys.exit(f"nothing in {folder}")
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(str(folder / "_sheet.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    for i, p in enumerate(kept):
        print(i, p)
    print(f"{len(kept)} photos, contact sheet: {folder / '_sheet.jpg'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["search", "related", "sheet"])
    ap.add_argument("what", help="search words, a pin's link or number, or a folder")
    ap.add_argument("--limit", type=int, default=60)
    a = ap.parse_args()
    if a.cmd == "sheet":
        sheet(a.what)
    elif a.cmd == "search":
        fetch("https://www.pinterest.com/search/pins/?q=" + urllib.parse.quote_plus(a.what), re.sub(r"[^a-z0-9]+", "-", a.what.lower()).strip("-"), a.limit)
    else:
        pin = re.search(r"(\d{6,})", a.what)
        if not pin:
            sys.exit("related needs a pin with a number in its link, e.g. https://www.pinterest.com/pin/1234567890/")
        fetch(f"https://www.pinterest.com/pin/{pin.group(1)}/#related", "related-" + pin.group(1), a.limit)


if __name__ == "__main__":
    main()

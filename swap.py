#!/usr/bin/env python3
"""Put any image on the screen of every phone template.

  python3 swap.py my-image.png                      all templates -> out/my-image/<template>.jpg
  python3 swap.py my-image.png -t a841f108,79186658 only these
  python3 swap.py screenshot.png --statusbar        a real iPhone screenshot: its time and icons move to where each phone shows them
  python3 swap.py                                   list the templates
"""
import argparse, json, os, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
SW, SH = 591, 1280   # the screen, in the pixels every template is measured in
T = "templates/corners"


def templates():
    return sorted(p.name.split(".")[0] for p in (ROOT / T).glob("*.recipe.json"))


def fit(path, out):
    """Scale and centre-crop to the screen's shape. Nothing gets stretched."""
    im = cv2.imread(str(path))
    if im is None:
        sys.exit(f"can't read {path}")
    h, w = im.shape[:2]
    if abs(w / h - SW / SH) > .01:
        print(f"{path}: {w}x{h} is not phone-shaped ({SW}x{SH}), centre-cropping it")
    s = max(SW / w, SH / h)
    r = cv2.resize(im, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    y, x = (r.shape[0] - SH) // 2, (r.shape[1] - SW) // 2
    cv2.imwrite(str(out), r[y:y + SH, x:x + SW])


def one(name, shot, statusbar, out, tmp):
    screen = shot
    if statusbar and (ROOT / T / f"{name}.status.json").exists():
        screen = f"{tmp}/{name}.screen.png"
        env = dict(os.environ, SLIDE_SHOT=shot, SLIDE_SCREEN_OUT=screen)
        subprocess.run([sys.executable, "tools/slide_status.py", name, "build"], cwd=ROOT, env=env, check=True, capture_output=True)
    recipe = json.load(open(ROOT / T / f"{name}.recipe.json"))
    recipe["screenshot"] = screen
    rf = f"{tmp}/{name}.json"
    json.dump(recipe, open(rf, "w"))
    r = subprocess.run([sys.executable, "tools/slide_final.py", rf, str(out / f"{name}.jpg")], cwd=ROOT, capture_output=True, text=True)
    if r.returncode:
        print(name, "failed:", r.stderr[-600:] or r.stdout[-600:])
        return False
    print(out / f"{name}.jpg")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="?")
    ap.add_argument("-t", "--templates", help="comma-separated template names; default: all")
    ap.add_argument("--statusbar", action="store_true", help="the image is a real iPhone screenshot: move its time and icons to each phone's own spot")
    ap.add_argument("-o", "--out", help="output folder; default: out/<image name>")
    a = ap.parse_args()
    names = templates()
    if a.image is None:
        print("\n".join(names))
        return
    if a.templates:
        wanted = a.templates.split(",")
        bad = [n for n in wanted if n not in names]
        if bad:
            sys.exit(f"no such template: {bad}. There are: {names}")
        names = wanted
    src = Path(a.image).resolve()
    out = Path(a.out).resolve() if a.out else ROOT / "out" / src.stem
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        shot = f"{tmp}/screen.png"
        fit(src, shot)
        with ThreadPoolExecutor(max(1, min(4, os.cpu_count() or 1))) as ex:
            ok = list(ex.map(lambda n: one(n, shot, a.statusbar, out, tmp), names))
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()

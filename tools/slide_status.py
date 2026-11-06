"""A real iPhone screenshot's time and icons moved to where this phone really shows them. They are cut off the screenshot and placed back by hand, one group at a time, on a contact sheet: each tile is the frame with our group at one position, the original photo's own bright shapes outlined in magenta over it.
  python3 tools/slide_status.py <prefix> init <time x> <time y> <time scale> <icons x> <icons y> <icons scale>   -> centres in screenshot pixels (591 wide). Measure the start on the flattened photo
  python3 tools/slide_status.py <prefix> base          -> the frame without any status, the sheets' background (needs the recipe)
  python3 tools/slide_status.py <prefix> sheet <time|icons> <step> <out.jpg>   -> 5x5: x goes across, y goes down
  python3 tools/slide_status.py <prefix> scales <time|icons> <s,s,s> <out.jpg>
  python3 tools/slide_status.py <prefix> set <time|icons> <dx> <dy> [scale]
  python3 tools/slide_status.py <prefix> bar <d>       -> Safari's bottom bar wider by d px each side (screenshot pixels): the back button and pill's left end go left, tabs and pill's right end go right, only the pill's flat filler stretches, so no letter or icon changes shape. Run build after it
  python3 tools/slide_status.py <prefix> build         -> writes corners/<prefix>.screen.png and points the recipe's screenshot at it
All in templates/corners/. The original screenshot is not changed."""
import glob, json, os, subprocess, sys
import cv2, numpy as np
BOX = dict(time=(84, 30, 140, 58), icons=(414, 28, 542, 58)); SW, SH = 591, 1280; D = "templates/corners"; SRC = os.environ.get("SLIDE_SHOT", "assets/example-screen.png"); OUT = os.environ.get("SLIDE_SCREEN_OUT")   # BOX: where these groups stand on a 591-wide screenshot from an iPhone with an island
pre, cmd = sys.argv[1:3]; sf = f"{D}/{pre}.status.json"; rf = f"{D}/{pre}.recipe.json"
def clean():
    s = cv2.imread(SRC).astype(np.float32)
    for x0, y0, x1, y1 in BOX.values():
        for y in range(y0 - 2, y1 + 3):
            l, r = s[y, x0 - 8:x0 - 3].mean(0), s[y, x1 + 3:x1 + 8].mean(0); t = np.linspace(0, 1, x1 - x0 + 6)[:, None]; s[y, x0 - 3:x1 + 3] = l * (1 - t) + r * t
    return widen(s, json.load(open(sf)).get("bar", 0) if os.path.exists(sf) else 0)
def widen(s, d):
    """piecewise along x, only in the bar's rows; the address text stays where it is"""
    if not d: return s
    # only the bar's own shapes move: the page's text above and below stays put. Shapes measured on the screenshot: two circles and a pill, rows 1160 to 1233
    m = np.zeros((SH, SW), np.uint8); cy, r = 1197, 37
    for cx in (82, 509): cv2.circle(m, (cx, cy), r, 255, -1, cv2.LINE_AA)
    cv2.rectangle(m, (133 + r, cy - r), (457 - r, cy + r), 255, -1); [cv2.circle(m, (c, cy), r, 255, -1, cv2.LINE_AA) for c in (133 + r, 457 - r)]
    hole = cv2.GaussianBlur(cv2.dilate(m, np.ones((3, 3), np.uint8)), (0, 0), .7).astype(np.float32)[..., None] / 255; bg = np.median(s[1250:1262], axis=(0, 1)); out = s * (1 - hole) + bg * hole
    m2 = m.copy(); cv2.rectangle(m2, (118, 1186), (134, 1210), 255, -1); cv2.rectangle(m2, (456, 1186), (472, 1210), 255, -1)   # the gaps go along with what they show of the hidden line
    src = np.interp(np.arange(SW, dtype=np.float32), [0, 198 - d, 206, 392, 402 + d, SW - 1], [d, 198, 206, 392, 402, SW - 1 - d]).astype(np.float32); X = np.tile(src, (SH, 1)); Y = np.tile(np.arange(SH, dtype=np.float32)[:, None], (1, SW)); moved = cv2.remap(s, X, Y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE); mm = cv2.GaussianBlur(cv2.remap(m2, X, Y, cv2.INTER_LINEAR), (0, 0), .6).astype(np.float32)[..., None] / 255; return out * (1 - mm) + moved * mm
def patch(g):
    x0, y0, x1, y1 = BOX[g]; return cv2.imread(SRC)[y0:y1, x0:x1].astype(np.float32)
def place(canvas, g, cx, cy, sc):
    """the group goes on with lighten or darken, so no edge of the patch can show"""
    p = patch(g); h, w = p.shape[:2]; bright = np.median(p) > 127; M = np.array([[sc, 0, cx - sc * w / 2], [0, sc, cy - sc * h / 2]], float); o = cv2.warpAffine(p, M, (canvas.shape[1], canvas.shape[0]), flags=cv2.INTER_AREA if sc < 1 else cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255) if bright else (0, 0, 0)); return np.minimum(canvas, o) if bright else np.maximum(canvas, o)   # dark page: lighten; bright page: darken
R = json.load(open(rf)); frame = R["frame"]; Q = np.float32(json.load(open(R["corners"]))["q"]); H = cv2.getPerspectiveTransform(np.float32([[0, 0], [SW, 0], [SW, SH], [0, SH]]), Q)
if cmd == "init":
    v = [float(x) for x in sys.argv[3:9]]; json.dump(dict(time=dict(x=v[0], y=v[1], s=v[2]), icons=dict(x=v[3], y=v[4], s=v[5])), open(sf, "w"), indent=1); sys.exit()
S = json.load(open(sf)) if os.path.exists(sf) else None
if cmd == "base":
    cv2.imwrite(f"{D}/{pre}.clean.png", np.clip(clean(), 0, 255).astype(np.uint8)); r = dict(R, screenshot=f"{D}/{pre}.clean.png"); t = f"{D}/{pre}.base.recipe.json"; json.dump(r, open(t, "w")); print(subprocess.run([sys.executable, "tools/slide_final.py", t, f"{D}/{pre}.base.jpg"], capture_output=True, text=True).stdout[-120:]); os.remove(t)
elif cmd == "bar":
    S["bar"] = float(sys.argv[3]); json.dump(S, open(sf, "w"), indent=1); print(S["bar"])
elif cmd == "set":
    g = sys.argv[3]; S[g]["x"] += float(sys.argv[4]); S[g]["y"] += float(sys.argv[5])
    if len(sys.argv) > 6: S[g]["s"] = float(sys.argv[6])
    json.dump(S, open(sf, "w"), indent=1); print(S)
elif cmd == "build":
    c = clean()
    for g in BOX: c = place(c, g, S[g]["x"], S[g]["y"], S[g]["s"])
    p = OUT or f"{D}/{pre}.screen.png"; cv2.imwrite(p, np.clip(c, 0, 255).astype(np.uint8)); print(p)
    if not OUT: R["screenshot"] = p; json.dump(R, open(rf, "w"), indent=1)
elif cmd in ("sheet", "scales"):
    g = sys.argv[3]; base = cv2.imread(f"{D}/{pre}.base.jpg"); crop = cv2.imread(f"templates/crop/{frame}.jpg"); st = S[g]; ctr = cv2.perspectiveTransform(np.float32([[[st["x"], st["y"]]]]), H)[0, 0]; hw, hh = (54, 26) if g == "time" else (76, 26); Z = 2.4
    x0, y0 = int(ctr[0] - hw), int(ctr[1] - hh); xc, yc = x0 + 2 * hw, y0 + 2 * hh; gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)[y0:yc, x0:xc]; gray = cv2.GaussianBlur(gray, (0, 0), .6); thr = np.percentile(gray, 50) + (np.percentile(gray, 99.5) - np.percentile(gray, 50)) * .5; cnts, _ = cv2.findContours((gray > thr).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    def tile(cx, cy, sc, lab, centre=False):
        scr = place(np.zeros((SH, SW, 3), np.float32), g, cx, cy, sc); w = cv2.warpPerspective(scr, H, (base.shape[1], base.shape[0]), flags=cv2.INTER_LINEAR); fr = np.maximum(base.astype(np.float32), w)[y0:yc, x0:xc]; fr = cv2.resize(np.clip(fr, 0, 255).astype(np.uint8), None, fx=Z, fy=Z, interpolation=cv2.INTER_CUBIC)
        for c in cnts:
            if cv2.contourArea(c) > 6: cv2.polylines(fr, [np.round(c * Z * 8).astype(np.int32)], True, (255, 0, 255), 1, cv2.LINE_AA, shift=3)
        cv2.rectangle(fr, (0, 0), (8 + 10 * len(lab), 18), (0, 200, 200) if centre else (0, 0, 0), -1); cv2.putText(fr, lab, (3, 14), 0, .42, (255, 255, 255), 1, cv2.LINE_AA); return fr
    if cmd == "sheet":
        step = float(sys.argv[4]); rows = [np.hstack([tile(st["x"] + i * step, st["y"] + j * step, st["s"], f"{i * step:+g} {j * step:+g}", i == 0 and j == 0) for i in range(-2, 3)]) for j in range(-2, 3)]; out = sys.argv[5]
    else:
        ss = [float(x) for x in sys.argv[4].split(",")]; rows = [np.hstack([tile(st["x"], st["y"], s, f"s={s}") for s in ss])]; out = sys.argv[5]
    ref = cv2.resize(crop[y0:yc, x0:xc], None, fx=Z, fy=Z, interpolation=cv2.INTER_CUBIC); cv2.imwrite(out, np.vstack(rows + [np.hstack([ref] + [np.zeros_like(ref)] * (rows[0].shape[1] // ref.shape[1] - 1))]), [cv2.IMWRITE_JPEG_QUALITY, 92])

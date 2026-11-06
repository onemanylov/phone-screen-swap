"""Everything after the trace, on the straightened phone: measure once, build, compare with the photo. So nothing is read off a frame by eye first and a phone takes minutes, not rounds.
  python3 tools/slide_check.py <prefix> flat <out.jpg> [margin]   -> the phone straightened, ruler in screenshot px (591 wide). State: corners/<prefix>.json
  python3 tools/slide_check.py <prefix> measure                   -> notch or island, status groups, the blacks; prints the options and the `slide_status.py init` line to use
  python3 tools/slide_check.py <prefix> leak                          -> thermal map of what the photo still shows through the glass
  python3 tools/slide_check.py <prefix> build                     -> the recipe's frame to out/<prefix>.jpg, red test, out/<prefix>-split.jpg (photo | red | ours), out/<prefix>-zoom.jpg, out/<prefix>-top.jpg (straightened top: the photo with our cutout's outline, ours, red), blacks photo against ours"""
import glob, json, os, subprocess, sys, tempfile
import cv2, numpy as np
pre, cmd = sys.argv[1:3]; D = "templates"; St = json.load(open(f"{D}/corners/{pre}.json")); C = np.array(St["c"], float); K = St["k"]; Q = np.array(St["q"], float); FW, FH = 591., 1280.
H = cv2.getPerspectiveTransform(np.float32([[0, 0], [FW, 0], [FW, FH], [0, FH]]), np.float32(Q)).astype(float); CROP = glob.glob(f"{D}/crop/{pre}*")[0]
def grid(xs, ys):
    gx, gy = np.meshgrid(xs, ys); h = np.c_[gx.ravel(), gy.ravel(), np.ones(gx.size)] @ H.T; p = h[:, :2] / h[:, 2:]; r = p.copy()
    for _ in range(4): d = r - C; r = C + (p - C) / (1 + K * (d ** 2).sum(-1, keepdims=True) / 1e6)
    return r.reshape(gx.shape + (2,)).astype(np.float32)
def flat(im, m=24, Z=2):
    g = grid(np.arange(-m, FW + m, 1 / Z), np.arange(-m, FH + m, 1 / Z)); return cv2.remap(im, g[..., 0], g[..., 1], cv2.INTER_CUBIC)
AREAS = dict(frame_top=(100, -14, 180, -5), frame_right=(596, 300, 605, 900), frame_bottom=(150, 1283, 450, 1294), frame_left=(-14, 300, -5, 900))
def level(im, b):
    g = grid(np.arange(b[0], b[2], 1.), np.arange(b[1], b[3], 1.)); return np.percentile(cv2.remap(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), g[..., 0], g[..., 1], cv2.INTER_LINEAR), [5, 50, 95]).round(0).astype(int).tolist()
if cmd == "flat":
    m = int(sys.argv[4]) if len(sys.argv) > 4 else 24; f = flat(cv2.imread(CROP), m)
    for x in range(0, 591, 50): cv2.line(f, ((m + x) * 2, 0), ((m + x) * 2, 14), (0, 255, 255), 1); cv2.putText(f, str(x), ((m + x) * 2 + 2, 12), 0, .35, (0, 255, 255), 1)
    for y in range(0, 1280, 20): cv2.line(f, (0, (m + y) * 2), (14, (m + y) * 2), (0, 255, 255), 1); cv2.putText(f, str(y), (16, (m + y) * 2 + 4), 0, .35, (0, 255, 255), 1)
    cv2.imwrite(sys.argv[3], f, [cv2.IMWRITE_JPEG_QUALITY, 95]); sys.exit()
if cmd == "leak":   # thermal map of what the photo still shows through the glass: the frame is built on a pure black screen with no grain and no reflection, straightened, brightness on a log scale. Anything lit on the map came from the photo, not from us: a line along a cutout, specks, glare. The speaker is meant to be there. Look at it on every phone
    os.makedirs(os.path.dirname(f"out/{pre}-leak.jpg") or ".", exist_ok=True); rf = f"{D}/corners/{pre}.recipe.json"; R = json.load(open(rf)); t = tempfile.mkdtemp(); cv2.imwrite(f"{t}/black.png", np.zeros((1280, 591, 3), np.uint8)); json.dump(dict(R, screenshot=f"{t}/black.png", opts=dict(R["opts"], egain=0, grain=0, floor=0, bloom=0)), open(f"{t}/r.json", "w")); subprocess.run([sys.executable, "tools/slide_final.py", f"{t}/r.json", f"{t}/b.jpg"], check=True, stdout=subprocess.DEVNULL)
    g = cv2.cvtColor(flat(cv2.imread(f"{t}/b.jpg"), 24, 1), cv2.COLOR_BGR2GRAY).astype(np.float32); heat = cv2.applyColorMap(np.clip(np.log1p(g) / np.log(64) * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    inner = g[24:24 + 1280, 24:24 + 591]; hot = inner > 12; n, lab, st, _ = cv2.connectedComponentsWithStats(hot.astype(np.uint8), connectivity=8); spots = sorted(((int(st[i, 4]), int(st[i, 0]), int(st[i, 1]), int(st[i, 2]), int(st[i, 3])) for i in range(1, n)), reverse=True)[:6]
    print(f"{pre}: lit pixels on the screen {int(hot.sum())}, biggest spots (area, x, y, w, h): {spots}"); cv2.imwrite(f"out/{pre}-leak.jpg", heat, [cv2.IMWRITE_JPEG_QUALITY, 92]); sys.exit()
if cmd == "measure":
    Z = 2; rows = np.arange(0, 90, .5); g = grid(np.arange(100, 491, 1 / Z), rows); im = cv2.imread(CROP); L_ = cv2.GaussianBlur(cv2.remap(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), g[..., 0], g[..., 1], cv2.INTER_LINEAR).astype(np.float32), (0, 0), 1.); xs = np.arange(100, 491, 1 / Z); c = int(np.searchsorted(xs, 295.5)); black = float(np.median(L_[np.searchsorted(rows, 20):np.searchsorted(rows, 34), c - 60:c + 60])); pgL, pgR = np.median(L_[:, :40], axis=1), np.median(L_[:, -40:], axis=1); W = []
    for j, y in enumerate(rows):
        run = lambda s, tr: next((i for i in range(c, 8, -1) if (L_[j, i - 8:i] > tr).all()) if s < 0 else (i for i in range(c, len(xs) - 8) if (L_[j, i:i + 8] > tr).all()), None); a, b = run(-1, (black + pgL[j]) / 2), run(1, (black + pgR[j]) / 2); W.append((y, xs[a] if a else np.nan, xs[b] if b else np.nan, L_[j, c] < (black + (pgL[j] + pgR[j]) / 2) / 2))
    W = np.array(W, float); on = W[W[:, 3] > 0]; y0, y1 = on[:, 0].min(), on[:, 0].max(); mid = on[(on[:, 0] > y0 + (y1 - y0) * .2) & (on[:, 0] < y0 + (y1 - y0) * .5)]; x0, x1 = np.nanmedian(mid[:, 1]), np.nanmedian(mid[:, 2]); print(f"black here {black:.0f}, page beside it {np.median(pgL):.0f} and {np.median(pgR):.0f}: if these are close, the numbers below mean nothing, measure by eye on `flat`"); print(f"width {x0:.1f} to {x1:.1f}, centre {(x0 + x1) / 2:.1f} of 295.5, rows {y0:.1f} to {y1:.1f}")
    near = on[on[:, 0] > y1 - 40]; off = lambda r, v: 0. if v <= y1 - r else r - np.sqrt(max(r ** 2 - (v - (y1 - r)) ** 2, 0)); best = min(range(14, 50), key=lambda r: np.nanmean([min(abs((v[1] - x0) - off(r, v[0])), 6) + min(abs((x1 - v[2]) - off(r, v[0])), 6) for v in near]))
    if y0 < 4: print(f"opts: cut=1 cx0={x0:.1f} cx1={x1:.1f} cy1={y1:.1f} cr={best} ce=9 grille=45")
    else: print(f"opts: cut=2 cx0={x0:.1f} cx1={x1:.1f} cy0={y0:.1f} cy1={y1:.1f}")
    f = cv2.GaussianBlur(cv2.cvtColor(flat(im, 0), cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.)
    def group(a, b, y_=(20, 64), w=.5):
        p = f[y_[0] * 2:y_[1] * 2, int(a * 2):int(b * 2)]; t = np.median(p) + (np.percentile(p, 99.5) - np.median(p)) * w; ys, x_ = np.where(p > t); return a + x_.min() / 2, a + x_.max() / 2, y_[0] + ys.min() / 2, y_[0] + ys.max() / 2
    lg = group(40, x0 - 8); rg = group(x1 + 8, 565, (int(lg[2]) - 6, int(lg[3]) + 4), .25); s = 1.03; print(f"left group x {lg[0]:.1f}-{lg[1]:.1f} y {lg[2]:.1f}-{lg[3]:.1f}, right group x {rg[0]:.1f}-{rg[1]:.1f} y {rg[2]:.1f}-{rg[3]:.1f}")
    print(f"python3 tools/slide_status.py {pre} init {(lg[0] + lg[1]) / 2:.1f} {(lg[2] + lg[3]) / 2 + .5:.1f} {s} {rg[1] - 57 * s:.1f} {(lg[2] + lg[3]) / 2:.1f} {s}")
    print("blacks on the photo", {a: level(im, b) for a, b in dict(AREAS, cutout=(x0 + 45, y0 + 8, x1 - 45, y1 - 8)).items()}); sys.exit()
if cmd == "build":
    os.makedirs("out", exist_ok=True); rf = f"{D}/corners/{pre}.recipe.json"; R = json.load(open(rf)); o = R["opts"]; t = tempfile.mkdtemp(); red = np.zeros((1280, 591, 3), np.uint8); red[...] = (30, 30, 220); cv2.imwrite(f"{t}/red.png", red); rr = dict(R, screenshot=f"{t}/red.png", opts=dict(o, egain=0, grain=0, grille=0)); json.dump(rr, open(f"{t}/red.json", "w")); out = f"out/{pre}.jpg"
    for rc, ot in ((rf, out), (f"{t}/red.json", f"{t}/red.jpg")): subprocess.run([sys.executable, "tools/slide_final.py", rc, ot], check=True, stdout=subprocess.DEVNULL)
    a, r_, c = cv2.imread(CROP), cv2.imread(f"{t}/red.jpg"), cv2.imread(out); pts = np.array(St["glass"]); y0, y1 = max(int(pts[:, 1].min()) - 60, 0), min(int(pts[:, 1].max()) + 60, a.shape[0]); cv2.imwrite(f"out/{pre}-split.jpg", cv2.resize(np.hstack([a, r_, c])[y0:y1], None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 92])
    def tile(p, w=90, Z=2): x, y = int(np.clip(p[0], w, a.shape[1] - w)), int(np.clip(p[1], w, a.shape[0] - w)); return np.hstack([cv2.resize(i[y - w:y + w, x - w:x + w], None, fx=Z, fy=Z, interpolation=cv2.INTER_CUBIC) for i in (a, c)])
    cv2.imwrite(f"out/{pre}-zoom.jpg", np.hstack([np.vstack([tile(Q[0]), tile((Q[0] + Q[1]) / 2)]), np.vstack([tile(Q[1]), tile(Q[2])]), np.vstack([tile(Q[3]), tile((Q[0] + Q[3]) / 2)])]), [cv2.IMWRITE_JPEG_QUALITY, 90])
    fo, fr, fc = (flat(i, 24, 4)[:94 * 4, (24 + 100) * 4:(24 + 491) * 4] for i in (a, r_, c)); cn, _ = cv2.findContours((((fr[..., 2].astype(int) - fr[..., 1]) > 100) * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE); fo = cv2.convertScaleAbs(fo, alpha=1.6); cv2.drawContours(fo, cn, -1, (0, 255, 0), 1); cv2.imwrite(f"out/{pre}-top.jpg", np.vstack([fo, fc, fr]), [cv2.IMWRITE_JPEG_QUALITY, 90])   # the straightened top: the photo with our cutout's outline, ours, red
    A = dict(AREAS)
    if "cx0" in o: y_ = o.get("cy0", 0); A["cutout"] = (o["cx0"] + 45, y_ + 8, o["cx1"] - 45, o["cy1"] - 8)
    for n, b in A.items(): print(f"{n}: photo {level(a, b)}, ours {level(c, b)}")

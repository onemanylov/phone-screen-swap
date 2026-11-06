"""Proof of an outline from slide_trace.py on what the fit didn't see.
A. Each long line of the old screen in turn is left out, the fit runs on the rest, and that line is judged: how straight, how far off the axis.
B. The perimeter is cut into 16 blocks; the fit sees every other block, the rest are judged.
C. Never in the fit: a square cover must come out square; left and right margins equal; the island centred; the corner radius against iPhone's published one.
D. The picture: the outline on real pixels, nothing stretched, 8x, 16 places, all four corners among them; and the miss of every traced point.
  python3 tools/slide_proof.py <prefix> <state.json> <out.jpg>"""
import json, sys
import cv2, numpy as np
pre, sf, out = sys.argv[1:4]; sys.argv = ["x", pre, "strips", out, sf]
src = open(__file__.replace("slide_proof.py", "slide_trace.py")).read(); exec(src[:src.index('if MODE == "offset"')])
St = json.load(open(sf)); q, R, k = np.array(St["q"], float), St["R"], St["k"]; C = np.array(St.get("c", C)); C0 = C.copy()
USEC = bool(St.get("dark")); rs, dr = trace(q, R, k); P = rs if St.get("kind", "rise") == "rise" else dr; lines = ui_lines(q, R, k); seam = seams(q, R, k); d0 = miss(P, q, R, k); Q = P[np.abs(d0) < 1.5]
def stat(d): d = np.abs(d); return f"median {np.median(d):.2f}, 90% under {np.percentile(d, 90):.2f}, under 1 px {100 * (d < 1).mean():.0f}% of {len(d)}"
print(f"bend centre {C.tolist()}; traced {len(P)} of 720 rays, used {len(Q)}; R {R} = {R / FW * 393:.1f} pt, k {k:.4f}")
print("A. line left out of the fit, then judged (rms: strays from straight; tilt: end to end off the axis; flat px)")
for i, l in enumerate(lines):
    C = C0.copy(); q_, R_, k_ = fit(Q, lines[:i] + lines[i + 1:], seam, q, R, k); H = homog(q_, BOX); f = apply(H, und(l["P"], k_)); a, b = (f[:, 0], f[:, 1]) if l["axis"] == "h" else (f[:, 1], f[:, 0]); c = np.polyfit(a, b, 1); print(f"   {l['axis']} at {l['at']:7.1f}, {l['len']:4.0f} long: rms {np.std(b - np.polyval(c, a)):.2f}, tilt {c[0] * np.ptp(a):+.2f}")
print("B. edge blocks left out")
idx = np.arange(len(Q)) * 16 // len(Q); v = []
for par in (0, 1):
    tr = idx % 2 == par; C = C0.copy(); q_, R_, k_ = fit(Q[tr], lines, seam, q, R, k); v.append(miss(Q[~tr], q_, R_, k_))
v = np.concatenate(v); print("  ", stat(v), f"signed mean {v.mean():+.2f}")
C = C0.copy(); print("C. never in the fit")
H = homog(q, BOX); by = {}
for l in lines: by.setdefault(l["axis"], []).append(l["at"])
g = rect_grid(q, k, *np.meshgrid(np.arange(591.), np.arange(1280.)), gray)
vs = sorted(by.get("v", [])); hs = sorted(by.get("h", []))
if len(vs) >= 2:
    L_, R__ = vs[0], vs[-1]; print(f"   margins: left {L_:.1f}, right {FW - R__:.1f}")
    pairs = [(a, b) for a in hs for b in hs if b - a > 200]; best = min(pairs, key=lambda p: abs((p[1] - p[0]) - (R__ - L_))) if pairs else None
    if best: print(f"   cover: {R__ - L_:.1f} wide, {best[1] - best[0]:.1f} high, square to {abs(best[1] - best[0] - (R__ - L_)) / (R__ - L_) * 100:.2f}%")
row = [np.where(g[y, 150:440] < 50)[0] + 150 for y in range(30, 58)]; row = [r for r in row if len(r) > 100]
if row: l_, r_ = np.median([r.min() for r in row]), np.median([r.max() for r in row]); print(f"   island: {l_:.1f} to {r_:.1f}, centre {(l_ + r_) / 2:.1f} of {FW / 2}")
print(f"   radius: {R / FW * 393:.1f} pt if 393 wide (14/15 Pro published 55), {R / FW * 375:.1f} pt if 375 wide (X/XS/11 Pro published 39)")
print("D. all traced, no point thrown out:", stat(d0))
f = apply(H, und(P, k)); arc = (np.abs(f[:, 0] - FW / 2) > FW / 2 - 1.53 * R) & (np.abs(f[:, 1] - FH / 2) > FH / 2 - 1.53 * R); print("   corners only:", stat(d0[arc])); print("   straight runs:", stat(d0[~arc]))
Hi = homog(BOX, q); o, _ = outline(R, 4000); O = bend(apply(Hi, o), k); s = .29 * R
places = [(s, s), (FW * .3, 0), (FW * .7, 0), (FW - s, s), (FW, FH * .2), (FW, FH * .4), (FW, FH * .6), (FW, FH * .8), (FW - s, FH - s), (FW * .7, FH), (FW * .3, FH), (s, FH - s), (0, FH * .8), (0, FH * .6), (0, FH * .4), (0, FH * .2)]; T = []; Z = 8; H_ = 26
for f_ in places:
    c = bend(apply(Hi, np.array([f_], float)), k)[0]; x0, y0 = c - H_; t = cv2.warpAffine(im, np.array([[Z, 0, -x0 * Z], [0, Z, -y0 * Z]], float), (2 * H_ * Z, 2 * H_ * Z), flags=cv2.INTER_CUBIC); cv2.polylines(t, [np.round((O - [x0, y0]) * Z * 8).astype(np.int32)], True, (0, 0, 255), 1, cv2.LINE_AA, shift=3); T.append(t)
sheet = np.vstack([np.hstack(T[i:i + 4]) for i in range(0, 16, 4)]); W_ = sheet.shape[1]; ch = np.full((200, W_, 3), 255, np.uint8)
for v_, col in ((0, (0, 0, 0)), (1, (190, 190, 190)), (-1, (190, 190, 190))): ch[int(100 - v_ * 40)] = col
for x_, b_ in zip((np.arange(len(P)) / len(P) * (W_ - 20) + 10).astype(int), (100 - np.clip(d0, -2.4, 2.4) * 40).astype(int)): cv2.circle(ch, (x_, b_), 2, (0, 0, 220), -1, cv2.LINE_AA)
cv2.putText(ch, "miss of every traced point round the phone. Grey: +-1 px", (10, 22), 0, .6, (0, 0, 0), 1, cv2.LINE_AA); cv2.imwrite(out, np.vstack([sheet, ch]), [cv2.IMWRITE_JPEG_QUALITY, 92])

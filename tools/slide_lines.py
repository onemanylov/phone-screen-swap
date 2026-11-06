"""The screen's four straight edges.
A side is a straight line, by physics: so only straight lines are ever tried. Each side is straightened along the rough guess; a line is then its y at end A and its y at end B, and every pair on a quarter-pixel grid gets a score. Nothing is fitted through local points, so the line can't bend or follow a blob.
Score: brightness goes through a logarithm (black bezel to a dark screen counts as much as black bezel to a bright one). Every short piece along the side measures the step at each y: what lies 1.5 to 6 px inside minus what lies 1.5 to 6 px outside. A thin bright seam or scratch has black on both sides, so no step; the screen has. Each piece votes for its innermost real step from black (a case or a bright band lies further out), so a stretch hidden by a thumb or glare just gives few votes and can't pull the line. The corners are where the lines cross.
  python3 tools/slide_lines.py <id prefix> '<[[x,y]*4 TL,TR,BR,BL] rough>' <out.jpg> [hand corners state.json to compare with]
`agree` is the share of pieces whose sharpest step lies within 1.6 px of the line; `bow` is where those steps lie against the line in 8 stretches along the side: a low agree or a bow over 2 px means look with your own eye"""
import glob, json, os, sys
import cv2, numpy as np

NAMES = ("top", "right", "bottom", "left")
pre, rough, out = sys.argv[1:4]
im = cv2.imread(glob.glob(f"templates/crop/{pre}*")[0])
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
LOG = np.log(gray + 4)
q = np.array(json.loads(rough), float)
R, STEP, PIECES = float(os.environ.get("R", 14)), .25, 200   # how far from the rough line a line may be tried, px
TOL = 1.6; OUT = 22.; MINW = 7.; BAND = (1.5, 6.); BLACK = np.log(50 + 4)   # the bezel, seen through glare at worst


def frame(a, b, c0):
    L = np.linalg.norm(b - a); u = (b - a) / L; n = np.array([-u[1], u[0]]); return L, u, (n if n @ (c0 - a) > 0 else -n)


def side(k, q, R):
    a, b = q[k], q[(k + 1) % 4]; L, u, n = frame(a, b, q.mean(0)); margin = .17 if L < 800 else .10   # past the margin the arc starts
    pad = R + BAND[1] + 1; ns = np.arange(-pad - OUT, pad + STEP, STEP); ts = np.linspace(margin * L, (1 - margin) * L, int((1 - 2 * margin) * L)); T, N = np.meshgrid(ts, ns)
    s = cv2.remap(LOG, (a[0] + u[0] * T + n[0] * N).astype(np.float32), (a[1] + u[1] * T + n[1] * N).astype(np.float32), cv2.INTER_LINEAR)
    ps = np.array_split(np.arange(len(ts)), PIECES); pt = np.array([ts[p].mean() for p in ps]); prof = np.stack([np.median(s[:, p], axis=1) for p in ps], axis=1)   # y by piece
    I = np.cumsum(np.vstack([np.zeros(prof.shape[1]), prof]), axis=0); lo, hi = int(BAND[0] / STEP), int(BAND[1] / STEP); ys = np.arange(-R, R + STEP, STEP); y0 = np.searchsorted(ns, ys)
    step = (I[y0 + hi + 1] - I[y0 + lo]) / (hi + 1 - lo) - (I[y0 - lo + 1] - I[y0 - hi]) / (hi + 1 - lo)   # inside minus outside
    outside = (I[y0 - lo + 1] - I[y0 - hi]) / (hi + 1 - lo); drop = np.gradient(cv2.GaussianBlur(prof, (0, 0), sigmaX=.01, sigmaY=2.5), STEP, axis=0); rim = ns[np.argmin(drop, axis=0)]   # the strongest fall going in: the frame's outer edge or the seam on it. The screen can't start closer to it than MINW: right behind it lies glare over the frame
    open_ = (outside < BLACK) & (ys[:, None] >= rim[None, :] + MINW); peak = np.where(open_, step, 0).max(axis=0); live = peak > .2; own = np.full(len(pt), np.nan)
    for c in np.where(live)[0]:   # walking out from the screen: the first real step from black is the screen's edge. A case or a bright band further out comes after it
        ok = (step[:, c] >= peak[c] * .6) & open_[:, c] & (step[:, c] >= np.roll(step[:, c], 1)) & (step[:, c] >= np.roll(step[:, c], -1)); own[c] = ys[np.where(ok)[0][-1]] if ok.any() else np.nan
    live = live & ~np.isnan(own)
    # every straight line on the grid: each piece votes for the lines passing within TOL of its own y. A photo bends a true edge by a couple of px (lens: on 895d3ee8 the band and the screen's edge bend alike), so the winner goes through the middle of the bend instead of lying on one stretch and missing the other by 5 px
    f = (pt - margin * L) / ((1 - 2 * margin) * L); best = (-1, 0, 0); col = np.where(live)[0]
    for ya in ys:
        d = (ya + (ys[None, :] - ya) * f[col, None]) - own[col, None]; sc = (np.exp(-(d / TOL) ** 4) + .1 * np.exp(-(d / .6) ** 2)).sum(axis=0); j = int(np.argmax(sc))
        if sc[j] > best[0]: best = (float(sc[j]), float(ya), float(ys[j]))
    ya, yb = best[1:]; line = ya + (yb - ya) * f; off = own - line; near = live & (np.abs(off) < 1.6)
    bow = [round(float(np.median(off[c][np.abs(off[c]) < 3])), 1) if (np.abs(off[c]) < 7).sum() > 3 else None for c in np.array_split(np.arange(len(pt)), 8)]
    per = (yb - ya) / ((1 - 2 * margin) * L); A = a + u * (margin * L) + n * ya - (u + n * per) * (margin * L); B = A + (u + n * per) * L   # the line itself, carried out to the rough side's ends
    return A, B, dict(side=NAMES[k], agree=round(float(near.sum() / max(live.sum(), 1)), 2), live=round(float(live.mean()), 2), shift=[round(float(np.dot(A - a, n)), 1), round(float(np.dot(B - b, n)), 1)], bow=bow)


def cross(p1, p2, p3, p4):
    t = np.linalg.solve(np.array([p2 - p1, p3 - p4]).T, p3 - p1); return p1 + t[0] * (p2 - p1)


for R_ in (R, 4, 4):   # the wide look only once: the search must not walk away from the rough guess round after round
    lines = [side(k, q, R_) for k in range(4)]
    q = np.array([cross(*lines[3][:2], *lines[0][:2]), cross(*lines[0][:2], *lines[1][:2]), cross(*lines[1][:2], *lines[2][:2]), cross(*lines[2][:2], *lines[3][:2])])
for l in lines: print(l[2])
print("corners", q.round(2).tolist())
if len(sys.argv) > 4:
    h = np.array(json.load(open(sys.argv[4]))["q"]); print("minus hand", (q - h).round(1).tolist(), "px", np.linalg.norm(q - h, axis=1).round(1).tolist())

# the same log view for the eye: each side straightened, red is the line, a tick every 5 px across
c0 = q.mean(0); rows = []; Z = 5; W = 1200
for k in range(4):
    a, b = q[k], q[(k + 1) % 4]; L, u, n = frame(a, b, c0); ts = np.linspace(-L * .05, L * 1.05, W); nz = np.arange(-24, 16, 1 / Z); T, N = np.meshgrid(ts, nz); X = (a[0] + u[0] * T + n[0] * N).astype(np.float32); Y = (a[1] + u[1] * T + n[1] * N).astype(np.float32); s_ = cv2.remap(im, X, Y, cv2.INTER_CUBIC); l_ = cv2.remap(LOG, X, Y, cv2.INTER_CUBIC); lo_, hi_ = np.log(6), np.log(150)
    lab = cv2.cvtColor(s_, cv2.COLOR_BGR2LAB).astype(np.float32); lab[:, :, 0] = np.clip((l_ - lo_) / (hi_ - lo_) * 255, 0, 255); s_ = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR); s_[int(24 * Z)] = (0, 0, 255)
    for t_ in range(-20, 16, 5): s_[int((24 + t_) * Z), :10] = (0, 255, 255); s_[int((24 + t_) * Z), -10:] = (0, 255, 255)
    cv2.putText(s_, NAMES[k], (14, 18), 0, .55, (255, 255, 255), 1, cv2.LINE_AA); rows += [s_, np.full((5, W, 3), 255, np.uint8)]
cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 93])
json.dump(dict(q=q.round(2).tolist()), open(out + ".json", "w"))

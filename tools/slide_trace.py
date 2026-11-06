"""Mark the phone, then rectify.
From first principles: the phone is rigid, its screen a rectangle with iOS's continuous corners. On the photo that is the shape + perspective + one smooth bend of the lens. So the fit is the check itself: it finds the perspective, the bend and the radius at which at once
 - every traced point of the edge lies on the shape,
 - every long straight line the old screen shows (a cover's sides, a progress bar, a banner) is straight and runs exactly along the screen's axes,
 - the thin seam on the frame is straight.
Edge points are found in log brightness (black to a dark screen counts as much as black to a bright one), but placed half way in linear light: a blurred edge of light really is there.
  python3 tools/slide_trace.py <prefix> '<[[x,y]*4 TL,TR,BR,BL] rough>' <out.jpg> [state.json]
  python3 tools/slide_trace.py <prefix> strips <out.jpg> <state.json>   -> each side rectified, across 5x
  python3 tools/slide_trace.py <prefix> offset <top,right,bottom,left> <state.json>   -> sides out by these flat px
Proof on what's left out: tools/slide_proof.py"""
import glob, json, sys
import cv2, numpy as np
from scipy.optimize import least_squares

pre, rough, out = sys.argv[1:4]; MODE = rough if rough in ("strips", "offset") else None
im = cv2.imread(glob.glob(f"templates/crop/{pre}*")[0]); h_, w_ = im.shape[:2]; gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32); LOG = np.log(gray + 4); lab_ = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32); CHROMA = np.hypot(lab_[..., 1] - 128, lab_[..., 2] - 128)   # the frame is neutral black; a dark screen is still a colour
FW, FH = 591., 1280.; BOX = np.array([[0, 0], [FW, 0], [FW, FH], [0, FH]]); C = np.array([w_ / 2., h_ / 2.]); BLACK = np.log(50 + 4); FREEC = False; OUTR = 45.; USEC = False; CHROMA_STEP = 3.; PT = 591 / 375
q0 = np.array(json.loads(rough), float) if not MODE else None


def homog(src, dst):
    A = []; b = []
    for (x, y), (u, v) in zip(np.asarray(src, float), np.asarray(dst, float)): A += [[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]]; b += [u, v]
    return np.r_[np.linalg.solve(np.array(A), np.array(b)), 1].reshape(3, 3)


def apply(H, p):
    h = np.c_[p, np.ones(len(p))] @ H.T; return h[:, :2] / h[:, 2:]


def und(p, k):
    d = p - C; return C + d * (1 + k * (d ** 2).sum(-1, keepdims=True) / 1e6)


def bend(p, k):
    r = p.copy()
    for _ in range(4): d = r - C; r = C + (p - C) / (1 + k * (d ** 2).sum(-1, keepdims=True) / 1e6)
    return r


def bez(a, b, c, d, n=24):
    t = np.linspace(0, 1, n)[:, None]; return (1 - t) ** 3 * a + 3 * (1 - t) ** 2 * t * b + 3 * (1 - t) * t ** 2 * c + t ** 3 * d


def corner(R):
    """iOS's continuous corner in folded coords (x, y from the centre): the published curve of UIBezierPath"""
    W, H = FW / 2, FH / 2; P = lambda a, b: np.array([W - a * R, H - b * R])
    return np.vstack([np.array([[0, H]]), bez(P(1.52866483, 0), P(1.08849296, 0), P(.86840694, 0), P(.63149399, .074911)), bez(P(.63149399, .074911), P(.37282392, .16905899), P(.16905899, .37282392), P(.074911, .63149399)), bez(P(.074911, .63149399), P(0, .86840701), P(0, 1.08849293), P(0, 1.52866471)), np.array([[W, 0]])])


def sdf(f, R):
    """signed distance to the shape, flat px, negative inside"""
    c = corner(R); a, b = c[:-1], c[1:]; p = np.abs(f - [FW / 2, FH / 2]); ab = b - a; L2 = (ab ** 2).sum(1); t = np.clip(((p[:, None] - a[None]) * ab[None]).sum(-1) / np.maximum(L2, 1e-9), 0, 1); d = p[:, None] - (a[None] + t[..., None] * ab[None]); D = (d ** 2).sum(-1); i = D.argmin(1); r = np.arange(len(p)); cr = ab[i, 0] * (p[r, 1] - a[i, 1]) - ab[i, 1] * (p[r, 0] - a[i, 0]); return np.sqrt(D[r, i]) * np.where(cr > 0, -1., 1.)


def outline(R, n):
    """points round the shape, even spacing, with outward normals"""
    c = corner(R); Q = np.vstack([c * [1, -1], c[::-1][1:] * [1, 1], (c * [-1, 1])[1:], (c[::-1] * [-1, -1])[1:]]) + [FW / 2, FH / 2]; s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))]; u = np.linspace(0, s[-1], n, endpoint=False); P = np.c_[np.interp(u, s, Q[:, 0]), np.interp(u, s, Q[:, 1])]; e = 2.; A = np.c_[np.interp(u + e, s, Q[:, 0]), np.interp(u + e, s, Q[:, 1])]; B = np.c_[np.interp((u - e) % s[-1], s, Q[:, 0]), np.interp((u - e) % s[-1], s, Q[:, 1])]; T = A - B; T /= np.linalg.norm(T, axis=1, keepdims=True); return P, np.c_[T[:, 1], -T[:, 0]]


def half(p, i, inn, raw=False):
    """the log found the edge at i; it is where brightness is half way between the two sides, in linear light"""
    lin = p if raw else ((np.exp(p) - 4).clip(0, 255) / 255) ** 2.2; st = inn[1] - inn[0]; a = lambda x, y: np.median(lin[max(0, i + int(x / st)):i + int(y / st) + 1]); o, b = a(-7, -3), a(3, 7); m = (o + b) / 2; w = int(3 / st); sg = np.sign(b - o)
    for j in range(max(i - w, 1), min(i + w, len(p) - 1)):
        if sg * (lin[j] - m) < 0 <= sg * (lin[j + 1] - m): return inn[j] + st * (m - lin[j]) / (lin[j + 1] - lin[j])
    return inn[i]


def trace(q, R, k):
    Hi = homog(BOX, q); P, N = outline(R, 720); to = lambda p: bend(apply(Hi, p), k); A, B = to(P), to(P + N); nor = B - A; nor /= np.linalg.norm(nor, axis=1, keepdims=True); tan = np.c_[-nor[:, 1], nor[:, 0]]
    os_ = np.arange(-OUTR, 22, .25); ts = np.arange(-5, 6); rise, drop = [], []
    for a, n, u in zip(A, nor, tan):
        T, O = np.meshgrid(ts, -os_); p = np.median(cv2.remap(LOG, (a[0] + u[0] * T + n[0] * O).astype(np.float32), (a[1] + u[1] * T + n[1] * O).astype(np.float32), cv2.INTER_LINEAR), axis=1); inn = os_
        g = np.gradient(cv2.GaussianBlur(p.reshape(-1, 1), (0, 0), 3.).ravel(), .25); i = int(np.argmin(g)); aft = np.array([np.median(p[j + 6:j + 21]) if j + 21 <= len(p) else 9. for j in range(len(p))]); gb = np.where(aft < BLACK, g, 0); lim = min(-.08, .3 * gb.min()); mins = np.where((gb < lim) & (gb <= np.roll(gb, 1)) & (gb <= np.roll(gb, -1)))[0]   # the first fall into black from outside. Jeans into a grey case is a fall too, but black doesn't follow it; a home bar or a notch's edge lies further in and can be stronger
        if len(mins): drop.append((a - n * half(p, int(mins[0]), inn)).tolist())
        I = np.cumsum(np.r_[0, p]); lo, hi = 6, 24; st = np.full(len(p), -9.); ou = np.full(len(p), 9.)
        for j in range(hi, len(p) - hi): st[j] = (I[j + hi + 1] - I[j + lo]) / (hi + 1 - lo) - (I[j - lo + 1] - I[j - hi]) / (hi + 1 - lo); ou[j] = (I[j - lo + 1] - I[j - hi]) / (hi + 1 - lo)
        ok = (inn > inn[i] + 7) & (ou < BLACK) & (st > .2) & (inn < 16) & (inn > -16); done = False
        if ok.any():
            pk = st[ok].max(); c = np.where(ok & (st >= pk * .6) & (st >= np.roll(st, 1)) & (st >= np.roll(st, -1)))[0]
            if len(c): rise.append((a - n * half(p, int(c[-1]), inn)).tolist()); done = True
        if not done and USEC:   # dark to dark: outer is neutral black, inner is the screen's colour, averaged over a longer stretch
            T2, O2 = np.meshgrid(np.arange(-12, 13), -os_); pc = np.median(cv2.remap(CHROMA, (a[0] + u[0] * T2 + n[0] * O2).astype(np.float32), (a[1] + u[1] * T2 + n[1] * O2).astype(np.float32), cv2.INTER_LINEAR), axis=1); Ic = np.cumsum(np.r_[0, pc]); sc = np.full(len(pc), -9.)
            for j in range(hi, len(pc) - hi): sc[j] = (Ic[j + hi + 1] - Ic[j + lo]) / (hi + 1 - lo) - (Ic[j - lo + 1] - Ic[j - hi]) / (hi + 1 - lo)
            okc = (ou < BLACK) & (sc > CHROMA_STEP) & (np.abs(inn) < 16)
            if okc.any():
                j = np.where(okc)[0][np.argmax(sc[okc])]; rise.append((a - n * half(pc, int(j), inn, raw=True)).tolist())
    return np.array(rise), np.array(drop)


def rect_grid(q, k, gx, gy, img):
    p = bend(apply(homog(BOX, q), np.c_[gx.ravel(), gy.ravel()]), k).reshape(gx.shape + (2,)).astype(np.float32); return cv2.remap(img, p[..., 0], p[..., 1], cv2.INTER_LINEAR)


def ui_lines(q, R, k):
    """long straight edges the old screen shows, along its axes. Each is a list of photo points"""
    Hi = homog(BOX, q); out_ = []; step = .5; SL = 12
    for axis in "hv":
        al = np.arange(16, (FW if axis == "h" else FH) - 16, 1.); ac = np.arange(16, (FH if axis == "h" else FW) - 16, step)
        gx, gy = np.meshgrid(al, ac) if axis == "h" else np.meshgrid(ac, al); g = rect_grid(q, k, gx, gy, LOG)
        if axis == "v": g = g.T
        gg = np.gradient(cv2.GaussianBlur(g, (0, 0), sigmaX=.1, sigmaY=2.), step, axis=0); sl = np.array_split(np.arange(len(al)), max(1, len(al) // SL)); M = np.stack([np.median(gg[:, s], axis=1) for s in sl], 1); cand = []
        for y in range(4, len(ac) - 4):
            for sg in (1, -1):
                on = sg * M[y] > .05; best = (0, 0); st_ = None
                for c, v in enumerate(np.r_[on, False]):
                    if v and st_ is None: st_ = c
                    if not v and st_ is not None:
                        if c - st_ > best[0]: best = (c - st_, st_)
                        st_ = None
                if best[0] * SL >= (150 if axis == "h" else 180): cand.append((best[0] * float(np.abs(M[y, best[1]:best[1] + best[0]]).mean()), y, sg, best[1], best[0]))
        cand.sort(reverse=True); used = []
        for sc, y, sg, c0, n in cand:
            if any(abs(y - u) < 8 / step for u in used): continue
            used.append(y); ys = []; xs = []
            for c in range(c0, c0 + n):
                p = np.median(g[:, sl[c]], axis=1); win = np.arange(max(y - 12, 0), min(y + 13, len(ac))); j = win[np.argmax(sg * np.gradient(cv2.GaussianBlur(p, (0, 0), sigmaX=.1, sigmaY=2.).ravel())[win])]; ys.append(half(p, int(j), ac)); xs.append(al[sl[c]].mean())
            ys, xs = np.array(ys), np.array(xs); a = np.polyfit(xs, ys, 1); r = ys - np.polyval(a, xs); keep = np.abs(r) < 1.
            if keep.sum() < 6 or r[keep].std() > .5 or abs(a[0]) > .012: continue
            fp = np.c_[xs[keep], ys[keep]] if axis == "h" else np.c_[ys[keep], xs[keep]]; out_.append(dict(axis=axis, at=round(float(np.median(ys)), 1), n=int(keep.sum()), len=float(np.ptp(xs[keep])), P=bend(apply(Hi, fp), k)))
            if len(used) >= 8: break
    return out_


def seams(q, R, k):
    """the thin bright line on the frame, each side: straight, outside the screen. It need not be parallel: it isn't on the display's plane"""
    Hi = homog(BOX, q); o = []
    for nm, ax, at, sgn in (("top", "h", 0, -1), ("right", "v", FW, 1), ("bottom", "h", FH, 1), ("left", "v", 0, -1)):
        al = np.arange(R * 1.6, (FW if ax == "h" else FH) - R * 1.6, 8.); ac = np.arange(4, 19, .25); xs, ys = [], []
        for x in al:
            t, n = np.meshgrid(np.arange(x - 4, x + 4), ac); fx, fy = (t, at + sgn * n) if ax == "h" else (at + sgn * n, t); p = np.median(rect_grid(q, k, fx, fy, gray), axis=1); p = cv2.GaussianBlur(p.reshape(-1, 1), (0, 0), 1.).ravel(); j = int(np.argmax(p))
            if p[j] - np.median(p) < 25 or j < 2 or j > len(p) - 3: continue
            a_, b_, c_ = p[j - 1], p[j], p[j + 1]; ys.append(ac[j] + .25 * .5 * (a_ - c_) / (a_ - 2 * b_ + c_ + 1e-6)); xs.append(x)
        if len(xs) < 8: continue
        xs, ys = np.array(xs), np.array(ys); m = np.median(ys); keep = np.abs(ys - m) < 3.5
        for _ in range(3): a = np.polyfit(xs[keep], ys[keep], 1); keep = np.abs(ys - np.polyval(a, xs)) < 1.5
        if keep.sum() < 20: continue
        fp = np.c_[xs[keep], at + sgn * ys[keep]] if ax == "h" else np.c_[at + sgn * ys[keep], xs[keep]]; o.append(dict(side=nm, axis=ax, n=int(keep.sum()), P=bend(apply(Hi, fp), k)))
    return o


def fit(P, lines, seam, q, R, k0, usek=True, wt=1., anchors=()):
    """FREEC: the bend's centre free too. Tried on 895d3ee8: the fit got a little closer but the cover came out less square and the island off centre, so it stays fixed"""
    global C; c0 = C.copy()
    def un(x): return q + x[:8].reshape(4, 2), R + x[8], (k0 + x[9] * .01 if usek else 0.)
    def res(x):
        global C; C = c0 + (x[10:12] * 40 if usek and FREEC else 0); q_, r_, k_ = un(x); H = homog(q_, BOX); out_ = [sdf(apply(H, und(P, k_)), r_)]
        for l in lines: f = apply(H, und(l["P"], k_))[:, 1 if l["axis"] == "h" else 0]; out_.append(wt * (f - f.mean()))
        for l in seam:
            f = apply(H, und(l["P"], k_)); a_, b_ = (f[:, 0], f[:, 1]) if l["axis"] == "h" else (f[:, 1], f[:, 0]); out_.append(wt * (b_ - np.polyval(np.polyfit(a_, b_, 1), a_)))
        for l in anchors: f = apply(H, und(l["P"], k_))[:, 1]; out_.append(wt * (f - l["at"]))
        return np.concatenate(out_)
    s = least_squares(res, np.zeros(12), loss="soft_l1", f_scale=.5, max_nfev=40); q_, r_, k_ = un(s.x); C = c0 + (s.x[10:12] * 40 if usek and FREEC else 0); return q_, r_, k_


def side_inliers(P, q, R, k, tol=1.5):
    """a side is straight: its points must lie on one line. Where a hand changes what is traced (the case's outer edge instead of its inner), the bigger straight run wins"""
    f = apply(homog(q, BOX), und(P, k)); keep = np.zeros(len(P), bool)
    for m, al, ac in (((f[:, 1] < FH / 2) & (np.abs(f[:, 0] - FW / 2) < FW / 2 - R), f[:, 0], f[:, 1]), ((f[:, 1] > FH / 2) & (np.abs(f[:, 0] - FW / 2) < FW / 2 - R), f[:, 0], f[:, 1]), ((f[:, 0] < FW / 2) & (np.abs(f[:, 1] - FH / 2) < FH / 2 - R), f[:, 1], f[:, 0]), ((f[:, 0] > FW / 2) & (np.abs(f[:, 1] - FH / 2) < FH / 2 - R), f[:, 1], f[:, 0])):
        i = np.where(m & (np.abs(sdf(f, R)) < 60))[0]
        if len(i) < 6: continue
        best = None; a, b = al[i], ac[i]
        for u in range(0, len(i), max(1, len(i) // 30)):
            for v in range(u + 1, len(i), max(1, len(i) // 30)):
                if abs(a[u] - a[v]) < 80: continue
                sl = (b[v] - b[u]) / (a[v] - a[u])
                if abs(sl) > .06: continue
                w = np.abs(b - (b[u] + sl * (a - a[u]))) < tol
                if best is None or w.sum() > best.sum(): best = w
        if best is not None: keep[i[best]] = True
    return keep


def homebar(q, R, k):
    """the home bar's lower edge lies 8 pt above the screen's bottom, on every such iPhone: where a dark page melts into the frame this says where the bottom is"""
    Hi = homog(BOX, q); xs, ys = [], []; ac = np.arange(FH - 70, FH + 25, .25)
    for x in np.arange(FW / 2 - 60, FW / 2 + 60, 6.):
        t, n = np.meshgrid(np.arange(x - 3, x + 3), ac); p = np.median(rect_grid(q, k, t, n, LOG), axis=1); g = np.gradient(cv2.GaussianBlur(p.reshape(-1, 1), (0, 0), 2.).ravel(), .25); i = int(np.argmax(np.exp(p))); 
        if np.exp(p[i]) - 4 < 90: continue
        j = i + int(np.argmin(g[i:i + 40])); ys.append(half(p, j, ac) if j - 28 > 0 and j + 28 < len(p) else ac[j]); xs.append(x)
    if len(xs) < 8: return None
    ys, xs = np.array(ys), np.array(xs); m = np.abs(ys - np.median(ys)) < 2; return dict(at=FH - 8 * PT, n=int(m.sum()), P=bend(apply(Hi, np.c_[xs[m], ys[m]]), k))


def expand(q, w): return apply(homog(BOX, q), np.array([[-w, -w], [FW + w, -w], [FW + w, FH + w], [-w, FH + w]]))
def miss(P, q, R, k): return sdf(apply(homog(q, BOX), und(P, k)), R)


def band(q, R, k):
    """from a case's edge inward, each side as a whole: where neutral black turns into the screen. Brightness or colour, whichever shows"""
    o_ = {}
    for nm, ax, at, sgn, ln in (("t", "h", 0, 1, FW), ("r", "v", FW, -1, FH), ("b", "h", FH, -1, FW), ("l", "v", 0, 1, FH)):
        ac = np.arange(8, 64, .5); pcs = np.r_[np.arange(R * 1.3, 126, 2.), np.arange(466, ln - R * 1.3, 2.)] if nm == "t" else np.arange(R * 1.3, ln - R * 1.3, 2.); t, n = np.meshgrid(pcs, ac); fx, fy = (t, at + sgn * n) if ax == "h" else (at + sgn * n, t); pg = np.median(rect_grid(q, k, fx, fy, LOG), axis=1); pc = np.median(rect_grid(q, k, fx, fy, CHROMA), axis=1); st_ = lambda v: np.array([np.median(v[min(j + 3, len(v) - 1):j + 13]) - np.median(v[max(j - 12, 0):max(j - 2, 1)]) for j in range(len(v))]); sg, sc = st_(pg), st_(pc); sn = np.maximum(sg / .2, sc / 3); up = np.where(sn > 1)[0]   # walking in, the first step: the icons and the page's content lie further in
        o_[nm] = None
        if len(up): e = up[0]; run_ = up[(up >= e) & (up < e + 24)]; o_[nm] = float(ac[run_[np.argmax(sn[run_])]])
    return o_


def rim_out(q, R, k):
    """from a screen already found, outward, each side as a whole: where the black ends in the case's inner edge. Walking from the screen can't be fooled by folds of jeans beyond the case, which a search from outside locks onto (a841f108). A thin seam has black on both sides, so it is no step"""
    o_ = {}; ac = np.arange(4, 60, .5)
    for nm, ax, at, sgn, ln in (("top", "h", 0, -1, FW), ("right", "v", FW, 1, FH), ("bottom", "h", FH, 1, FW), ("left", "v", 0, -1, FH)):
        t, n = np.meshgrid(np.arange(R * 1.3, ln - R * 1.3, 2.), ac); fx, fy = (t, at + sgn * n) if ax == "h" else (at + sgn * n, t); p = np.median(rect_grid(q, k, fx, fy, LOG), axis=1); st_ = np.array([np.median(p[min(j + 4, len(p) - 1):j + 25]) - np.median(p[max(j - 12, 0):max(j - 3, 1)]) for j in range(len(p))]); up = np.where(st_ > .5)[0]; o_[nm] = None
        if len(up): e = up[0]; run_ = up[(up >= e) & (up < e + 16)]; o_[nm] = float(half(p, int(run_[np.argmax(st_[run_])]), ac))
    known = [v for v in o_.values() if v is not None]; m_ = float(np.median(known)) if known else 20.; o_ = {a: (m_ if v is None else v) for a, v in o_.items()}
    # the black frame is one width all round, that's hardware. A glare or a case lip can hide part of one side, and a thin seam can stop the step early, so the sides are read again as visible black and all get the median
    vis_ = []
    for nm, ax, at, sgn, ln in (("top", "h", 0, -1, FW), ("right", "v", FW, 1, FH), ("bottom", "h", FH, 1, FW), ("left", "v", 0, -1, FH)):
        t, n = np.meshgrid(np.arange(R * 1.3, ln - R * 1.3, 2.), ac); fx, fy = (t, at + sgn * n) if ax == "h" else (at + sgn * n, t); p = np.median(rect_grid(q, k, fx, fy, LOG), axis=1); e = np.where(p > np.log(70 + 4))[0]
        if len(e) and o_[nm] - 1 <= ac[e[0]] <= o_[nm] + 16: vis_.append(float(ac[e[0]]))
    w_ = float(np.median(vis_)) if len(vis_) >= 3 else float(np.median(list(o_.values()))); print("frame: visible black", [round(v, 1) for v in vis_], "all sides", round(w_, 1)); return {a: w_ for a in o_}


def run(kind, q, R, usek=True, usec=False, anchored=False):
    global OUTR, USEC; k = 0.; lines = []; seam = []; anchors = []; USEC = usec; OUTR = 90. if kind == "drop" else 45.   # the case can lie far from a rough guess that sat on a notch
    for tol, withlines in ((10, 0), (5, 0), (3, 1), (2, 1), (1.5, 1), (1.2, 1)):
        rs, dr = trace(q, R, k); P = rs if kind == "rise" else dr
        if len(P) < 150: return None
        d = miss(P, q, R, k)
        if kind == "drop" and tol == 10:   # the rough guess may be off by a different amount on each side: each side's own shift first
            f_ = apply(homog(q, BOX), und(P, k)); o_ = {}
            for nm, m_, v_ in (("t", (f_[:, 0] > 120) & (f_[:, 0] < FW - 120) & (f_[:, 1] < FH / 2), -f_[:, 1]), ("b", (f_[:, 0] > 120) & (f_[:, 0] < FW - 120) & (f_[:, 1] > FH / 2), f_[:, 1] - FH), ("l", (f_[:, 1] > 200) & (f_[:, 1] < FH - 200) & (f_[:, 0] < FW / 2), -f_[:, 0]), ("r", (f_[:, 1] > 200) & (f_[:, 1] < FH - 200) & (f_[:, 0] > FW / 2), f_[:, 0] - FW)): o_[nm] = float(np.median(v_[m_])) if m_.sum() > 5 else 0.
            q = apply(homog(BOX, q), np.array([[-o_["l"], -o_["t"]], [FW + o_["r"], -o_["t"]], [FW + o_["r"], FH + o_["b"]], [-o_["l"], FH + o_["b"]]])); d = miss(P, q, R, k)
        Q = P[np.abs(d) < tol]
        if kind == "drop" and tol >= 3: Q = P[side_inliers(P, q, R, k)]
        if anchored: f_ = apply(homog(q, BOX), und(Q, k)); Q = Q[~((f_[:, 1] > FH - 60) & (np.abs(f_[:, 0] - FW / 2) < 67 * PT + 12))]   # the home bar's own edge is no screen edge
        if len(Q) < 150: return None
        if withlines: lines = ui_lines(q, R, k); seam = seams(q, R, k) if kind == "rise" else []; bar_ = homebar(q, R, k) if anchored else None; anchors = [bar_] if bar_ else []
        q, R, k = fit(Q, lines, seam, q, R, k, usek=usek and bool(withlines), anchors=anchors)
    return dict(kind=kind, q=np.round(q, 2).tolist(), R=round(float(R), 2), k=float(k), Q=Q, lines=lines, seam=seam, rs=rs if kind == "rise" else dr, anchors=anchors, dark=usec)


if MODE == "offset":
    f_ = sys.argv[4]; St = json.load(open(f_)); t, r_, b, l = [float(v) for v in out.split(",")]; St["q"] = np.round(apply(homog(BOX, np.array(St["q"], float)), np.array([[-l, -t], [FW + r_, -t], [FW + r_, FH + b], [-l, FH + b]])), 2).tolist(); json.dump(St, open(f_, "w"), indent=1); print(St); sys.exit()
if MODE == "strips":
    St = json.load(open(sys.argv[4])); C = np.array(St.get("c", C)); Hi = homog(BOX, np.array(St["q"], float)); rows = []; Z = 5; W = 1200
    for nm, a, b in (("top", (0, 0), (FW, 0)), ("right", (FW, 0), (FW, FH)), ("bottom", (FW, FH), (0, FH)), ("left", (0, FH), (0, 0))):
        a, b = np.array(a, float), np.array(b, float); L = np.linalg.norm(b - a); u = (b - a) / L; n = np.array([-u[1], u[0]]); ts = np.linspace(-L * .05, L * 1.05, W); ns = np.arange(-24, 16, 1 / Z); T, N = np.meshgrid(ts, ns); pb = bend(apply(Hi, np.c_[(a[0] + u[0] * T + n[0] * N).ravel(), (a[1] + u[1] * T + n[1] * N).ravel()]), St["k"]).reshape(T.shape + (2,)).astype(np.float32); s_ = cv2.remap(im, pb[..., 0], pb[..., 1], cv2.INTER_CUBIC); l_ = cv2.remap(LOG, pb[..., 0], pb[..., 1], cv2.INTER_CUBIC)
        lab = cv2.cvtColor(s_, cv2.COLOR_BGR2LAB).astype(np.float32); lab[:, :, 0] = np.clip((l_ - np.log(6)) / (np.log(150) - np.log(6)) * 255, 0, 255); s_ = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR); s_[int(24 * Z)] = (0, 0, 255)
        for t_ in range(-20, 16, 5): s_[int((24 + t_) * Z), :10] = (0, 255, 255); s_[int((24 + t_) * Z), -10:] = (0, 255, 255)
        cv2.putText(s_, nm, (14, 18), 0, .55, (255, 255, 255), 1, cv2.LINE_AA); rows += [s_, np.full((5, W, 3), 255, np.uint8)]
    cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 93]); sys.exit()
if __name__ == "__main__" and not globals().get("NOFIT"):
    r = run("rise", q0, 80.)
    if r is None or len(r["Q"]) < 400:   # dark to dark. The case or the band is what shows: straighten by it, then step in to where black turns into the screen, and trace again with colour
        c_ = run("drop", q0, 80.); qc, Rc, kc = np.array(c_["q"]), c_["R"], c_["k"]; o_ = band(qc, Rc, kc); print("case", c_["q"], "R", Rc, "points", len(c_["Q"]), "band", o_); bar_ = homebar(qc, Rc, kc)
        if bar_: o_["b"] = FH - (float(np.median(apply(homog(qc, BOX), und(bar_["P"], kc))[:, 1])) + 8 * PT); print("bottom from the home bar", round(o_["b"], 1))
        known = [v for v in o_.values() if v is not None]; m_ = float(np.median(known)); o_ = {a: (m_ if v is None else v) for a, v in o_.items()}; print("start", o_)
        qs = apply(homog(BOX, qc), np.array([[o_["l"], o_["t"]], [FW - o_["r"], o_["t"]], [FW - o_["r"], FH - o_["b"]], [o_["l"], FH - o_["b"]]])); r = run("rise", qs, max(Rc - m_, 50.), usec=True, anchored=True)
    # the glass: what we draw is the screen PLUS the whole frame. It ends at the seam, or where there's none at the case: on the photo that's a contour too
    if r["dark"]: gq, gR, gk = qc, Rc, kc
    else:
        H0 = homog(np.array(r["q"]), BOX); off = {}
        for l in r["seam"]:
            f0 = apply(H0, und(l["P"], r["k"])); off[l["side"]] = float(np.median(-f0[:, 1] if l["side"] == "top" else f0[:, 1] - FH if l["side"] == "bottom" else -f0[:, 0] if l["side"] == "left" else f0[:, 0] - FW))
        if len(off) >= 3:
            m0 = float(np.median(list(off.values()))); o0 = {a: off.get(a, m0) for a in ("top", "right", "bottom", "left")}; gq = apply(homog(BOX, np.array(r["q"])), np.array([[-o0["left"], -o0["top"]], [FW + o0["right"], -o0["top"]], [FW + o0["right"], FH + o0["bottom"]], [-o0["left"], FH + o0["bottom"]]])); gR = r["R"] + m0; gk = r["k"]; print("glass: seam", {a: round(v, 1) for a, v in o0.items()})
        else: o0 = rim_out(np.array(r["q"]), r["R"], r["k"]); m0 = float(np.median(list(o0.values()))); gq = apply(homog(BOX, np.array(r["q"])), np.array([[-o0["left"], -o0["top"]], [FW + o0["right"], -o0["top"]], [FW + o0["right"], FH + o0["bottom"]], [-o0["left"], FH + o0["bottom"]]])); gR = r["R"] + m0; gk = r["k"]; print("glass: case, from the screen outward", {a: round(v, 1) for a, v in o0.items()})
    gO, _ = outline(gR, 600); gP = und(bend(apply(homog(BOX, gq), gO), gk), r["k"])
    d = miss(r["Q"], np.array(r["q"]), r["R"], r["k"])
    print(r["kind"], "corners", r["q"], "R", r["R"], f"= {r['R'] / FW * 393:.1f} pt of 393 wide", "k", round(r["k"], 4)); print("bend centre", C.round(1).tolist(), "edge points", len(r["Q"]), "of 720, mean miss", round(float(np.abs(d).mean()), 3))
    H = homog(np.array(r["q"]), BOX)
    for l in r["lines"]: f = apply(H, und(l["P"], r["k"]))[:, 1 if l["axis"] == "h" else 0]; print(f" line {l['axis']} at {l['at']}: {l['n']} pieces, {l['len']:.0f} long, rms {f.std():.2f}")
    print(" seams", [(l["side"], l["n"]) for l in r["seam"]], "home bar pieces", [l["n"] for l in r["anchors"]])
    if len(sys.argv) > 4: json.dump(dict(q=r["q"], k=r["k"], c=C.round(2).tolist(), R=r["R"], kind=r["kind"], dark=r["dark"], glass=gP.round(2).tolist(), dr=8.0), open(sys.argv[4], "w"), indent=1)

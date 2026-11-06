"""Where the phone stands in space, from the four screen corners alone. For a flat rectangle the third row of the homography IS the camera depth of each point (up to one scale),
whatever the lens and wherever the crop was: so which corner is nearest is plain arithmetic. The lens is found by asking for a rectangle of the screen's known shape.
  python3 tools/slide_pose.py <quads.json> [<out.json>]"""
import sys, json, glob, os
import cv2, numpy as np
from scipy.optimize import least_squares
SW, SH = 591., 1280.; box = np.array([[0, 0], [SW, 0], [SW, SH], [0, SH]], np.float32)
NAMES = ["TL", "TR", "BR", "BL"]
def homog(q): return cv2.getPerspectiveTransform(box, np.array(q, np.float32)).astype(float)
def depths(H):
    w = np.array([H[2] @ [x, y, 1] for x, y in box]); return w / w.min()
def near(q, jitter=3, n=400, seed=1):
    """the depth slope along each axis (>0: the right side / the bottom end is farther) and how sure each is, if a corner may be off by `jitter` px. An axis below 95% has no answer."""
    q = np.array(q, float); rng = np.random.default_rng(seed)
    def slopes(q_):
        d = depths(homog(q_)); return ((d[1] + d[2]) - (d[0] + d[3])) / 2, ((d[3] + d[2]) - (d[0] + d[1])) / 2
    du, dv = slopes(q); S = np.array([slopes(q + rng.normal(0, jitter, (4, 2))) for _ in range(n)])
    su, sv = float(max((S[:, 0] > 0).mean(), (S[:, 0] < 0).mean())), float(max((S[:, 1] > 0).mean(), (S[:, 1] < 0).mean()))
    side = ("L" if du > 0 else "R") if su >= .95 else None; end = ("T" if dv > 0 else "B") if sv >= .95 else None
    return dict(du=float(du), dv=float(dv), sure_u=su, sure_v=sv, side=side, end=end, corner=(end + side) if side and end else None)
def principal(id_, crop):
    """the original photo's centre, in crop pixels: the crop came out of the original at some offset and scale"""
    o = cv2.imread(f"photos/{id_}.jpg")
    if o is None: return None
    s = cv2.SIFT_create(4000); a = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY); b = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY)
    ka, da = s.detectAndCompute(a, None); kb, db = s.detectAndCompute(b, None)
    m = [x for x, y in cv2.BFMatcher().knnMatch(db, da, k=2) if x.distance < .7 * y.distance]
    if len(m) < 12: return None
    A, inl = cv2.estimateAffinePartial2D(np.float32([kb[x.queryIdx].pt for x in m]), np.float32([ka[x.trainIdx].pt for x in m]), ransacReprojThreshold=4)
    if A is None: return None
    c = A @ np.array([o.shape[1] / 2, o.shape[0] / 2, 1]); return c.tolist(), float(np.hypot(*A[:, 0])), o.shape[1], o.shape[0]
def pose(q, pp):
    """a rectangle of the known shape, a pinhole lens: seven numbers (focal, 3 turns, 3 shifts) against eight corner coordinates; what's left over says how physical the quad is"""
    q = np.array(q, float); P = np.c_[box - [SW / 2, SH / 2], np.zeros(4)] / SH
    def proj(v):
        f, r, t = np.exp(v[0]), v[1:4], v[4:7]; R = cv2.Rodrigues(r)[0]; X = P @ R.T + t
        return np.c_[f * X[:, 0] / X[:, 2] + pp[0], f * X[:, 1] / X[:, 2] + pp[1]], X
    best = None
    for yaw in (-.5, -.15, .15, .5):
        for pitch in (-.5, -.15, .15, .5):
            f0 = 2500.; r0 = np.array([pitch, yaw, np.arctan2(*(q[1] - q[0])[::-1])]); c = q.mean(0) - pp; d = 1 * f0 / np.linalg.norm(q[3] - q[0])
            t0 = np.array([c[0] * d / f0, c[1] * d / f0, d]); v0 = np.r_[np.log(f0), r0, t0]
            r = least_squares(lambda v: (proj(v)[0] - q).ravel(), v0, bounds=(np.r_[np.log(300), [-3] * 3, -20, -20, .1], np.r_[np.log(30000), [3] * 3, 20, 20, 60]))
            if best is None or r.cost < best.cost: best = r
    v = best.x; R = cv2.Rodrigues(v[1:4])[0]; n = R[2]   # the screen's normal in camera space; its two tilts seen from the screen's own axes:
    # how far a screen's side edge recedes: the axis x of the screen in camera coordinates, its z part
    tilt_u = float(np.degrees(np.arcsin(np.clip(R[0, 2], -1, 1)))); tilt_v = float(np.degrees(np.arcsin(np.clip(R[1, 2], -1, 1))))
    res = np.linalg.norm(proj(v)[0] - q, axis=1); X = proj(v)[1]
    return dict(f=float(np.exp(v[0])), dist=float(v[6]), tilt_u=tilt_u, tilt_v=tilt_v, resid=res.round(1).tolist(), depth=(X[:, 2] / X[:, 2].min()).round(4).tolist())
near_fn = near

"""The phone body's four straight sides on the photo, found outward from the screen: along each side, the edges met walking outward; the farthest straight line that many of them agree on is the outline.
  from slide_body import body_lines"""
import sys, json
import cv2, numpy as np
def body_lines(crop, q):
    lab = cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32), (0, 0), 1.3); q = np.array(q, float); c = q.mean(0); opts = []
    dmax = .135 * (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2    # bezel, case and a visible wall: about 9 mm at most
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]; u = (b - a) / np.linalg.norm(b - a); n = np.array([-u[1], u[0]]); n = n if (a - c) @ n > 0 else -n
        ds = np.arange(3, dmax, .5); cand = []
        for t in np.linspace(.1, .9, 160):
            p = a + (b - a) * t; X = (p[0] + n[0] * ds).astype(np.float32).reshape(1, -1); Y = (p[1] + n[1] * ds).astype(np.float32).reshape(1, -1)
            prof = cv2.remap(lab, X, Y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[0]; g = np.linalg.norm(prof[2:] - prof[:-2], axis=1)
            for k in range(2, len(g) - 2):
                if g[k] > 5 and g[k] >= g[k - 1] and g[k] >= g[k + 1]: cand.append((t, ds[k + 1], g[k], *(p + n * ds[k + 1])))
        cand = np.array(cand); side = []
        if len(cand) >= 20:
            rng = np.random.default_rng(i); found = []
            for _ in range(4000):
                s_ = cand[rng.integers(0, len(cand), 2)]
                if abs(s_[0, 0] - s_[1, 0]) < .2: continue
                m = (s_[1, 1] - s_[0, 1]) / (s_[1, 0] - s_[0, 0]); d0 = s_[0, 1] - m * s_[0, 0]
                if abs(m) > 14: continue                                           # the body and the screen lie in parallel planes: over a whole side the gap barely changes
                r = np.abs(cand[:, 1] - (m * cand[:, 0] + d0)); found.append((len(np.unique((cand[r < 1.3, 0] * 160).astype(int))), m, d0))
            found.sort(key=lambda x: -x[0]); keep = []
            for f in found:
                if all(abs((f[2] + f[1] * .5) - (k[2] + k[1] * .5)) > 3 for k in keep): keep.append(f)
                if len(keep) > 7: break
            for k in keep:
                inl = np.abs(cand[:, 1] - (k[1] * cand[:, 0] + k[2])) < 1.3; vx, vy, x0, y0 = cv2.fitLine(cand[inl][:, 3:5].astype(np.float32), cv2.DIST_HUBER, 0, .01, .01).ravel()
                side.append(dict(p=[float(x0), float(y0)], d=[float(vx), float(vy)], support=k[0] / 160, gap=float(k[2] + k[1] * .5), gap0=float(k[2]), gap1=float(k[2] + k[1])))
        opts.append(side)
    out = [None] * 4
    for A, B in ((0, 2), (1, 3)):             # opposite sides go together: the same case on both, so about the same gap, the outer lines preferred
        best, bs = None, -1e9
        for x in opts[A]:
            for y in opts[B]:
                sc = x["support"] + y["support"] + .004 * (x["gap"] + y["gap"]) - .012 * abs(x["gap"] - y["gap"])
                if min(x["support"], y["support"]) >= .2 and sc > bs: best, bs = (x, y), sc
        if best: out[A], out[B] = best
    return out

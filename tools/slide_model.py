"""A whole phone in 3D, laid on the photo. Real millimetres, a pinhole camera. The model must meet two
things at once: the screen's four corners and the body's four straight sides on the photo (slide_body.py). The body sits behind the glass, so which side walls show says how the phone is turned, which the flat screen can't say.
  python3 tools/slide_model.py <quads.json> <out.json> <dir for pictures>"""
import sys, json
import cv2, numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, "tools/ads")
from slide_body import body_lines
from slide_pose import principal
BW, BH, T = 71.5, 146.7, 7.8; DW, DH, DR = 64.6, 139.8, 8.0
SW, SH = 591, 1280
DISP = np.array([[-DW / 2, -DH / 2, 0], [DW / 2, -DH / 2, 0], [DW / 2, DH / 2, 0], [-DW / 2, DH / 2, 0]], float)
def rrect(w, h, r, z, n=14):
    a = np.linspace(0, np.pi / 2, n); pts = []
    for cx, cy, s in ((w / 2 - r, -h / 2 + r, -np.pi / 2), (w / 2 - r, h / 2 - r, 0), (-w / 2 + r, h / 2 - r, np.pi / 2), (-w / 2 + r, -h / 2 + r, np.pi)):
        pts.append(np.c_[cx + r * np.cos(a + s), cy + r * np.sin(a + s)])
    P = np.vstack(pts); return np.c_[P, np.full(len(P), z)]
def project(X, v, pp):
    R = cv2.Rodrigues(np.asarray(v[:3], float))[0]; C = X @ R.T + v[3:6]; f = np.exp(v[6]); return np.c_[f * C[:, 0] / C[:, 2] + pp[0], f * C[:, 1] / C[:, 2] + pp[1]]
def side_points(v, t=(.15, .5, .85)):
    """two rims of the body at each side: front (with the case's lip) and back. Whichever comes out farther on the photo is the outline"""
    W, H = BW / 2 + v[7], BH / 2 + v[8]; out = []
    for z in (-.3, T + .8):      # +z runs away from the camera: the body lies behind the glass
        S = []
        for a, b in (((-W, -H), (W, -H)), ((W, -H), (W, H)), ((W, H), (-W, H)), ((-W, H), (-W, -H))):
            for k in t: S.append([a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k, z])
        out.append(np.array(S))
    return out
def residuals(x, q0, lines, pp, w_corner):
    v = x; r = [(project(DISP, v, pp) - q0).ravel() * w_corner]
    F, B = side_points(v); pF, pB = project(F, v, pp), project(B, v, pp); c = q0.mean(0)
    for i, l in enumerate(lines):
        if l is None: continue
        p, d = np.array(l["p"]), np.array(l["d"]); n = np.array([-d[1], d[0]]); n = n if (p - c) @ n > 0 else -n; w = min(1.5, l["support"] * 2)
        for k in range(3):
            a, b = pF[i * 3 + k], pB[i * 3 + k]; r.append([max((a - p) @ n, (b - p) @ n) * w])       # distance from the line, for the rim that sticks out
    r.append(np.array([(v[7] - 1.5) * .02, (v[8] - 1.5) * .02]))
    return np.concatenate([np.ravel(a) for a in r])
def fit(q0, lines, pp, L, w_corner=2.5):
    best = None
    for f in (.5, .72, 1.0, 1.44, 2.2, 3.2):
        F = f * L; K = np.array([[F, 0, pp[0]], [0, F, pp[1]], [0, 0, 1.]]); ok, rv, t = cv2.solvePnP(DISP, q0, K, None, flags=cv2.SOLVEPNP_IPPE)
        x0 = np.r_[rv.ravel(), t.ravel(), np.log(F), 1.5, 1.5]
        r = least_squares(residuals, x0, args=(q0, lines, pp, w_corner), loss="soft_l1", f_scale=2., bounds=(np.r_[-4, -4, -4, -200, -200, 30, np.log(.3 * L), 0, 0], np.r_[4, 4, 4, 200, 200, 2000, np.log(5 * L), 6, 6]))
        if best is None or r.cost < best.cost: best = r
    return best
def ratios(q):
    Lg = lambda a, b: float(np.linalg.norm(q[a] - q[b])); return Lg(3, 2) / Lg(0, 1), Lg(1, 2) / Lg(0, 3)
def line_errors(v, lines, pp, q0):
    F, B = side_points(v); pF, pB = project(F, v, pp), project(B, v, pp); c = q0.mean(0); e = []
    for i, l in enumerate(lines):
        if l is None: e.append(None); continue
        p, d = np.array(l["p"]), np.array(l["d"]); n = np.array([-d[1], d[0]]); n = n if (p - c) @ n > 0 else -n
        e.append(round(float(np.mean([abs(max((pF[i * 3 + k] - p) @ n, (pB[i * 3 + k] - p) @ n)) for k in range(3)])), 1))
    return e

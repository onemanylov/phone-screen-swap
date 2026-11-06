"""Fitting the 3D phone by hand, one phone at a time. Each row is one value of one number, each tile a corner of the phone, enlarged, with the model drawn thinly on it.
  python3 tools/slide_sheet.py <id prefix> <state.json> init                   -> starts from the screen's corners
  python3 tools/slide_sheet.py <id prefix> <state.json> show <out.jpg>         -> the whole phone and its corners at the current state
  python3 tools/slide_sheet.py <id prefix> <state.json> sweep <param> <step> <out.jpg> -> five rows: -2, -1, 0, +1, +2 steps
  python3 tools/slide_sheet.py <id prefix> <state.json> set <param> <value>    -> adds value to the param
params: x y (mm, in the phone's own plane), z (mm, towards or away, i.e. size), roll yaw pitch (degrees), lens (ratio: stronger or weaker perspective at the same size), cx cy (case, mm per side), r (body corner radius, mm), t (thickness)"""
import sys, json, glob, os
import cv2, numpy as np
sys.path.insert(0, "tools/ads")
from slide_model import rrect, DW, DH, DR, BW, BH
BASE = dict(cx=0., cy=0., r=10.5, t=7.8)
def cam(s): return np.array(s["rv"], float), np.array(s["tv"], float), s["f"]
def proj(X, s, pp):
    R = cv2.Rodrigues(np.array(s["rv"], float))[0]; C = X @ R.T + np.array(s["tv"], float); return np.c_[s["f"] * C[:, 0] / C[:, 2] + pp[0], s["f"] * C[:, 1] / C[:, 2] + pp[1]]
def apply(s, p, d):
    s = json.loads(json.dumps(s)); R = cv2.Rodrigues(np.array(s["rv"], float))[0]; t = np.array(s["tv"], float); c = np.array(s["tv"], float)
    if p in ("x", "y"): t = t + R.T[0 if p == "x" else 1] * d if False else t + (np.array([1, 0, 0.]) if p == "x" else np.array([0, 1, 0.])) @ R * d
    elif p == "z": t = t * (1 + d / t[2])
    elif p in ("roll", "yaw", "pitch"):
        ax = dict(pitch=[1, 0, 0], yaw=[0, 1, 0], roll=[0, 0, 1])[p]; D = cv2.Rodrigues(np.array(ax, float) * np.radians(d))[0]; s["rv"] = cv2.Rodrigues(D @ R)[0].ravel().tolist() if False else cv2.Rodrigues(R @ D)[0].ravel().tolist()
    elif p == "lens": s["f"] = s["f"] * (1 + d); t = t * (1 + d)
    else: s[p] = s[p] + d
    s["tv"] = t.tolist(); return s
def outline(s, pp):
    bw, bh = BW + 2 * s["cx"], BH + 2 * s["cy"]; r = s["r"] + (s["cx"] + s["cy"]) / 2; t = s["t"]
    rims = [rrect(bw - 1.2, bh - 1.2, r - .6, -.3, 20), rrect(bw, bh, r, 1.2, 20), rrect(bw, bh, r, t - .8, 20), rrect(bw - 1.2, bh - 1.2, r - .6, t + .4, 20)]   # +z runs away from the camera: the body lies behind the glass
    return cv2.convexHull(np.vstack([proj(x, s, pp) for x in rims]).astype(np.float32))[:, 0]
def tiles(crop, s, pp, q0, half=80, Z=3):
    out = []; line = outline(s, pp); disp = proj(rrect(DW, DH, DR, 0, 20), s, pp); cor = proj(np.array([[-BW / 2, -BH / 2, 0], [BW / 2, -BH / 2, 0], [BW / 2, BH / 2, 0], [-BW / 2, BH / 2, 0]], float), s, pp); P = cor
    for c in P:
        x0, y0 = int(c[0] - half), int(c[1] - half); M = np.array([[Z, 0, -x0 * Z], [0, Z, -y0 * Z]], float); t = cv2.warpAffine(crop, M, (2 * half * Z, 2 * half * Z), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        for pts, col in ((line, (255, 255, 0)), (disp, (0, 255, 0))):
            cv2.polylines(t, [np.round((pts - [x0, y0]) * Z).astype(np.int32)], True, col, 1, cv2.LINE_AA)
        out.append(t)
    return out
def load(pre):
    f = glob.glob(f"templates/crop/{pre}*")[0]; i = os.path.basename(f).split(".")[0]; c = f"templates/corners/{pre[:8]}.json"; return i, cv2.imread(f), json.load(open(c))["q"] if os.path.exists(c) else json.load(open("templates/quads.json"))[i]["q"]
if __name__ == "__main__":
    pre, sf, cmd = sys.argv[1:4]; i, crop, q0 = load(pre); q0 = np.array(q0, float)
    if cmd == "init":
        from slide_pose import principal
        pr = principal(i, crop); pp = pr[0] if pr else [540., 960.]; L = max(pr[2:]) * pr[1] if pr else 1920.; f = .72 * L; K = np.array([[f, 0, pp[0]], [0, f, pp[1]], [0, 0, 1.]])
        DISP = np.array([[-DW / 2, -DH / 2, 0], [DW / 2, -DH / 2, 0], [DW / 2, DH / 2, 0], [-DW / 2, DH / 2, 0]], float); ok, r, t = cv2.solvePnP(DISP, q0, K, None, flags=cv2.SOLVEPNP_IPPE)
        s = dict(rv=r.ravel().tolist(), tv=t.ravel().tolist(), f=f, pp=list(pp), **BASE); json.dump(s, open(sf, "w"), indent=1); print("start", s); sys.exit()
    s = json.load(open(sf)); pp = np.array(s["pp"])
    if cmd == "set":
        s = apply(s, sys.argv[4], float(sys.argv[5])); json.dump(s, open(sf, "w"), indent=1); print({k: (round(v, 2) if isinstance(v, float) else [round(x, 3) for x in v]) for k, v in s.items()})
    elif cmd == "show":
        c = tiles(crop, s, pp, q0, half=110, Z=2); vis = crop.copy(); cv2.polylines(vis, [outline(s, pp).astype(np.int32)], True, (255, 255, 0), 1, cv2.LINE_AA); cv2.polylines(vis, [proj(rrect(DW, DH, DR, 0, 20), s, pp).astype(np.int32)], True, (0, 255, 0), 1, cv2.LINE_AA)
        o = proj(rrect(DW, DH, DR, 0, 4), s, pp); ctr = o.mean(0); x0, y0 = np.maximum(0, (ctr - [430, 760]).astype(int)); big = vis[y0:y0 + 1520, x0:x0 + 860]; right = np.vstack([np.hstack(c[:2]), np.hstack(c[3:1:-1])]); big = cv2.resize(big, None, fx=880 / big.shape[0], fy=880 / big.shape[0], interpolation=cv2.INTER_AREA); cv2.imwrite(sys.argv[4], np.hstack([big, right]), [cv2.IMWRITE_JPEG_QUALITY, 92])
    elif cmd == "sweep":
        p, st = sys.argv[4], float(sys.argv[5]); rows = []
        for k in (-2, -1, 0, 1, 2):
            row = np.hstack(tiles(crop, apply(s, p, k * st), pp, q0, half=75, Z=2)); cv2.rectangle(row, (0, 0), (150, 34), (0, 0, 0), -1); cv2.putText(row, f"{p} {k * st:+g}", (6, 25), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2, cv2.LINE_AA); rows.append(row)
        cv2.imwrite(sys.argv[6], np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])

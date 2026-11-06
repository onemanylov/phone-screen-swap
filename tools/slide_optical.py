"""The picture on the phone, composed as light through a camera: in linear light, the screen is a
light source, so its whites go past 1.0 and roll off in the camera's shoulder, a little of it glows over the bezel and fingers, and additive glass lies on top. The photo around
stays the real one: the crop, aligned to the grey frame, which is only used to find the screen and what's in front of it. No 3D pose: a plane in one camera is a homography.
  python3 tools/slide_optical.py <crop.jpg> <grey.jpg> <screenshot> <out.jpg> [gain=1.3 bloom=1 glass=.05 lift=.02 seed=7]"""
import sys, os, json
import cv2, numpy as np
src = open(__file__.replace("slide_optical.py", "slide_screen.py")).read()
exec(src[:src.index("img = cv2.imread(sys.argv[1])")])

opt = dict(gain=1.3, bloom=1.0, glass=.035, lift=.02, seed=7, usecrop=1, grain=3, jpeg=84, floor=.006, persp=0, perspu=0, perspv=0, sb=0, sc=0, epov=0, ecenter=0, ex=.5, ey=.3, efov=120, egain=.4, ehdr=12, eblur=5, eperson=1, spec=0, sgain=1, bglass=0, grille=0, rim=0, whole=0, gw=22, gr=0, cut=0, cx0=0, cx1=0, cy0=0, cy1=0, cr=0, ce=0, bulge=0, dof=0, soft=-1, scale=1)
for a in sys.argv[5:]:
    k, v = a.split("="); opt[k] = float(v)
to_lin = lambda x: np.where(x <= .04045, x / 12.92, ((x + .055) / 1.055) ** 2.4)
to_srgb = lambda x: np.where(np.clip(x, 0, 1) <= .0031308, np.clip(x, 0, 1) * 12.92, 1.055 * np.clip(x, 0, 1) ** (1 / 2.4) - .055)

crop = cv2.imread(sys.argv[1]); grey = cv2.imread(sys.argv[2]); shot = cv2.imread(sys.argv[3])
if grey.shape != crop.shape: grey = cv2.resize(grey, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_AREA)
h, w = crop.shape[:2]; k = max(9, int(min(h, w) * .05) | 1)
r = biggest(mask_of(grey), k); gval = np.median(grey[r[0] > 0], axis=0); tight = mask_of(grey, gval, 16, 12); r = biggest(tight, k) or r
q = order(quad(r[0]))

# a grey frame made by an image editor may be shifted a little: align it onto the real crop, away from the screen
poly = np.zeros((h, w), np.uint8); cv2.fillConvexPoly(poly, q.astype(np.int32), 255)
away = cv2.bitwise_not(cv2.dilate(poly, np.ones((31, 31), np.uint8)))
W = np.eye(2, 3, dtype=np.float32)
try:
    s = .5; a = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), None, fx=s, fy=s); b = cv2.resize(cv2.cvtColor(grey, cv2.COLOR_BGR2GRAY), None, fx=s, fy=s); m = cv2.resize(away, None, fx=s, fy=s)
    _, W = cv2.findTransformECC(a, b, W, cv2.MOTION_AFFINE, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 150, 1e-5), m, 5)
    W[:, 2] /= s
except cv2.error as e:
    print("alignment did not converge, grey frame as is")
print("grey frame shift, px:", np.round(W[:, 2], 1).tolist(), "scale:", round(float(W[0, 0]), 4))
inv = cv2.invertAffineTransform(W)
grey_a = cv2.warpAffine(grey, inv, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
tight_a = cv2.warpAffine(tight, inv, (w, h), flags=cv2.INTER_NEAREST)
q = (inv[:, :2] @ q.T).T + inv[:, 2]

# the screen's four sides, to the real photo's edges: for each side, the strongest step across it near where the grey says
side_pts = []
def refine(q, gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3); c = q.mean(0); lines = []
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]; u = (b - a) / np.linalg.norm(b - a); n = np.array([-u[1], u[0]]); n = n if (c - a) @ n < 0 else -n
        pts = []
        for t in np.linspace(.12, .88, 90):
            p = a + (b - a) * t; best, bv = None, 0
            for d in np.arange(-7, 7.01, .5):
                x, y = p + n * d
                if 1 <= x < w - 1 and 1 <= y < h - 1:
                    v = abs(gx[int(y), int(x)] * n[0] + gy[int(y), int(x)] * n[1])
                    if v > bv: best, bv = p + n * d, v
            if best is not None and bv > 25: pts.append(best)
        side_pts.append(np.array(pts, np.float32).reshape(-1, 2))
        if len(pts) < 20: lines.append((a, u)); continue
        vx, vy, x0, y0 = cv2.fitLine(np.array(pts, dtype=np.float32), cv2.DIST_HUBER, 0, .01, .01).ravel(); lines.append((np.array([x0, y0]), np.array([vx, vy])))
    cross = lambda u, v: u[0] * v[1] - u[1] * v[0]; out = []
    for i in range(4):
        (p, r_), (p2, s2) = lines[i - 1], lines[i]; out.append(p + r_ * cross(p2 - p, s2) / cross(r_, s2))
    return np.array(out, dtype=np.float32)
if not os.environ.get("SLIDE_Q"):   # traced corners come from outside: nothing to refine
    q2 = refine(q, cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)); d = np.linalg.norm(q2 - q, axis=1); print("corners refined, px:", np.round(d, 1).tolist())
    if d.max() < 12: q = q2

# the old screen's own interface is a ruler lying on the very plane we want. Rows must come out level and columns upright; the four sides only keep it in place. Eight numbers, robust least squares, no camera model.
def fit_plane(q, gray, SW=591, SH=1280, edges=()):
    from scipy.optimize import least_squares
    box = np.array([[0, 0], [SW, 0], [SW, SH], [0, SH]], np.float32)
    H0i = np.linalg.inv(cv2.getPerspectiveTransform(box, q))
    seg = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV).detect(gray)[0]
    if seg is None: return q, 0, 0
    rows, cols = [], []
    for x1, y1, x2, y2 in seg[:, 0, :]:
        L = np.hypot(x2 - x1, y2 - y1)
        if L < 35: continue
        e = cv2.perspectiveTransform(np.array([[[x1, y1], [x2, y2]]], np.float32), H0i)[0]
        if (e[:, 0] < 14).any() or (e[:, 0] > SW - 14).any() or (e[:, 1] < 30).any() or (e[:, 1] > SH - 14).any(): continue
        dx, dy = e[1] - e[0]
        if abs(dy) < .07 * abs(dx): rows.append([x1, y1, x2, y2, L])
        elif abs(dx) < .07 * abs(dy): cols.append([x1, y1, x2, y2, L])
    if len(rows) + len(cols) < 6: return q, len(rows), len(cols)
    rows, cols = np.array(rows or np.zeros((0, 5))), np.array(cols or np.zeros((0, 5)))
    def res(v):
        A_ = []; B_ = []
        for (x_, y_), (u_, w_) in zip(box.astype(np.float64), v.reshape(4, 2)):
            A_ += [[x_, y_, 1, 0, 0, 0, -u_ * x_, -u_ * y_], [0, 0, 0, x_, y_, 1, -w_ * x_, -w_ * y_]]; B_ += [u_, w_]
        Hi = np.linalg.inv(np.append(np.linalg.solve(np.array(A_), np.array(B_)), 1).reshape(3, 3))
        def proj(A):
            P = np.c_[A, np.ones(len(A))] @ Hi.T; return P[:, :2] / P[:, 2:]
        out = []
        if len(rows):
            a, b = proj(rows[:, :2]), proj(rows[:, 2:4]); out.append((b[:, 1] - a[:, 1]) * (rows[:, 4] / 80))
        if len(cols):
            a, b = proj(cols[:, :2]), proj(cols[:, 2:4]); out.append((b[:, 0] - a[:, 0]) * (cols[:, 4] / 80))
        for i_, P_ in enumerate(edges):     # each real edge point stays on its side of the screen: lines give the angles, edges the place
            if len(P_) < 20: continue
            e_ = proj(P_); side = [e_[:, 1], e_[:, 0] - SW, e_[:, 1] - SH, e_[:, 0]][i_]; out.append(side * .25)
        out.append((v - q.ravel()) * .03)
        return np.concatenate(out)
    r = least_squares(res, q.ravel().astype(np.float64), loss="soft_l1", f_scale=1.0, diff_step=2e-3)
    return r.x.reshape(4, 2).astype(np.float32), len(rows), len(cols)
if not os.environ.get("SLIDE_Q"):
    q3, nr, nc = fit_plane(q, cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), edges=side_pts); q3, nr, nc = fit_plane(q3, cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), edges=side_pts); d = np.linalg.norm(q3 - q, axis=1); print(f"by the old screen's lines ({nr} rows, {nc} columns), corners moved, px:", np.round(d, 1).tolist())
    if d.max() < 40: q = q3
import os, json
if os.environ.get("SLIDE_Q"): q = np.array(json.loads(os.environ["SLIDE_Q"]), np.float32)   # corners from elsewhere, for a test (the 3D model's screen)
SH, SW = shot.shape[:2]; H_ = cv2.getPerspectiveTransform(np.array([[0, 0], [SW, 0], [SW, SH], [0, SH]], dtype=np.float32), q)
# what's the screen: inside the outline and grey on the grey frame; so the phone's own notch or island and any finger stay out
# the display's own outline, drawn in screen coordinates and carried by the same homography: its rounded corners can't overshoot the phone. Slightly past the old screen's last row of pixels
SS = 2; can = np.zeros((SH * SS, SW * SS), np.uint8); R = int(.115 * SW * SS); pad = int(.010 * SW * SS)
cv2.rectangle(can, (R - pad, -pad), (SW * SS - R + pad, SH * SS + pad), 255, -1); cv2.rectangle(can, (-pad, R - pad), (SW * SS + pad, SH * SS - R + pad), 255, -1)
for cx, cy in ((R, R), (SW * SS - R, R), (R, SH * SS - R), (SW * SS - R, SH * SS - R)): cv2.circle(can, (cx, cy), R + pad, 255, -1, cv2.LINE_AA)
can = cv2.resize(can, (SW, SH), interpolation=cv2.INTER_AREA)
if os.environ.get("SLIDE_R"):   # slide_trace.py: the true shape, iOS's continuous corners, nothing past the edge
    rr_ = float(os.environ["SLIDE_R"]); W_, H2_ = SW / 2, SH / 2; bz_ = lambda a, b, c, d: [((1 - t) ** 3 * np.array(a) + 3 * (1 - t) ** 2 * t * np.array(b) + 3 * (1 - t) * t ** 2 * np.array(c) + t ** 3 * np.array(d)) for t in np.linspace(0, 1, 24)]; P_ = lambda a, b: (W_ - a * rr_, H2_ - b * rr_)
    cc_ = np.array(bz_(P_(1.52866483, 0), P_(1.08849296, 0), P_(.86840694, 0), P_(.63149399, .074911)) + bz_(P_(.63149399, .074911), P_(.37282392, .16905899), P_(.16905899, .37282392), P_(.074911, .63149399)) + bz_(P_(.074911, .63149399), P_(0, .86840701), P_(0, 1.08849293), P_(0, 1.52866471)))
    poly_ = np.vstack([cc_ * [1, -1], cc_[::-1] * [1, 1], cc_ * [-1, 1], cc_[::-1] * [-1, -1]]) + [W_, H2_]; can = np.zeros((SH * 2, SW * 2), np.uint8); cv2.fillPoly(can, [np.round(poly_ * 2).astype(np.int32)], 255, cv2.LINE_AA); can = cv2.resize(can, (SW, SH), interpolation=cv2.INTER_AREA)
inside = cv2.warpPerspective(can, H_, (w, h), flags=cv2.INTER_LINEAR)
solid = biggest(tight_a, 15)[0]; solid = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
g = cv2.dilate(solid, np.ones((7, 7), np.uint8))
a = cv2.GaussianBlur(cv2.bitwise_and(inside, g).astype(np.float32) / 255, (0, 0), .9)[..., None]
# the phone's own island or notch stays, but dark: any light left in it is the old screen
Hi = np.linalg.inv(H_); pts = np.stack(np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)), -1).reshape(-1, 1, 2)
uv = cv2.perspectiveTransform(pts, Hi).reshape(h, w, 2)
isl = (uv[..., 1] > -SH * .03) & (uv[..., 1] < SH * .07) & (np.abs(uv[..., 0] - SW / 2) < SW * .27) & (inside > 127) & (a[..., 0] < .5)
keep = cv2.GaussianBlur(isl.astype(np.float32), (0, 0), 1.2)[..., None]

# the phone's own cutout, drawn in screen coordinates with the same map, so it lies in the same perspective as everything else..
# cut=1: a notch, from the top edge down to cy1; cut=2: an island, a capsule from cy0 to cy1. Sizes are measured on this photo's real screen.
cut_ = None
if opt["cut"]:
    Z_ = 4; c2 = np.zeros((SH * Z_, SW * Z_), np.uint8); x0_, x1_, y0_, y1_, r_ = [int(opt[k_] * Z_) for k_ in ("cx0", "cx1", "cy0", "cy1", "cr")]; e_ = int(opt["ce"] * Z_)
    if opt["cut"] == 1:
        cv2.rectangle(c2, (x0_, 0), (x1_, y1_ - r_), 255, -1); cv2.rectangle(c2, (x0_ + r_, 0), (x1_ - r_, y1_), 255, -1)
        for cx_ in (x0_ + r_, x1_ - r_): cv2.circle(c2, (cx_, y1_ - r_), r_, 255, -1, cv2.LINE_AA)
        if e_:                                                                                 # the small concave fillets where the notch meets the top edge
            cv2.rectangle(c2, (x0_ - e_, 0), (x0_, e_), 255, -1); cv2.rectangle(c2, (x1_, 0), (x1_ + e_, e_), 255, -1)
            for cx_ in (x0_ - e_, x1_ + e_): cv2.circle(c2, (cx_, e_), e_, 0, -1, cv2.LINE_AA)
    else:
        r_ = (y1_ - y0_) // 2; cv2.rectangle(c2, (x0_ + r_, y0_), (x1_ - r_, y1_), 255, -1)
        for cx_ in (x0_ + r_, x1_ - r_): cv2.circle(c2, (cx_, y0_ + r_), r_, 255, -1, cv2.LINE_AA)
    c2 = cv2.resize(c2, (SW, SH), interpolation=cv2.INTER_AREA); cut_ = cv2.warpPerspective(c2, H_, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE if os.environ.get("SLIDE_GLASS") else cv2.BORDER_CONSTANT).astype(np.float32) / 255; cut_ = cv2.GaussianBlur(cut_, (0, 0), .8); notch_ = cut_.copy()   # we draw above the screen's top too, so the notch goes on up: else the page shows through in a thin line along its top (seen only with the red screen)
    near_ = ((uv[..., 1] < SH * .1) & (np.abs(uv[..., 0] - SW / 2) < SW * .4) & (inside > 127)).astype(np.float32); near_ = cv2.GaussianBlur(near_, (0, 0), .9)   # around it the screen is simply the screen, whatever the grey frame says
    edge_ = cv2.GaussianBlur((inside.astype(np.float32) / 255) * (cv2.cvtColor(grey_a, cv2.COLOR_BGR2GRAY) < 90), (0, 0), .9)   # the grey frame's display may stop short of our outline: what it shows dark there is no sleeve or finger, so it's screen too
    a = (np.maximum(np.maximum(a[..., 0], near_), edge_) * (1 - cut_))[..., None]; keep = keep * 0
# whole=1: the entire glass is ours, bezel included, so there's no seam between old and new pixels: from the phone's rim inward everything is drawn, the page where the display is and pure black around it, under one glass.
# gw: how far past the display the glass goes, in screenshot pixels. A sleeve or finger stays in front: what's bright on the grey frame and isn't its grey display.
disp_ = None
if opt["whole"]:
    gw_ = opt["gw"]; R_ = int(.125 * SW + (opt["gr"] or gw_)); pts_ = []
    for cx_, cy_, st_ in ((SW + gw_ - R_, -gw_ + R_, -np.pi / 2), (SW + gw_ - R_, SH + gw_ - R_, 0), (-gw_ + R_, SH + gw_ - R_, np.pi / 2), (-gw_ + R_, -gw_ + R_, np.pi)):
        for t_ in np.linspace(0, np.pi / 2, 40): pts_.append([cx_ + R_ * np.cos(t_ + st_), cy_ + R_ * np.sin(t_ + st_)])
    P_ = cv2.perspectiveTransform(np.array(pts_, np.float32)[None], H_)[0]
    if os.environ.get("SLIDE_GLASS"): P_ = np.array(json.loads(os.environ["SLIDE_GLASS"]), np.float32)   # slide_trace.py: the glass's own traced contour, up to the seam or the case
    gl_ = np.zeros((h * 3, w * 3), np.uint8); cv2.fillPoly(gl_, [np.round(P_ * 3).astype(np.int32)], 255, cv2.LINE_AA); glass_ = cv2.resize(gl_, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    disp_ = cv2.GaussianBlur(inside.astype(np.float32) / 255 * (1 - (cut_ if cut_ is not None else 0)), (0, 0), .6)
    free_ = cv2.GaussianBlur(cv2.morphologyEx(cv2.morphologyEx((((solid > 0) | (cv2.cvtColor(grey_a, cv2.COLOR_BGR2GRAY) < 110)) * 255).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8)), cv2.MORPH_OPEN, np.ones((15, 15), np.uint8)).astype(np.float32) / 255, (0, 0), 1.)   # the thin soft line between the grey display and its bezel on a grey frame is no sleeve: closed
    if opt["rim"]:    # a case in front of the glass: the grey frame's edge is torn, so the case comes from the photo itself. Past the display, what's brighter than the bezel and connected to the outside is case; a glare spot on the bezel is not connected, so it stays glass
        L_ = cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), .7); past_ = (inside.astype(np.float32) / 255 < .5); solid_ = ((L_ > opt["rim"]) & past_).astype(np.uint8); n_, lab_ = cv2.connectedComponents(solid_, connectivity=8); edge_ = np.unique(lab_[(glass_ < .02) & (solid_ > 0)]); case_ = np.isin(lab_, edge_[edge_ > 0])
        ramp_ = np.clip((L_ - opt["rim"] + 14) / 28, 0, 1); near_ = cv2.dilate(case_.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0; free_ = 1 - np.where(near_ & past_, ramp_, 0).astype(np.float32); free_ = np.where(case_, np.minimum(free_, 1 - ramp_), free_)
    if opt["cut"] == 2:   # the thin strip of display above an island is cut off the grey frame's big area, so it read as "something in front" and the old screen stayed there (895d3ee8: a green bar). Nothing is in front of it
        x0_, x1_ = opt["cx0"] - 14, opt["cx1"] + 14; pb_ = cv2.perspectiveTransform(np.array([[[x0_, -gw_ - 3], [x1_, -gw_ - 3], [x1_, opt["cy0"] + 6], [x0_, opt["cy0"] + 6]]], np.float32), H_)[0]; m_ = np.zeros((h, w), np.uint8); cv2.fillConvexPoly(m_, np.round(pb_).astype(np.int32), 255, cv2.LINE_AA); free_ = np.maximum(free_, m_.astype(np.float32) / 255)
    sleeve_ = None
    if opt.get("sleeve"):   # a pale sleeve or anything wide over the glass, seen on a real grey frame: the rules below only know skin near the edge. Wide shapes stay whole with their fringe, thin torn edges don't count
        nf0_ = (free_ < .5).astype(np.uint8); wide_ = cv2.dilate(cv2.morphologyEx(nf0_, cv2.MORPH_OPEN, np.ones((25, 25), np.uint8)), np.ones((13, 13), np.uint8)); sleeve_ = cv2.GaussianBlur(((nf0_ > 0) & (wide_ > 0) & (glass_ > .5)).astype(np.float32), (0, 0), .8)
    if os.environ.get("SLIDE_GLASS"):   # along the traced contour the grey frame's torn edge has no say: only a finger or a sleeve well inside it does
        far_ = cv2.GaussianBlur(cv2.erode((glass_ > .5).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32), (0, 0), 1.); nearedge_ = (cv2.distanceTransform((glass_ > .5).astype(np.uint8), cv2.DIST_L2, 3) < 26) & (inside < 128); sk_ = cv2.morphologyEx(((cv2.cvtColor(grey_a, cv2.COLOR_BGR2HSV)[..., 1] > 60) & (cv2.cvtColor(grey_a, cv2.COLOR_BGR2GRAY) > 70)).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)); n2_, l2_ = cv2.connectedComponents(sk_, connectivity=8); out2_ = np.unique(l2_[(glass_ < .02) & (sk_ > 0)]); solid2_ = cv2.GaussianBlur(np.isin(l2_, out2_[out2_ > 0]).astype(np.float32), (0, 0), 1.); free_ = np.where(nearedge_, 1 - solid2_, free_)   # a finger comes from outside the glass; a hand reflected in it doesn't
        nf_ = (free_ < .5).astype(np.uint8); n3_, l3_ = cv2.connectedComponents(nf_, connectivity=8); out3_ = np.unique(l3_[(glass_ < .02) & (nf_ > 0)]); fg3_ = cv2.dilate(np.isin(l3_, out3_[out3_ > 0]).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0; free_ = np.where(fg3_, free_, 1.).astype(np.float32)   # same for all of the glass: a glare the grey frame kept (a841f108: white chips on the left) is cut off from the outside by the black frame, so nothing real stands there
    if sleeve_ is not None: free_ = np.minimum(free_, 1 - sleeve_)
    a = (cv2.GaussianBlur(glass_, (0, 0), .7) * free_)[..., None]; cut_ = None; keep = keep * 0
base = crop if opt["usecrop"] else grey_a
base0_ = base.copy()
# the bezel's dark ring may still hold the old screen's colour (a green rim on the red phone): dark pixels just past the outline lose it
ring_box = np.array([[-.02 * SW, -.02 * SW], [1.02 * SW, -.02 * SW], [1.02 * SW, SH + .02 * SW], [-.02 * SW, SH + .02 * SW]], dtype=np.float32)
ring_q = cv2.perspectiveTransform(ring_box[None], H_)[0]; outer = np.zeros((h, w), np.uint8); cv2.fillConvexPoly(outer, ring_q.astype(np.int32), 255)
ring = (outer > 0) & (inside == 0) & (cv2.cvtColor(base, cv2.COLOR_BGR2GRAY) < 115)
if opt["cut"]:    # a ring that follows the rounded outline, not a square one (dark squares at the corners), reaching right up to the screen's edge and into the cutout, where the old screen's colour bleeds
    outer = cv2.dilate(((inside > 0) * 255).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)))
    ring = ((outer > 0) & (inside < 250) & (cv2.cvtColor(base, cv2.COLOR_BGR2GRAY) < 175)) | (cut_ > .01 if cut_ is not None else False)
if ring.any():
    base = base.copy(); gr_ = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)[..., None].astype(np.float32); w_ = cv2.GaussianBlur(ring.astype(np.float32), (0, 0), 1.2)[..., None]
    base = (base * (1 - w_) + gr_ * (.45 if opt["cut"] else .22) * w_).astype(np.uint8)   # the bezel is black: nothing of the old screen's glow may stay in it
if opt["cut"]:    # pale lines just past the screen: the old screen's bright pixels outside our outline. Where the grey frame shows bezel there (dark, not a sleeve or a finger, not the phone's rim), its clean bezel replaces them
    band_ = (cv2.dilate(((inside > 0) * 255).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))) > 0) & (inside < 250) & (cv2.cvtColor(grey_a, cv2.COLOR_BGR2GRAY) < 90)
    clean_ = grey_a.astype(np.float32)
    if opt["bglass"]: g__ = cv2.cvtColor(grey_a, cv2.COLOR_BGR2GRAY).astype(np.float32)[..., None]; clean_ = clean_ * .25 + g__ * .75   # most of the bezel's blue is the old blue screen's glow: our page wouldn't cast it
    w_ = cv2.GaussianBlur(band_.astype(np.float32), (0, 0), 1.)[..., None]; base = (base * (1 - w_) + clean_ * w_).astype(np.uint8)
bg = to_lin(base.astype(np.float32) / 255) * (1 - keep * .995)
bgm_ = bg
if disp_ is not None: bg = to_lin(base0_.astype(np.float32) / 255)   # the glass is all ours: outside it the photo stays untouched; the cleaned ring only says how dark the glass is
if cut_ is not None: bg = np.minimum(bg, bg * (1 - cut_[..., None]) + .008 * cut_[..., None])   # inside the cutout: the photo, but never brighter than a black bezel
scr = to_lin(cv2.GaussianBlur(shot, (0, 0), .4 / opt["scale"]).astype(np.float32) / 255)
warp = cv2.warpPerspective(scr, H_, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
# by feel: the four corners stay, but what's inside may be pushed.
# persp: a 1D projective remap on each axis, straight lines stay straight, the near end grows, the far end shrinks; 0 = exactly the homography. The default is 0, the plain homography.
# bulge: a barrel, the middle swells. dof: the far end goes soft
if opt["persp"] or opt["perspu"] or opt["perspv"] or opt["bulge"]:
    us, vs = uv[..., 0].astype(np.float32) / SW, uv[..., 1].astype(np.float32) / SH
    if opt["bulge"]:
        cx_, cy_ = us - .5, (vs - .5) * SH / SW; r2 = (cx_ ** 2 + cy_ ** 2) / (.25 + .25 * (SH / SW) ** 2); f_ = 1 - opt["bulge"] * (1 - r2)
        us, vs = .5 + (us - .5) * f_, .5 + (vs - .5) * f_
    if opt["persp"] or opt["perspu"] or opt["perspv"]:
        # how far each corner is from the camera: the homography's third row is the depth, up to one scale, whatever the lens. Each side's depth is
        # the mean of its two corners; a side whose order flips if a corner is 3 px off has no answer and isn't pushed. With the corners pinned to the bezel a diagonal tilt can only be split per axis.
        box_ = np.array([[0, 0], [SW, 0], [SW, SH], [0, SH]], np.float32)
        def depth_(q_):
            H0_ = cv2.getPerspectiveTransform(box_, np.float32(q_)).astype(float); d_ = np.array([H0_[2] @ [x_, y_, 1] for x_, y_ in box_]); return d_ / d_.min()
        ratios = lambda d_: ((d_[0] + d_[3]) / (d_[1] + d_[2]), (d_[0] + d_[1]) / (d_[3] + d_[2]))   # left over right, top over bottom: > 1 means the right side, the bottom end, is nearer
        ru, rv = ratios(depth_(q)); rng_ = np.random.default_rng(1); S_ = np.array([ratios(depth_(q + rng_.normal(0, 3, (4, 2)))) for _ in range(200)])
        sure = lambda r_, x_: ((x_ > 1) == (r_ > 1)).mean() >= .95
        ru, rv = ru if sure(ru, S_[:, 0]) else 1., rv if sure(rv, S_[:, 1]) else 1.
        av, au = float(rv) ** (opt["persp"] + opt["perspv"]), float(ru) ** (opt["persp"] + opt["perspu"])    # the inverse of the display map: magnifying the near end means reading it from a smaller span
        mob = lambda t, a_: a_ * t / (1 + (a_ - 1) * t)
        us, vs = mob(us, au), mob(vs, av)
    warp = cv2.remap(scr, us * SW, vs * SH, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
warp = np.clip(warp, 0, 1)
if opt["dof"]:
    w_ = H_[2, 0] * uv[..., 0] + H_[2, 1] * uv[..., 1] + H_[2, 2]; ins_ = inside > 127; lo_, hi_ = np.percentile(w_[ins_], 1), np.percentile(w_[ins_], 99)
    far = np.clip((w_ - lo_) / max(hi_ - lo_, 1e-6), 0, 1).astype(np.float32)[..., None]
    warp = warp * (1 - far) + cv2.GaussianBlur(warp, (0, 0), opt["dof"] * opt["scale"]) * far
def edge_sharp(im, mask):
    g_ = cv2.cvtColor(np.clip(im, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32); gx_ = cv2.Sobel(g_, cv2.CV_32F, 1, 0, ksize=3); gy_ = cv2.Sobel(g_, cv2.CV_32F, 0, 1, ksize=3); m_ = cv2.erode(mask, np.ones((25, 25), np.uint8)) > 0
    v_ = np.hypot(gx_, gy_)[m_]; return float(np.percentile(v_, 99.7)) / max(float(np.percentile(g_[m_], 99.7) - np.percentile(g_[m_], .3)), 1.)
memo = globals().setdefault("memo", {})
if "soft" not in memo:
    target = edge_sharp(crop, inside); ws = (to_srgb(warp) * 255).astype(np.float32); best_s, best = 0., 9e9
    for sig in (0, .4, .6, .8, 1., 1.25, 1.5, 1.8, 2.1, 2.5):
        t_ = edge_sharp(cv2.GaussianBlur(ws, (0, 0), sig * opt["scale"]) if sig else ws, inside)
        if abs(t_ - target) < best: best, best_s = abs(t_ - target), sig
    memo["soft"] = (target, best_s)
target, best_s = memo["soft"]
best_s = min(best_s * .6, .9)   # the metric overshoots, and the address has to stay readable
if opt["soft"] >= 0: best_s = opt["soft"]
if opt["sb"] or opt["sc"]:     # softness as measured across this photo's real screen (slide_cues.py): the end in focus stays crisp, the other goes soft
    sm_ = np.clip(best_s + (opt["sb"] * (uv[..., 0] / SW - .5) + opt["sc"] * (uv[..., 1] / SH - .5)) * float(np.linalg.norm(q[1] - q[0])) / SW, 0, 3.9).astype(np.float32); lv_ = [0, .4, .8, 1.2, 1.6, 2.2, 3., 4.]
    st_ = [warp] + [cv2.GaussianBlur(warp, (0, 0), l_ * opt["scale"]) for l_ in lv_[1:]]; k_ = np.clip(np.searchsorted(lv_, sm_) - 1, 0, len(lv_) - 2); t_ = ((sm_ - np.take(lv_, k_)) / (np.take(lv_, k_ + 1) - np.take(lv_, k_))).astype(np.float32)[..., None]; out_ = np.zeros_like(warp)
    for n_ in range(len(lv_) - 1):
        m_ = k_ == n_; out_[m_] = (st_[n_] * (1 - t_) + st_[n_ + 1] * t_)[m_]
    warp = out_
elif best_s: warp = cv2.GaussianBlur(warp, (0, 0), max(best_s * opt["scale"], .01))
print(f"sharpness: old screen {target:.2f}, blurring ours by {best_s}")
wb = np.array([1.05, 1.01, .96], np.float32)                     # BGR, a touch cool against the room
emit = warp * wb * opt["gain"]
if disp_ is not None: emit = emit * disp_[..., None]
if opt["floor"]: emit = emit + np.array([1.2, .95, 1.], np.float32) * opt["floor"]   # a lit panel never quite reaches the bezel's black, so its edge can be seen

# glass: the plate is mostly dark (the person holding the phone), with soft bright shapes in the room's own colour. Additive, as glass is; contrast is what keeps it glossy
rng = np.random.default_rng(int(opt["seed"]))
P = np.zeros((SH, SW), np.float32); yy, xx = np.mgrid[0:SH, 0:SW].astype(np.float32)
ang = rng.uniform(-.5, .5)
d1 = (xx - SW * rng.uniform(.3, .75)) * np.cos(ang) + (yy - SH * rng.uniform(.08, .22)) * np.sin(ang); d2 = -(xx - SW * .5) * np.sin(ang) + (yy - SH * .12) * np.cos(ang)
P += ((np.abs(d1) < SW * .23) & (np.abs(d2) < SH * .12)).astype(np.float32)            # a window
P += .45 * (np.abs((xx - SW * rng.uniform(.1, .9)) * .85 + (yy - SH * .42) * .5) < SW * .035).astype(np.float32)   # a lamp's streak
head = ((xx - SW * .5) / (SW * .3)) ** 2 + ((yy - SH * .72) / (SH * .16)) ** 2 < 1; P[head] = 0; P[yy > SH * .78] *= .12  # the person
P = cv2.GaussianBlur(P, (0, 0), 42)   # no shape's edge may read as a line
plate = cv2.warpPerspective(P, H_, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
out = cv2.bitwise_not(inside); bright = np.percentile(bg[out > 0], 92, axis=0); tint = bright / max(float(bright.max()), 1e-3)
refl = plate[..., None] * tint * opt["glass"] * 1.6
lift = bg[out > 0].mean(axis=0) * opt["lift"]
light = emit + refl + lift
if opt["bglass"]:
    # the glass is one piece: an off pixel looks exactly like the bezel next to it. So what the glass adds to the screen is measured on the bezel around it and carried across, no invented glare
    m_ = (band_ & (cv2.cvtColor(base, cv2.COLOR_BGR2GRAY) < 70)).astype(np.float32); N_ = np.dstack([cv2.GaussianBlur(bgm_[..., k_] * m_, (0, 0), 90) for k_ in range(3)]); D_ = cv2.GaussianBlur(m_, (0, 0), 90)[..., None]
    map_ = N_ / np.maximum(D_, 1e-4); lim_ = np.percentile(bgm_[m_ > 0], 30, axis=0); print("glass: bezel", np.round(to_srgb(lim_) * 255, 0).tolist())
    light = emit + opt["bglass"] * np.minimum(map_, lim_)

# a real reflection: the panorama of the room around the phone (SLIDE_ENV), mirrored in the glass by the phone's 3D pose (SLIDE_POSE, slide_sheet.py), with glass's own reflectance. Only what's bright shows, the rest stays black: no veil.
# ex, ey: where in the panorama the mirror looks; efov: degrees across the panorama's width. The person holding the phone stands in the mirror as a dark shape. The reflection is out of the plane of focus, so it is soft.
if os.environ.get("SLIDE_ENV"):
    pose_ = json.loads(os.environ["SLIDE_POSE"]); Rp_ = cv2.Rodrigues(np.array(pose_["rv"], float))[0]; n_ = Rp_[:, 2].astype(np.float32); f_ = float(pose_["f"]); pp_ = np.array(pose_["pp"], np.float32)
    env_ = cv2.imread(os.environ["SLIDE_ENV"]).astype(np.float32) / 255; env_ = cv2.GaussianBlur(env_, (0, 0), 3); elin_ = to_lin(env_); elum_ = elin_.mean(axis=2, keepdims=True); boost_ = np.clip((elum_ - .35) / .6, 0, 1) ** 2; elin_ = elin_ * (1 + opt["ehdr"] * boost_)
    yy_, xx_ = np.mgrid[0:h, 0:w].astype(np.float32); d_ = np.dstack([(xx_ - pp_[0]) / f_, (yy_ - pp_[1]) / f_, np.ones_like(xx_)]); d_ /= np.linalg.norm(d_, axis=2, keepdims=True); dn_ = (d_ @ n_)[..., None]; r_ = d_ - 2 * dn_ * n_
    # the mirror looks back past the camera: straight back is the panorama's chosen point
    ang_x_ = np.degrees(np.arctan2(r_[..., 0], -r_[..., 2])); ang_y_ = np.degrees(np.arctan2(r_[..., 1], -r_[..., 2])); Hh_, Wh_ = env_.shape[:2]; pxd_ = Wh_ / opt["efov"]
    mapx_ = (opt["ex"] * Wh_ + ang_x_ * pxd_).astype(np.float32); mapy_ = (opt["ey"] * Hh_ + ang_y_ * pxd_).astype(np.float32)
    if opt["epov"]:   # the picture is the phone's own view, looking out of the glass with its top up: seen from there the phone's right is on the left
        X_, Y_ = Rp_[:, 0].astype(np.float32), Rp_[:, 1].astype(np.float32); c_ = -(r_ @ n_); ax_ = np.degrees(np.arctan2(r_ @ X_, c_)); ay_ = np.degrees(np.arctan2(r_ @ Y_, c_))
        if opt["ecenter"]:    # the picture's centre falls on the middle of the screen, however the phone is tilted
            c0_ = np.int32(np.round(q.mean(0))); ax_ = ax_ - ax_[c0_[1], c0_[0]]; ay_ = ay_ - ay_[c0_[1], c0_[0]]
        mapx_ = (opt["ex"] * Wh_ - ax_ * pxd_).astype(np.float32); mapy_ = (opt["ey"] * Hh_ + ay_ * pxd_).astype(np.float32)
    ref_ = cv2.remap(elin_, mapx_, mapy_, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if opt["eperson"]:
        hd_ = np.hypot(ang_x_, ang_y_ + 3); body_ = (np.abs(ang_x_) < 17 + np.clip(ang_y_ - 8, 0, 30) * .6) & (ang_y_ > 8); sh_ = ((hd_ < 9) | body_).astype(np.float32); ref_ = ref_ * (1 - cv2.GaussianBlur(sh_, (0, 0), 14)[..., None] * .93)
    cos_ = np.abs(dn_); F_ = .04 + .96 * (1 - cos_) ** 5
    light = light + cv2.GaussianBlur(ref_ * F_.astype(np.float32), (0, 0), opt["eblur"]) * opt["egain"]

# spec: a reflection on the glass, additive with crisp shapes and black everywhere else, so the screen stays glossy, no veil. Laid out in screen coordinates, so it lies in the phone's perspective.
if opt["spec"]:
    Hs_, Ws_ = SH + 80, SW + 80; yy_, xx_ = np.mgrid[0:Hs_, 0:Ws_].astype(np.float32); pl_ = np.zeros((Hs_, Ws_), np.float32); sp_ = int(opt["spec"])
    if sp_ == 1:      # a narrow streak, a strip light
        t_ = np.tan(np.radians(21)); d_ = (xx_ - (Ws_ * .73 + (yy_ - Hs_ * .2) * -t_)); pl_ = ((np.abs(d_) < Ws_ * .035) & (yy_ < Hs_ * .62)).astype(np.float32); pl_ *= np.clip(1.15 - yy_ / (Hs_ * .62), 0, 1)
        pl_ = cv2.GaussianBlur(pl_, (0, 0), 2.5); k_ = .055
    elif sp_ == 2:    # a window: two panes, sharp edges
        for x0_, x1_ in ((.52, .72), (.745, .95)):
            poly_ = np.array([[Ws_ * x0_, -10], [Ws_ * x1_, -10], [Ws_ * (x1_ - .1), Hs_ * .3], [Ws_ * (x0_ - .1), Hs_ * .3]], np.int32); cv2.fillConvexPoly(pl_, poly_, 1., cv2.LINE_AA)
        pl_ = cv2.GaussianBlur(pl_, (0, 0), 3.5) * np.clip(1.25 - yy_ / (Hs_ * .3), 0, 1); k_ = .05
    elif sp_ == 3:    # the car itself: the photo above the phone, mirrored, as a real glass would show it; never the phone
        top_ = max(40, int(q[:, 1].min()) - 30); rf_ = cv2.cvtColor(base0_, cv2.COLOR_BGR2GRAY)[:top_].astype(np.float32); rf_ = cv2.resize(rf_[::-1], (Ws_, int(Hs_ * .55)), interpolation=cv2.INTER_AREA); rf_ = to_lin(rf_ / 255); rf_ = np.clip(rf_ - np.percentile(rf_, 35), 0, None); rf_ = cv2.GaussianBlur(rf_, (0, 0), 6)
        pl_[:rf_.shape[0]] = rf_ * np.clip(1 - yy_[:rf_.shape[0], :] / rf_.shape[0], 0, 1) ** 1.2; pl_ /= max(float(np.percentile(pl_, 99.5)), 1e-4); k_ = .06
    else:             # a sweep across the top corner with one crisp edge, the sky through a windshield
        d_ = (xx_ - Ws_ * .55) * .5 + (Hs_ * .2 - yy_) * .866; pl_ = np.clip(cv2.GaussianBlur((d_ > 0).astype(np.float32), (0, 0), 2), 0, 1) * np.clip(1.1 - np.hypot(xx_ - Ws_, yy_) / (Hs_ * .55), 0, 1); k_ = .05
    M_ = H_.astype(float) @ np.array([[1, 0, -40], [0, 1, -40], [0, 0, 1.]]); tint_ = np.array([1.02, 1., .96], np.float32)
    light = light + cv2.warpPerspective(pl_, M_, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)[..., None] * tint_ * k_ * opt["sgain"]

# the lens: softness like the phone's own edge, a hair of colour split
b_, g_, r_ = cv2.split(light); light = cv2.merge([cv2.GaussianBlur(b_, (0, 0), .75 * opt["scale"]), cv2.GaussianBlur(g_, (0, 0), .6 * opt["scale"]), cv2.GaussianBlur(r_, (0, 0), .7 * opt["scale"])])
# the glow crosses the screen's edge: that's what stops the cut-out look. It's small here: this page is mostly black
masked = emit * a
glow = (.045 * cv2.GaussianBlur(masked, (0, 0), 6 * opt["scale"]) + .025 * cv2.GaussianBlur(masked, (0, 0), 28 * opt["scale"])) * opt["bloom"]
comp = bg * (1 - a) + light * a + glow * (1 - .5 * a)
# the camera's shoulder: only on the screen and near it, the photo keeps its pixels
def curve(v):
    A, B, C, D, E, F = .22, .30, .10, .20, .01, .30
    return (v * (A * v + C * B) + D * E) / (v * (A * v + B) + D * F) - E / F
tone = np.where(comp > .35, .35 + (curve(comp) / curve(1.6) - curve(np.float32(.35)) / curve(1.6)) * ((1 - .35) / (1 - curve(np.float32(.35)) / curve(1.6))), comp)
infl = np.clip(a + cv2.GaussianBlur(a, (0, 0), 15 * opt["scale"])[..., None] * .3, 0, 1)
res = bg * (1 - infl) + tone * infl
res = res + np.random.default_rng(3).normal(0, 1, res.shape).astype(np.float32) * np.sqrt(np.maximum(res, .001)) * .010 * a
# one grain over everything, so the screen and the room read as one exposure. First the screen is brought up to the
# photo's own noise, measured in its calmest blocks, then the whole frame gets the same fine grain, heavier in the shadows, a little in colour, slightly soft as a real sensor's is
srgb = to_srgb(res).astype(np.float32) * 255
if opt["grille"]:   # the phone's own speaker in the notch, from the photo: brighter than the notch's black; a black cutout without it reads as painted
    gm_ = notch_
    if os.environ.get("SLIDE_GLASS") and opt["cut"] == 1:   # we draw the frame too, so a speaker lying on it above the notch (a841f108) must come back from the photo as well
        pb_ = cv2.perspectiveTransform(np.array([[[opt["cx0"] + 30, -30], [opt["cx1"] - 30, -30], [opt["cx1"] - 30, 2], [opt["cx0"] + 30, 2]]], np.float32), H_)[0]; m_ = np.zeros(notch_.shape, np.uint8); cv2.fillConvexPoly(m_, np.round(pb_).astype(np.int32), 255, cv2.LINE_AA); gm_ = cv2.GaussianBlur(m_.astype(np.float32) / 255, (0, 0), 1.) if opt.get("gband") else np.maximum(notch_, cv2.GaussianBlur(m_.astype(np.float32) / 255, (0, 0), 1.))   # gband: only the speaker's band. The notch's own soft edge would bring back the old bright page as a thin line around it
    if os.environ.get("SLIDE_GLASS") and opt["cut"] == 2:   # above an island the bezel in the photo is lit, ours is black: its pixels would read as a patch and a threshold as drawn lines. So only the speaker's relief comes back: photo over its own smooth level, on top of our black (79186658)
        pb_ = cv2.perspectiveTransform(np.array([[[opt["cx0"] + 24, -19], [opt["cx1"] - 24, -19], [opt["cx1"] - 24, -6], [opt["cx0"] + 24, -6]]], np.float32), H_)[0]; m_ = np.zeros(notch_.shape, np.uint8); cv2.fillConvexPoly(m_, np.round(pb_).astype(np.int32), 255, cv2.LINE_AA); m_ = m_.astype(np.float32) / 255; g_ = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32); own_ = m_ * (g_ < 90)   # the case and the page are not the bezel's level
        lev_ = cv2.GaussianBlur(g_ * own_, (0, 0), 7) / np.maximum(cv2.GaussianBlur(own_, (0, 0), 7), 1e-3); rel_ = np.clip(g_ / np.maximum(lev_, 4), .4, 1.4); ms_ = cv2.GaussianBlur(own_, (0, 0), 1.)[..., None]; srgb = srgb * (1 + (rel_[..., None] - 1) * ms_)
    if os.environ.get("SLIDE_GLASS") and opt["cut"] == 1 and opt.get("relief"):   # the same relief for a notch: only the speaker's own shape over our black, so no glint or old page comes back as specks. The band runs from the bezel into the notch
        pb_ = cv2.perspectiveTransform(np.array([[[opt["cx0"] + 40, -22], [opt["cx1"] - 40, -22], [opt["cx1"] - 40, opt["cy1"] - 8], [opt["cx0"] + 40, opt["cy1"] - 8]]], np.float32), H_)[0]; m_ = np.zeros(notch_.shape, np.uint8); cv2.fillConvexPoly(m_, np.round(pb_).astype(np.int32), 255, cv2.LINE_AA); m_ = m_.astype(np.float32) / 255; g_ = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32); own_ = m_ * (g_ < 150)
        lev_ = cv2.GaussianBlur(g_ * own_, (0, 0), 9) / np.maximum(cv2.GaussianBlur(own_, (0, 0), 9), 1e-3); rel_ = np.clip(g_ / np.maximum(lev_, 4), .5, 1.2); ms_ = cv2.GaussianBlur(own_, (0, 0), 1.2)[..., None]; srgb = srgb * (1 + (rel_[..., None] - 1) * ms_)
    if not (os.environ.get("SLIDE_GLASS") and (opt["cut"] == 2 or opt.get("relief"))): kg_ = (np.clip((cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32) - opt["grille"]) / 40, 0, 1) * gm_)[..., None]; srgb = srgb * (1 - kg_) + crop.astype(np.float32) * kg_
def calm_noise(im, mask=None):
    g = cv2.cvtColor(np.clip(im, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32); hf = g - cv2.GaussianBlur(g, (0, 0), 1.5); v = []
    for y in range(0, h - 24, 24):
        for x in range(0, w - 24, 24):
            if mask is None or mask[y:y + 24, x:x + 24].min() > .9: v.append(hf[y:y + 24, x:x + 24].std())
    return float(np.percentile(v, 12)) if v else 0.
if "noise" not in memo: memo["noise"] = (calm_noise(base.astype(np.float32), 1 - a[..., 0]), calm_noise(srgb, a[..., 0]))
photo_n, screen_n = memo["noise"]
rg = np.random.default_rng(11)
def grainy(sig, soft=.55):
    n = cv2.GaussianBlur(rg.normal(0, 1, (h, w)).astype(np.float32), (0, 0), soft); return n / n.std() * sig
need = min(2.5, max(0., photo_n ** 2 - screen_n ** 2) ** .5)   # a textured photo (denim everywhere) would read as huge noise
L_ = cv2.cvtColor(np.clip(srgb, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
tonal = (1.25 - .65 * L_)[..., None]
srgb = srgb + grainy(need)[..., None] * a
srgb = srgb + (grainy(opt["grain"])[..., None] + np.dstack([grainy(opt["grain"] * .35, 1.1) for _ in range(3)])) * tonal
u8 = np.clip(srgb, 0, 255).astype(np.uint8)
print(f"noise: photo {photo_n:.2f}, old screen {screen_n:.2f}, added to the screen {need:.2f}, whole frame {opt['grain']}")
ok, enc = cv2.imencode(".jpg", u8, [cv2.IMWRITE_JPEG_QUALITY, int(opt["jpeg"])])
cv2.imwrite(sys.argv[4], cv2.imdecode(enc, cv2.IMREAD_COLOR), [cv2.IMWRITE_JPEG_QUALITY, 96])
print("black on screen:", round(float(np.percentile(u8[(a[..., 0] > .95)], 5)), 1), "white:", round(float(np.percentile(u8[(a[..., 0] > .95)], 99.5)), 1))

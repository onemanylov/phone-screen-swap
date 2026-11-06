"""Helpers shared by the tools: finding a flat grey screen on a frame, ordering its corners."""
import sys
import cv2, numpy as np

def mask_of(img, grey=(200, 200, 200), tol=34, spread_max=16):
    d = np.abs(img.astype(np.float32) - np.array(grey, dtype=np.float32)).max(axis=2)
    spread = img.astype(np.int16).max(axis=2) - img.astype(np.int16).min(axis=2)
    return ((d < tol) & (spread < spread_max)).astype(np.uint8) * 255

def biggest(m, k):
    # a wide opening first: a grey sleeve or a window touching the screen comes off along the thin bridge
    o = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(o)
    if n < 2: return None
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return (lab == i).astype(np.uint8) * 255, int(st[i, cv2.CC_STAT_AREA])

def quad(blob):
    """Four straight lines fitted to the outline's four sides, crossed: the true corners, past the screen's rounded ones."""
    c = max(cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
    hull = cv2.convexHull(c)[:, 0, :].astype(np.float32)
    pts = []  # the hull, sampled evenly, so that a long straight side counts for much
    for i in range(len(hull)):
        a, b = hull[i], hull[(i + 1) % len(hull)]
        pts += [a + (b - a) * t for t in np.linspace(0, 1, max(2, int(np.linalg.norm(b - a))), endpoint=False)]
    pts = np.array(pts, dtype=np.float32)
    box = cv2.boxPoints(cv2.minAreaRect(pts))
    lines = []
    for i in range(4):
        a, b = box[i], box[(i + 1) % 4]; u = (b - a) / np.linalg.norm(b - a); L = np.linalg.norm(b - a)
        t = (pts - a) @ u; dist = np.abs((pts - a) @ np.array([-u[1], u[0]]))
        near = pts[(t > L * .2) & (t < L * .8) & (dist < max(L, np.linalg.norm(box[(i + 2) % 4] - b)) * .12)]
        if len(near) < 8: return box
        vx, vy, x0, y0 = cv2.fitLine(near, cv2.DIST_HUBER, 0, .01, .01).ravel()
        lines.append((np.array([x0, y0]), np.array([vx, vy])))
    out = []
    for i in range(4):
        (p, r), (q2, s2) = lines[i - 1], lines[i]
        cross = lambda u, v: u[0] * v[1] - u[1] * v[0]
        out.append(p + r * cross(q2 - p, s2) / cross(r, s2))
    return np.array(out, dtype=np.float32)

def order(q):
    # the phone's top is the short side higher in the frame: corners go top-left, top-right, bottom-right, bottom-left, as seen with the phone upright
    q = q[np.argsort(np.arctan2(q[:, 1] - q[:, 1].mean(), q[:, 0] - q[:, 0].mean()))]  # clockwise on screen
    sides = [np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)]
    s = 0 if sides[0] + sides[2] < sides[1] + sides[3] else 1   # q[s]→q[s+1] is a short side
    a, b = (q[s] + q[(s + 1) % 4]) / 2, (q[(s + 2) % 4] + q[(s + 3) % 4]) / 2
    if a[1] > b[1]: s = (s + 2) % 4
    return np.array([q[(s + i) % 4] for i in range(4)], dtype=np.float32)

# img = cv2.imread(sys.argv[1])   (the tools exec everything above this line)

"""The crop, before anything is drawn: the phone is made big first. The crop comes from the original photo, 9:16, screen about half the width.
  python3 tools/slide_crop.py --q '<[[x,y]*4 TL,TR,BR,BL] rough, on the original>' <original.jpg> <out.jpg>   -> prints the same corners on the crop, for slide_lines.py"""
import sys
import cv2, numpy as np
src = open(__file__.replace("slide_crop.py", "slide_screen.py")).read()
exec(src[:src.index("img = cv2.imread(sys.argv[1])")])

TARGET = .48
# no extreme tilt or perspective: on 8 frames, a screen 1.34× wider at one end was too much, 21° of plain rotation was fine
MAX_PERSPECTIVE, MAX_ANGLE = 1.2, 25
import json
ROUGH = sys.argv[1] == "--q"
if ROUGH:
    q = np.array(json.loads(sys.argv[2]), float); sys.argv = [sys.argv[0], None] + sys.argv[3:]; o = cv2.imread(sys.argv[2])
else:
    g = cv2.imread(sys.argv[1]); o = cv2.imread(sys.argv[2])
    k = max(9, int(min(g.shape[:2]) * .05) | 1)
    r = biggest(mask_of(g), k)
    if r is None: print("no screen", sys.argv[1]); sys.exit(2)
    grey = np.median(g[r[0] > 0], axis=0); r = biggest(mask_of(g, grey, 16, 12), k) or r
    q = order(quad(r[0])) * (o.shape[1] / g.shape[1])
sides = [np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)]
axis = (q[3] + q[2]) - (q[0] + q[1]); angle = abs(float(np.degrees(np.arctan2(axis[0], axis[1]))))
ratio = max(max(sides[0], sides[2]) / min(sides[0], sides[2]), max(sides[1], sides[3]) / min(sides[1], sides[3]))
if ratio > MAX_PERSPECTIVE or angle > MAX_ANGLE:
    print(f"{sys.argv[2].rsplit('/', 1)[-1][:10]}: rejected, perspective {ratio:.2f}, tilt {angle:.0f}°"); sys.exit(3)
H, W = o.shape[:2]
screen = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
span = q.max(axis=0) - q.min(axis=0)
cw = screen / TARGET
cw = max(cw, span[0] * 1.12, span[1] * 1.08 * 9 / 16)   # the whole phone stays in, tilt counted
cw = min(cw, W, H * 9 / 16); ch = cw * 16 / 9
c = q.mean(axis=0)
x = float(np.clip(c[0] - cw / 2, 0, W - cw)); y = float(np.clip(c[1] - ch * .52, 0, H - ch))
crop = o[int(y):int(y + ch), int(x):int(x + cw)]
crop = cv2.resize(crop, (1080, 1920), interpolation=cv2.INTER_LANCZOS4 if crop.shape[1] < 1080 else cv2.INTER_AREA)
cv2.imwrite(sys.argv[3], crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
if ROUGH: print("corners on the crop", ((q - [int(x), int(y)]) * (1080 / (int(x + cw) - int(x)))).round(1).tolist())
print(f"{sys.argv[3].rsplit('/', 1)[-1][:10]}: screen {screen / W:.0%} of the photo -> {screen / cw:.0%} of the crop, cut {int(cw)} px wide of {W}")

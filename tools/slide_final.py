"""One frame from its recipe: the phone's pose gives the display's corners, slide_optical.py draws the glass. Anything after the out path overrides the recipe, for trying values.
  python3 tools/slide_final.py <recipe.json> <out.jpg> [key=value...]
A recipe: {frame, screenshot, env, pose, opts, corners}; corners is the state of slide_corners.py and gives the screen's corners directly, the pose then serves only the reflection. Without it the corners come through the pose. Examples: templates/corners/*.recipe.json"""
import json, os, subprocess, sys, tempfile
import cv2, numpy as np
sys.path.insert(0, "tools/ads")
from slide_sheet import proj
from slide_model import DW, DH
R = json.load(open(sys.argv[1])); out = sys.argv[2]; opts = dict(R["opts"])
for a in sys.argv[3:]:
    k, v = a.split("="); opts[k] = float(v)
pose = json.load(open(R["pose"])); D = np.array([[-DW / 2, -DH / 2, 0], [DW / 2, -DH / 2, 0], [DW / 2, DH / 2, 0], [-DW / 2, DH / 2, 0]], float)
env = dict(os.environ, SLIDE_Q=json.dumps(json.load(open(R["corners"]))["q"] if R.get("corners") else proj(D, pose, np.array(pose["pp"])).round(2).tolist()), SLIDE_POSE=json.dumps(pose))
if R.get("env"): env["SLIDE_ENV"] = R["env"]
f = R["frame"]; state = json.load(open(R["corners"])) if R.get("corners") else {}; k = state.get("k")
if k and state.get("R"): env["SLIDE_R"] = str(state["R"])
if k and state.get("glass"): env["SLIDE_GLASS"] = json.dumps(state["glass"])
crop_, grey_, out_ = f"templates/crop/{f}.jpg", f"templates/grey/{f}.jpg", out
# no grey frame: the photo with the traced screen filled flat. What the grey frame gave is covered by the traced outline and glass (5 phones rebuilt this way: no visible difference). Not covered: a finger lying over the screen
def flat_screen(im): g_ = im.copy(); cv2.fillConvexPoly(g_, np.round(np.array(json.loads(env["SLIDE_Q"]))).astype(np.int32), (200, 200, 200)); return g_
if not k and not os.path.exists(grey_): g0 = tempfile.mkdtemp() + "/grey.png"; cv2.imwrite(g0, flat_screen(cv2.imread(crop_))); grey_ = g0
if k:   # slide_trace.py: the corners live on the straightened photo. Draw there, then bend the glass back: the page then bends along with the photo and fits the edge all the way round. Outside the glass the photo stays untouched
    im_ = cv2.imread(crop_); h_, w_ = im_.shape[:2]; C_ = np.array(state.get("c", [w_ / 2., h_ / 2.])); gx, gy = np.meshgrid(np.arange(w_, dtype=float), np.arange(h_, dtype=float)); g_ = np.stack([gx, gy], -1)
    def und(p): d = p - C_; return C_ + d * (1 + k * (d ** 2).sum(-1, keepdims=True) / 1e6)
    def bend(p):
        r_ = p.copy()
        for _ in range(4): d = r_ - C_; r_ = C_ + (p - C_) / (1 + k * (d ** 2).sum(-1, keepdims=True) / 1e6)
        return r_
    tmp = tempfile.mkdtemp(); b_ = bend(g_).astype(np.float32); bm = lambda i_: cv2.remap(i_, b_[..., 0], b_[..., 1], cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
    crop_ = f"{tmp}/crop.png"; cv2.imwrite(crop_, bm(im_)); g0 = f"{tmp}/grey.png"; cv2.imwrite(g0, bm(cv2.imread(grey_)) if os.path.exists(grey_) else flat_screen(bm(im_))); grey_ = g0; out_ = f"{tmp}/out.png"
r = subprocess.run([sys.executable, "tools/slide_optical.py", crop_, grey_, R["screenshot"], out_] + [f"{k_}={v}" for k_, v in opts.items()], capture_output=True, text=True, env=env)
print(r.stdout[-400:] if r.returncode == 0 else r.stderr[-800:])
if r.returncode == 0 and k:
    u_ = und(g_).astype(np.float32); back = cv2.remap(cv2.imread(out_), u_[..., 0], u_[..., 1], cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
    # the glass: rectangle grown by gw, round; its own edge is feathered inside the drawn frame, so a little past it the drawn picture equals the photo
    w0, h0 = 591, 1280; gw = opts.get("gw", 24) + 2; Rr = .125 * w0 + gw; pts = []
    for cx, cy, st in ((w0 + gw - Rr, -gw + Rr, -np.pi / 2), (w0 + gw - Rr, h0 + gw - Rr, 0), (-gw + Rr, h0 + gw - Rr, np.pi / 2), (-gw + Rr, -gw + Rr, np.pi)):
        for t in np.linspace(0, np.pi / 2, 60): pts.append([cx + Rr * np.cos(t + st), cy + Rr * np.sin(t + st)])
    H = cv2.getPerspectiveTransform(np.float32([[0, 0], [w0, 0], [w0, h0], [0, h0]]), np.float32(state["q"])); P = bend(cv2.perspectiveTransform(np.array(pts, np.float32)[None], H)[0].astype(float))
    if state.get("glass"): g0 = np.array(state["glass"], float); c0 = g0.mean(0); P = bend(c0 + (g0 - c0) * 1.012, )   # a little past the traced glass: the drawn picture has its own soft edge inside
    m = np.zeros((h_ * 2, w_ * 2), np.uint8); cv2.fillPoly(m, [np.round(P * 2).astype(np.int32)], 255, cv2.LINE_AA); m = cv2.GaussianBlur(cv2.resize(m, (w_, h_), interpolation=cv2.INTER_AREA).astype(np.float32) / 255, (0, 0), 1.)[..., None]
    cv2.imwrite(out, np.clip(back * m + im_ * (1 - m), 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 96])
sys.exit(r.returncode)

---
name: phone-screen
description: Turn a photo of a phone into a reusable template and put any image on its screen, by code, with no image generator. Use for a new phone photo, a new template, or checking one.
argument-hint: "[path to a phone photo]"
---

# Phone screen swap

Photo: $ARGUMENTS. If there is none, ask for one, or find some with `find_photos.py`.
The whole front of the phone is replaced by code: the screen, the black frame, and the notch or island. Outside the glass the photo is untouched.

For a template that already exists, this skill isn't needed: `python3 swap.py <image>`.

## New template
Take: a real photo of an iPhone, the screen to the camera. Skip: Android, a tilt over 25°, one side over 1.2× the other (the crop refuses them). **Skip photos with a finger or sleeve over the screen:** with no grey frame, it gets painted over. All of the glass is ours, a glare on it too. Only skin coming in from outside stays from the photo.

```
P=<name, 8 characters, unique>; D=templates/corners
mkdir -p out
# 1. Look at the photo and read the screen's rough corners, TL,TR,BR,BL
python3 tools/slide_crop.py --q '<corners>' <photo.jpg> templates/crop/$P.jpg       # prints the corners on the crop
# 2. The exact straight edges. If `live` is low: R=34
python3 tools/slide_lines.py $P '<corners on the crop>' out/l.jpg                    # refined corners at the end of the output
# 3. The outline: perspective, lens bend, radius, glass contour
python3 tools/slide_trace.py $P '<corners>' out/t.jpg $D/$P.json
# 4. Proof on what the fit didn't see. Always look at the picture
python3 tools/slide_proof.py $P $D/$P.json out/$P-proof.jpg
# 5. The pose, used only by the reflection
python3 tools/slide_sheet.py $P $D/$P.pose.json init
# 6. Measure the notch or island, the status icons, the blacks
python3 tools/slide_check.py $P measure
# Recipe $D/$P.recipe.json: copy a841f108 (notch) or f5dad639 (island), then set frame=$P, pose, corners and the measured opts
# 7. Time and icons to this phone's spot
python3 tools/slide_status.py $P init <line from measure>; python3 tools/slide_status.py $P build
# 8. Build and red test
python3 tools/slide_check.py $P build
# 9. Thermal map of leaks
python3 tools/slide_check.py $P leak
```
After that it is a template: `python3 swap.py <image> -t $P`.

## Look before you trust it
1. **`out/$P-split.jpg`:** photo, red, result. The red fits, the frame ends at the case, everything from the body inward is ours.
2. **`out/$P-leak.jpg`:** the frame on a black screen, brightness on a log scale. Anything lit came from the photo: a line around a cutout, specks, glare. Only a speaker and what really stands in front may show.
3. **`out/$P-top.jpg`:** the cutout's green outline on the photo, the icons in place.
4. **`out/$P-zoom.jpg`:** corners and frame, big, beside the photo.
5. The template with a **white image** through `swap.py`: a dark page hides bad corners and edges.
6. **Frame width:** the black frame is one width all round, that is hardware. `slide_trace.py` reads the visible black on each side and gives every side the median (the `frame:` line in its output). Check on the white image that top and side match.

## Settings
- **Black:** `floor`, so the frame and cutout match the photo. Start at `.003` and lower it if the photo is darker. `build` prints both.
- **Notch:** values from `measure`. Tune `cr` by the green outline, `measure` reads it low.
- **Speaker above a notch:** `grille=45`. If a thin line of the old page shows around the notch: `gband=1`.
- **Island:** its centre is hardware, 295.5 of 591. If `measure` says more than 1 px off, something bright ate an edge: take the clean edge and mirror it through the centre. On a lock screen, take `x` from a phone with an island (time 111, icons 479) and `y` from `measure`.
- **Speaker above an island:** if it lies inside the glass contour, `grille=45` brings back only its relief.
- **Sleeve over the glass:** `sleeve=1`, only with a real grey frame in `templates/grey/` (`f38fe44a`).
- **Dark on dark:** `slide_trace.py` switches by itself when under 400 of 720 points rise. Check `-split` with extra care.

## How it works
A phone is rigid, so the screen is a rectangle with iOS's continuous corners, seen through a perspective and one radial lens bend. Edges are found in log brightness, so dark on dark works, and placed at the half-way point in linear light. The picture is drawn on the straightened photo, then bent back with the glass.

## Known traps
- Judging by a dark page only.
- Fitting one way and checking another: the check is the fit, the proof is what was left out.
- Straight lines directly on the photo: it is bent. They exist only on the straightened one.
- Searching for a case edge from outside in on a bright screen: a fold of cloth looks like a case. Go from the screen outward.

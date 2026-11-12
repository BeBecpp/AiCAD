"""Simulate: client prints the plan, draws red/blue markup by hand (incl. a new door),
then photographs it with a phone. Writes photo.jpg + ground truth for evaluating the pipeline."""
import re, sys, json, cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont

S = sys.argv[1]
rng = np.random.default_rng(7)
scan = cv2.imread(S + '/scan.png', 0)
k = 2344 / 842.04
P = lambda x, y: (x * k, (1190.52 - y) * k)

# ---------- 1. "printed" page (unrotated), then hand ink layers ----------
H0, W0 = scan.shape
red = np.zeros((H0, W0), np.uint8); blue = np.zeros((H0, W0), np.uint8)

def wobble(pts, amp=1.5):
    pts = np.asarray(pts, float)
    # resample densely, add smoothed random walk perpendicular-ish noise
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    n = max(int(d[-1] / 2), 2); t = np.linspace(0, d[-1], n)
    xy = np.c_[np.interp(t, d, pts[:, 0]), np.interp(t, d, pts[:, 1])]
    noise = np.cumsum(rng.normal(0, amp * 0.3, (n, 2)), axis=0)
    noise = cv2.GaussianBlur(noise.reshape(-1, 1, 2).astype(np.float32), (1, 31), 0).reshape(-1, 2)
    noise -= np.linspace(noise[0], noise[-1], n)  # keep endpoints
    return (xy + noise * amp).astype(np.int32)

def stroke(mask, pts, w=4, amp=1.5):
    cv2.polylines(mask, [wobble(pts, amp)], False, 255, w, cv2.LINE_AA)

# real client ink from the PDF annotations (same shapes a person drew)
t = open(S + '/objstm41.bin', 'rb').read().decode('latin1')
for chunk in t.split('/Type/Annot>>'):
    st = re.search(r'/Subtype/(\w+)', chunk)
    if not st or st.group(1) not in ('Ink', 'Circle'): continue
    col = [float(v) for v in re.search(r'/C\[([\d. ]+)\]', chunk).group(1).split()]
    m = blue if col[2] > col[0] else red
    rect = [float(v) for v in re.search(r'/Rect\[([\d. ]+)\]', chunk).group(1).split()]
    if st.group(1) == 'Circle':
        c = P((rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2); r = (rect[2] - rect[0]) / 2 * k
        a = np.linspace(0, 2.1 * np.pi, 80)
        stroke(m, np.c_[c[0] + r * np.cos(a), c[1] + r * np.sin(a)], 4, 1.0)
    else:
        il = re.search(r'/InkList\[(.*?)\]\]', chunk, re.S).group(1)
        for s in il.split(']['):
            v = [float(x) for x in re.findall(r'[\d.]+', s)]
            pts = np.array([P(v[i], v[i + 1]) for i in range(0, len(v) - 1, 2)], np.int32)
            cv2.polylines(m, [pts], False, 255, 4, cv2.LINE_AA)

# rotate page to reading orientation (PDF has /Rotate 90)
page = cv2.rotate(scan, cv2.ROTATE_90_CLOCKWISE)
red = cv2.rotate(red, cv2.ROTATE_90_CLOCKWISE); blue = cv2.rotate(blue, cv2.ROTATE_90_CLOCKWISE)

# extra synthetic hand markup in reading orientation (full-res px)
# (a) NEW DOOR in the partition wall right of the big lower-left room
hx, hy, L = 1122, 1250, 55
stroke(red, [(hx, hy), (hx + L, hy)], 5)                               # door leaf
a = np.linspace(0, np.pi / 2, 30)
stroke(red, np.c_[hx + L * np.cos(a), hy - L * np.sin(a)], 4, 0.8)     # swing arc
stroke(red, [(hx - 8, hy - L), (hx + 8, hy - L)], 4); stroke(red, [(hx - 8, hy), (hx + 8, hy)], 4)  # opening ticks
stroke(red, [(hx + 40, hy - 30), (hx + 130, hy - 120)], 3)               # leader
# (b) 4 desks (2x2) in the big room
for i in range(2):
    for j in range(2):
        x, y = 800 + i * 64, 1130 + j * 40
        stroke(red, [(x, y), (x + 62, y), (x + 62, y + 38), (x, y + 38), (x, y)], 4)
# (c) X = remove existing item
cx, cy = 1555, 925
stroke(red, [(cx - 25, cy - 25), (cx + 25, cy + 25)], 5); stroke(red, [(cx + 25, cy - 25), (cx - 25, cy + 25)], 5)

# handwritten-ish text (rotated font), red
def htext(mask, txt, xy, size, ang):
    font = ImageFont.truetype('C:/Windows/Fonts/YuGothM.ttc', size)
    im = Image.new('L', (size * len(txt) + 40, size * 2), 0)
    ImageDraw.Draw(im).text((10, 10), txt, 255, font=font)
    im = im.rotate(ang, expand=True, resample=Image.BICUBIC)
    a = np.array(im); h, w = a.shape; x, y = xy
    mask[y:y + h, x:x + w] = np.maximum(mask[y:y + h, x:x + w], a)
htext(red, 'ドア新設 W900', (1255, 1080), 34, 6)
htext(red, 'デスク4台 W1200', (760, 1040), 34, -4)
htext(red, '撤去', (1590, 860), 34, 3)

cv2.imwrite(S + '/gt_red.png', red); cv2.imwrite(S + '/gt_blue.png', blue)

# compose printed + ink (pen colors, slightly translucent)
pg = cv2.cvtColor(page, cv2.COLOR_GRAY2BGR).astype(np.float32)
for m, col in ((red, (40, 35, 205)), (blue, (190, 80, 30))):
    a = (m.astype(np.float32) / 255 * 0.9)[..., None]
    pg = pg * (1 - a) + np.array(col, np.float32) * a
cv2.imwrite(S + '/page_marked.png', pg.astype(np.uint8))

# ---------- 2. phone photo ----------
Hp, Wp = pg.shape[:2]
PW, PH = 4032, 3024
desk = np.full((PH, PW, 3), (60, 95, 140), np.float32)                 # wooden desk (BGR)
desk += cv2.GaussianBlur(rng.normal(0, 25, (PH, PW)).astype(np.float32), (0, 0), 8)[..., None]
src = np.float32([[0, 0], [Wp, 0], [Wp, Hp], [0, Hp]])
dst = np.float32([[430, 360], [3650, 250], [3820, 2760], [250, 2870]])  # tilted, perspective
M = cv2.getPerspectiveTransform(src, dst)
paper = cv2.warpPerspective(pg, M, (PW, PH), flags=cv2.INTER_LINEAR)
pmask = cv2.warpPerspective(np.ones((Hp, Wp), np.float32), M, (PW, PH))[..., None]
img = desk * (1 - pmask) + paper * pmask
# warm white balance + lighting gradient + phone shadow + vignette
yy, xx = np.mgrid[0:PH, 0:PW].astype(np.float32)
light = 0.62 + 0.38 * (xx / PW) * 0.6 + 0.25 * (1 - yy / PH)
light *= 1 - 0.28 * np.exp(-(((xx - 1300) / 500) ** 2 + ((yy - 2500) / 350) ** 2))  # hand/phone shadow
light *= 1 - 0.25 * (((xx - PW / 2) / PW) ** 2 + ((yy - PH / 2) / PH) ** 2) * 2
img = img * light[..., None] * np.array([0.88, 0.97, 1.05], np.float32)
img = cv2.GaussianBlur(img, (0, 0), 1.3) + rng.normal(0, 4, img.shape)
img = np.clip(img, 0, 255).astype(np.uint8)
cv2.imwrite(S + '/photo.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 82])
json.dump({'corners': dst.tolist(), 'page_size': [Wp, Hp]}, open(S + '/photo_gt.json', 'w'))
print('photo', img.shape, 'page', (Wp, Hp))

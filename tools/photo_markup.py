"""Case C: phone photo of a printed plan with hand-drawn red/blue markup.

  python photo_markup.py photo.jpg OUTDIR [--size 3307x2344]

1. find the sheet (largest bright quadrilateral) -> perspective-correct to a flat page
2. flatten uneven lighting/shadows (divide by estimated paper background)
3. separate red / blue ink by color -> masks
4. group nearby ink into "markup groups" -> crops for Claude to read + groups.json
Outputs: flat.png, flat_clean.png (ink removed = base plan), red.png, blue.png,
         overlay.png, groups.json, crops/g##.png
"""
import sys, os, json, cv2, numpy as np


def order(q):
    q = q.reshape(4, 2).astype(np.float32); s = q.sum(1); d = np.diff(q, axis=1).ravel()
    return np.float32([q[s.argmin()], q[d.argmin()], q[s.argmax()], q[d.argmax()]])  # tl tr br bl


def find_sheet(img):
    sc = 1000 / max(img.shape[:2]); small = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    # paper = low saturation vs the desk (robust to shadows, unlike brightness)
    sat = cv2.GaussianBlur(hsv[..., 1], (9, 9), 0)
    m = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = cv2.convexHull(max(cs, key=cv2.contourArea))
    for eps in np.linspace(0.01, 0.1, 20):
        q = cv2.approxPolyDP(c, eps * cv2.arcLength(c, True), True)
        if len(q) == 4: break
    else:
        q = cv2.boxPoints(cv2.minAreaRect(c))
    q = order(q) / sc
    # refine corners on full-res image along strong edges
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    q = cv2.cornerSubPix(g, q.reshape(-1, 1, 2), (15, 15), (-1, -1),
                         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01)).reshape(4, 2)
    return q


def flatten_light(img):
    out = np.empty_like(img)
    for i in range(3):
        ch = img[..., i]
        bg = cv2.morphologyEx(ch, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))   # erase thin dark lines
        bg = cv2.GaussianBlur(bg, (0, 0), 25)
        out[..., i] = np.clip(ch.astype(np.float32) / np.maximum(bg, 1) * 245, 0, 255).astype(np.uint8)
    return out


def ink_masks(flat):
    hsv = cv2.cvtColor(flat, cv2.COLOR_BGR2HSV); h, s, v = cv2.split(hsv)
    b, g, r = [c.astype(np.int16) for c in cv2.split(flat)]
    red = (((h < 12) | (h > 160)) & (s > 70) & (r - np.maximum(g, b) > 45)).astype(np.uint8) * 255
    blue = ((h > 95) & (h < 135) & (s > 70) & (b - np.maximum(g, r) > 35)).astype(np.uint8) * 255
    k = np.ones((2, 2), np.uint8)
    return [cv2.morphologyEx(m, cv2.MORPH_OPEN, k) for m in (red, blue)]


def groups(red, blue, gap=45, min_px=60):
    out = []
    for name, m in (('red', red), ('blue', blue)):
        n, lab, st, _ = cv2.connectedComponentsWithStats(cv2.dilate(m, np.ones((gap, gap), np.uint8)))
        for i in range(1, n):
            x, y, w, h, _ = st[i]
            ink = int((m[y:y + h, x:x + w] > 0)[lab[y:y + h, x:x + w] == i].sum())
            if ink >= min_px:
                out.append({'color': name, 'bbox': [int(x), int(y), int(w), int(h)], 'ink_px': ink})
    out.sort(key=lambda g: (g['bbox'][1] // 200, g['bbox'][0]))
    for i, g in enumerate(out): g['id'] = f'g{i:02d}'
    return out


def main(photo, outdir, size='3307x2344'):
    W, H = map(int, size.split('x'))
    os.makedirs(outdir + '/crops', exist_ok=True)
    img = cv2.imread(photo)
    q = find_sheet(img)
    M = cv2.getPerspectiveTransform(q, np.float32([[0, 0], [W, 0], [W, H], [0, H]]))
    flat = flatten_light(cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_CUBIC))
    red, blue = ink_masks(flat)
    ink = cv2.dilate(red | blue, np.ones((5, 5), np.uint8))
    clean = flat.copy(); clean[ink > 0] = 255                       # base plan without markup
    gs = groups(red, blue)
    ov = cv2.cvtColor(cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR) // 2 + 127
    ov[red > 0] = (0, 0, 230); ov[blue > 0] = (230, 60, 0)
    for g in gs:
        x, y, w, h = g['bbox']; p = 40
        cv2.rectangle(ov, (x, y), (x + w, y + h), (0, 160, 0), 3)
        cv2.putText(ov, g['id'], (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 140, 0), 3)
        cv2.imwrite(f"{outdir}/crops/{g['id']}.png", flat[max(y - p, 0):y + h + p, max(x - p, 0):x + w + p])
    for nm, im in (('flat', flat), ('flat_clean', clean), ('red', red), ('blue', blue), ('overlay', ov)):
        cv2.imwrite(f'{outdir}/{nm}.png', im)
    json.dump({'sheet_corners_photo_px': q.round(1).tolist(), 'page_size': [W, H], 'groups': gs},
              open(outdir + '/groups.json', 'w'), indent=1, ensure_ascii=False)
    print(f'sheet corners {q.round(1).tolist()}\n{len(gs)} markup groups')
    return q, red, blue


if __name__ == '__main__':
    main(*sys.argv[1:3], *(sys.argv[4:5] if '--size' in sys.argv else []))

"""Vectorize a binary plan scan into horizontal/vertical segments (scan px, reading orientation).

  python scan_vectorize.py page.png out.json [--roi x0,y0,x1,y1]

Broken/dashed strokes are JOINED into one continuous segment (gap <= GAP px).
Each segment is classified by how much of its length is inked:
  fill >= SOLID_FILL -> 'solid'  (walls, partitions)
  otherwise          -> 'dashed' (existing furniture outlines on these sheets) - output as continuous lines
Dashed runs longer than MAX_DASH_RUN are dropped (fire-alarm wiring / ceiling-zone outlines).
"""
import json, argparse, cv2, numpy as np

GAP, MIN_LEN, MAX_THICK = 14, 10, 6
SNAP = 9               # px: extend/trim an endpoint onto a perpendicular segment this close (closes corners)
SOLID_FILL, MIN_FILL, MAX_DASH_RUN = 0.93, 0.35, 240

def segs(ink, horiz):
    k = (lambda a, b: (a, b)) if horiz else (lambda a, b: (b, a))
    piece = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, k(3, 1)))
    joined = cv2.morphologyEx(piece, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, k(GAP, 1)))
    n, _, st, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    out = []
    for x, y, w, h, _ in st[1:]:
        L, T = (w, h) if horiz else (h, w)
        if L < MIN_LEN or T > MAX_THICK: continue
        sub = piece[y:y + h, x:x + w] > 0
        fill = float((sub.any(axis=0) if horiz else sub.any(axis=1)).mean())
        if fill < MIN_FILL: continue
        kind = 'solid' if fill >= SOLID_FILL else 'dashed'
        if kind == 'dashed' and L > MAX_DASH_RUN: continue
        s = [int(x), int(y + h // 2), int(x + w), int(y + h // 2)] if horiz else [int(x + w // 2), int(y), int(x + w // 2), int(y + h)]
        out.append((kind, s, round(fill, 2)))
    return out

def snap_corners(ss):
    """Extend/trim endpoints of H segments onto nearby V segments and vice versa, so outlines close."""
    H = [s for s in ss if s[1] == s[3]]; V = [s for s in ss if s[0] == s[2]]
    for h in H:
        for i in (0, 2):
            best = None
            for v in V:
                if abs(v[0] - h[i]) <= SNAP and min(v[1], v[3]) - SNAP <= h[1] <= max(v[1], v[3]) + SNAP:
                    if best is None or abs(v[0] - h[i]) < abs(best - h[i]): best = v[0]
            if best is not None: h[i] = best
    for v in V:
        for i in (1, 3):
            best = None
            for h in H:
                if abs(h[1] - v[i]) <= SNAP and min(h[0], h[2]) - SNAP <= v[0] <= max(h[0], h[2]) + SNAP:
                    if best is None or abs(h[1] - v[i]) < abs(best - v[i]): best = h[1]
            if best is not None: v[i] = best
    return H + V

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('png'); ap.add_argument('out'); ap.add_argument('--roi')
    a = ap.parse_args()
    g = cv2.imread(a.png, 0)
    ink = (g < 140).astype(np.uint8) * 255
    if a.roi:
        x0, y0, x1, y1 = map(int, a.roi.split(',')); m = np.zeros_like(ink); m[y0:y1, x0:x1] = 1; ink *= m
    res = segs(ink, True) + segs(ink, False)
    d = {'solid': [s for k, s, _ in res if k == 'solid'], 'dashed': [s for k, s, _ in res if k == 'dashed']}
    for kind in d: d[kind] = snap_corners(d[kind])
    # short solid pieces that are not attached to a long wall line are furniture (benches, cabinets), not walls
    L = lambda s: abs(s[2] - s[0]) + abs(s[3] - s[1])
    long_ = [s for s in d['solid'] if L(s) >= 80]
    def touches(s):
        for w in long_:
            x0, x1 = sorted((w[0], w[2])); y0, y1 = sorted((w[1], w[3]))
            for ex, ey in (s[:2], s[2:]):
                if x0 - 5 <= ex <= x1 + 5 and y0 - 5 <= ey <= y1 + 5: return True
        return False
    short_free = [s for s in d['solid'] if L(s) < 45 and not touches(s)]
    d['solid'] = [s for s in d['solid'] if s not in short_free]; d['dashed'] += short_free
    json.dump(d, open(a.out, 'w'))
    print(f"solid {len(d['solid'])}  dashed->joined {len(d['dashed'])}")
    pv = np.full((*g.shape, 3), 255, np.uint8)
    for s in d['dashed']: cv2.line(pv, s[:2], s[2:], (150, 150, 150), 2)
    for s in d['solid']: cv2.line(pv, s[:2], s[2:], (0, 0, 0), 2)
    cv2.imwrite(a.out.rsplit('.', 1)[0] + '_preview.png', pv)

if __name__ == '__main__':
    main()

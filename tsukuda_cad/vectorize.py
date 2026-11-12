"""Metric raster -> clean vector geometry.

Everything here works on a photo.Raster (10 mm/px typical) and returns geometry in mm.
Design rules that make the output look like a CAD drawing rather than a trace:
  * only horizontal / vertical lines are taken from line ink (plans are orthogonal);
  * line coordinates are clustered and averaged (a wall face is ONE x value, not several);
  * coordinates within GRID_SNAP of a grid line become exactly the grid coordinate;
  * ends are snapped onto perpendicular lines so corners close;
  * arcs (door swings, curved facades) are fitted as true circles, radius rounded.
"""
from dataclasses import dataclass, field
import math
import cv2
import numpy as np

from .photo import normalize_light, red_mask


@dataclass
class Seg:
    x1: float; y1: float; x2: float; y2: float
    kind: str = 'line'            # line | grid
    @property
    def horiz(self): return abs(self.y1 - self.y2) < 1e-6
    @property
    def length(self): return math.hypot(self.x2 - self.x1, self.y2 - self.y1)


@dataclass
class Arc:
    cx: float; cy: float; r: float; a0: float; a1: float      # degrees, CCW from a0 to a1


@dataclass
class Geometry:
    segs: list = field(default_factory=list)
    arcs: list = field(default_factory=list)
    columns: list = field(default_factory=list)   # (cx, cy, size)


# ----------------------------------------------------------------------------- ink
def ink_masks(raster, dark=150):
    """print ink (black lines/text) and red markup, in raster px."""
    g = normalize_light(cv2.cvtColor(raster.img, cv2.COLOR_BGR2GRAY), k=41)
    red = cv2.dilate(red_mask(raster.img), np.ones((5, 5), np.uint8))
    prn = ((g < dark) & (red == 0)).astype(np.uint8) * 255
    return prn, red, g


# ----------------------------------------------------------------------------- orthogonal lines
def _bands(comp, min_len_px, max_thick_px):
    """comp: boolean (cross, along). Rows carrying a line -> [(a0, a1, c)] in comp coordinates.
    Blurred double walls merge into one blob, so lines are split by the row profile; a band thicker
    than one line but thinner than a wall is emitted as its two faces."""
    prof = comp.sum(axis=1)
    if prof.max() == 0:
        return []
    on = prof >= 0.5 * prof.max()
    out, r = [], 0
    while r < len(on):
        if not on[r]:
            r += 1; continue
        r2 = r
        while r2 + 1 < len(on) and on[r2 + 1]: r2 += 1
        T = r2 - r + 1
        if T <= 4 * max_thick_px:                         # thicker = filled area (column, hatch)
            idx = np.nonzero(comp[r:r2 + 1].any(axis=0))[0]
            if idx[-1] - idx[0] + 1 >= min_len_px:
                for c in ([(r + r2) / 2] if T <= max_thick_px else [r + 1.0, r2 - 1.0]):
                    out.append((float(idx[0]), float(idx[-1]), float(c)))
        r = r2 + 1
    return out


def _runs(mask, horiz, min_len_px, max_thick_px, chunk_px=150):
    """Orthogonal line runs. Long components are analysed in chunks along their length so that a
    slightly tilted line (paper curl) still yields clean bands; chunks are re-joined later by clustering."""
    k = (min_len_px, 1) if horiz else (1, min_len_px)
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, k))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1) if horiz else (1, 3)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = []
    for i, (x, y, w, h, a) in enumerate(st[1:], 1):
        L = w if horiz else h
        if L < min_len_px:
            continue
        comp = (lab[y:y + h, x:x + w] == i)
        if not horiz:
            comp = comp.T                                  # rows = cross axis, cols = along axis
        along0, cross0 = (x, y) if horiz else (y, x)
        nchunk = max(1, int(round(L / chunk_px)))
        edges = np.linspace(0, L, nchunk + 1).astype(int)
        for e0, e1 in zip(edges[:-1], edges[1:]):
            for a0, a1, c in _bands(comp[:, e0:e1], min(min_len_px, e1 - e0), max_thick_px):
                out.append((a0 + e0 + along0, a1 + e0 + along0, c + cross0))
    return out          # (a0, a1, c) in px: along-axis extent, cross-axis coordinate


def _cluster(vals, tol):
    """1-D clustering: returns dict value->cluster mean (values sorted, gap > tol splits)."""
    v = sorted(set(vals)); groups, cur = [], [v[0]] if v else []
    for a in v[1:]:
        if a - cur[-1] <= tol: cur.append(a)
        else: groups.append(cur); cur = [a]
    if cur: groups.append(cur)
    m = {}
    for g in groups:
        mean = sum(g) / len(g)
        for a in g: m[a] = mean
    return m


def _merge_collinear(items, gap):
    """items: (a0, a1, c) with identical c per group -> merged extents."""
    out = []
    by_c = {}
    for a0, a1, c in items: by_c.setdefault(c, []).append((a0, a1))
    for c, iv in by_c.items():
        iv.sort(); s, e = iv[0]
        for a0, a1 in iv[1:]:
            if a0 <= e + gap: e = max(e, a1)
            else: out.append((s, e, c)); s, e = a0, a1
        out.append((s, e, c))
    return out


def ortho_lines(raster, prn, grid_x=(), grid_y=(), min_len_mm=150, max_thick_mm=80, gap_mm=60,
                cluster_mm=15, grid_snap_mm=15, round_mm=5, corner_snap_mm=60):
    mmpp = raster.mmpp
    L, T = max(3, int(min_len_mm / mmpp)), max(2, int(max_thick_mm / mmpp))
    H = _runs(prn, True, L, T); V = _runs(prn, False, L, T)
    # px -> mm
    Hm = [(raster.mm(a0, 0)[0], raster.mm(a1, 0)[0], float(raster.mm(0, c)[1])) for a0, a1, c in H]
    Vm = [(float(raster.mm(0, a1)[1]), float(raster.mm(0, a0)[1]), float(raster.mm(c, 0)[0])) for a0, a1, c in V]
    Hm = [(float(a), float(b), c) for a, b, c in Hm]

    def regularise(items, grid):
        cm = _cluster([c for *_, c in items], cluster_mm / 1.0)
        def snap(c):
            c = cm[c]
            for g in grid:
                if abs(c - g) <= grid_snap_mm: return float(g)
            return round(c / round_mm) * round_mm
        items = [(a0, a1, snap(c)) for a0, a1, c in items]
        return _merge_collinear(items, gap_mm)
    Hm = regularise(Hm, grid_y); Vm = regularise(Vm, grid_x)

    # close corners: move each end onto the nearest perpendicular line if close
    def snap_ends(items, perp):
        out = []
        for a0, a1, c in items:
            ends = []
            for a in (a0, a1):
                best = None
                for b0, b1, pc in perp:
                    if b0 - corner_snap_mm <= c <= b1 + corner_snap_mm and abs(pc - a) <= corner_snap_mm:
                        if best is None or abs(pc - a) < abs(best - a): best = pc
                ends.append(best if best is not None else round(a / round_mm) * round_mm)
            out.append((min(ends), max(ends), c))
        return out
    Hs = snap_ends(Hm, Vm); Vs = snap_ends(Vm, Hm)
    segs = [Seg(a0, c, a1, c) for a0, a1, c in Hs if a1 - a0 >= min_len_mm] + \
           [Seg(c, a0, c, a1) for a0, a1, c in Vs if a1 - a0 >= min_len_mm]
    for s in segs:
        if (s.horiz and s.y1 in grid_y) or (not s.horiz and s.x1 in grid_x): s.kind = 'grid'
    return segs


def drop_light(raster, gray, segs, max_core=85):
    """Remove segments whose ink is light along their length: pencil handwriting, not print.
    For each sample point take the darkest pixel across the line (+-2 px); print cores are ~0-40."""
    keep = []
    for s in segs:
        n = max(4, int(s.length / raster.mmpp / 3))
        xs = np.linspace(s.x1, s.x2, n); ys = np.linspace(s.y1, s.y2, n)
        c, r = raster.px(xs, ys); c = np.round(c).astype(int); r = np.round(r).astype(int)
        H, W = gray.shape
        vals = [gray[max(rr - 2, 0):min(rr + 3, H), max(cc - 2, 0):min(cc + 3, W)].min() for cc, rr in zip(c, r)]
        if np.median(vals) <= max_core:
            keep.append(s)
    return keep


def line_raster(raster, segs, width_px=3):
    m = np.zeros(raster.img.shape[:2], np.uint8)
    for s in segs:
        p1 = tuple(int(round(v)) for v in raster.px(s.x1, s.y1)); p2 = tuple(int(round(v)) for v in raster.px(s.x2, s.y2))
        cv2.line(m, p1, p2, 255, width_px)
    return m


# ----------------------------------------------------------------------------- arcs
def _fit_circle(pts):
    x, y = pts[:, 0], pts[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones(len(x))]
    b = x ** 2 + y ** 2
    (cx, cy, c), *_ = np.linalg.lstsq(A, b, rcond=None)
    r = math.sqrt(max(c + cx ** 2 + cy ** 2, 0))
    res = np.abs(np.hypot(x - cx, y - cy) - r)
    return cx, cy, r, float(np.sqrt((res ** 2).mean()))


def _robust_circle(pts, tol):
    cx, cy, r, rms = _fit_circle(pts)
    for _ in range(2):
        res = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r)
        keep = res <= max(tol, 2 * rms)
        if keep.sum() < 8: break
        pts = pts[keep]; cx, cy, r, rms = _fit_circle(pts)
    return cx, cy, r, rms, pts


def _span(angles, max_gap=25):
    """angles (deg) of points on a circle -> (a0, a1) CCW covering them, ignoring gaps < max_gap."""
    hist = np.zeros(360, bool); hist[np.floor(np.asarray(angles) % 360).astype(int) % 360] = True
    idx = np.nonzero(hist)[0]
    gaps = np.diff(np.r_[idx, idx[0] + 360]); j = int(np.argmax(gaps))
    if gaps[j] < max_gap:
        return 0.0, 360.0
    return float(idx[(j + 1) % len(idx)]), float(idx[j] + 1)


def arcs(raster, gray, red, used, r_min_mm=450, r_max_mm=6000, round_r=50, light=235):
    """Curved strokes left after removing orthogonal lines -> circular arcs (mm).
    Uses a lighter threshold than the line pass (door swings are printed thin/light)."""
    m = ((gray < light) & (red == 0)).astype(np.uint8) * 255
    m[cv2.dilate(used, np.ones((7, 7), np.uint8)) > 0] = 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    cands = []
    tol = 1.5 * raster.mmpp
    for i, (x, y, w, h, a) in enumerate(st[1:], 1):
        if max(w, h) * raster.mmpp < r_min_mm * 0.35 or a < 15:
            continue
        ys, xs = np.nonzero(lab[y:y + h, x:x + w] == i)
        X, Y = raster.mm(xs + x, ys + y)
        cx, cy, r, rms, pts = _robust_circle(np.c_[X, Y], tol)
        if not (r_min_mm <= r <= r_max_mm) or rms > tol or len(pts) < 0.6 * len(X):
            continue
        cands.append([cx, cy, r, list(np.degrees(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)))])
    # pieces of the same circle (broken by crossings) -> one arc
    merged = []
    for c in sorted(cands, key=lambda c: -len(c[3])):
        for m_ in merged:
            if math.hypot(c[0] - m_[0], c[1] - m_[1]) < 0.1 * m_[2] and abs(c[2] - m_[2]) < 0.1 * m_[2]:
                m_[3] += c[3]; break
        else:
            merged.append(c)
    out = []
    for cx, cy, r, ang in merged:
        a0, a1 = _span(ang)
        span = (a1 - a0) % 360 or 360
        if span < 30:
            continue
        if abs(span - 90) < 18:                        # door swing: exact quarter on the axes
            mid = a0 + span / 2
            a0 = round((mid - 45) / 90) * 90 if abs(((mid - 45) / 90) - round((mid - 45) / 90)) < 0.2 else mid - 45
            a1 = a0 + 90
        out.append(Arc(float(cx), float(cy), float(round(r / round_r) * round_r), a0 % 360, a1 % 360))
    return out


def hinge_filter(arcs_, segs, tol=120, spans=((70, 110), (160, 200))):
    """Keep swings whose centre (the hinge) sits on a wall/leaf line end and whose span is a quarter or half."""
    ends = [(s.x1, s.y1) for s in segs] + [(s.x2, s.y2) for s in segs]
    out = []
    for a in arcs_:
        span = (a.a1 - a.a0) % 360
        if not any(lo <= span <= hi for lo, hi in spans):
            continue
        if min(math.hypot(a.cx - x, a.cy - y) for x, y in ends) <= tol:
            out.append(a)
    return out


def door_arcs(raster, gray, red, used, r_mm=(450, 1100), light=228, min_span=70, density=0.75):
    """Door swings: Hough circle candidates validated by how much of the circumference is inked.
    Robust to the thin / faded / broken strokes that defeat per-component fitting."""
    rest = ((gray < light) & (red == 0)).astype(np.uint8) * 255
    rest[cv2.dilate(used, np.ones((5, 5), np.uint8)) > 0] = 0
    r0, r1 = int(r_mm[0] / raster.mmpp), int(r_mm[1] / raster.mmpp)
    cands = cv2.HoughCircles(cv2.GaussianBlur(rest, (5, 5), 1.2), cv2.HOUGH_GRADIENT, dp=1, minDist=8,
                             param1=60, param2=14, minRadius=r0, maxRadius=r1)
    if cands is None:
        return []
    near = cv2.dilate(rest, np.ones((3, 3), np.uint8)) > 0
    H, W = rest.shape
    th = np.radians(np.arange(360))
    scored = []
    for cx, cy, r in cands[0]:
        best = None
        for rr in (r - 1, r, r + 1):
            xs = np.round(cx + rr * np.cos(th)).astype(int); ys = np.round(cy - rr * np.sin(th)).astype(int)
            ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
            cov = np.zeros(360, bool); cov[ok] = near[ys[ok], xs[ok]]
            # longest covered sector, tolerating small gaps (crossing lines were removed)
            c2 = np.r_[cov, cov]; run = gap = 0; bestrun = (0, 0, 0)   # (len, start, hits)
            start = 0; hits = 0
            for k in range(720):
                if c2[k]:
                    if run == 0: start, hits = k, 0
                    run += 1 + gap; hits += 1; gap = 0
                    if run > bestrun[0] and run <= 360: bestrun = (run, start, hits)
                elif run:
                    gap += 1
                    if gap > 8: run = gap = 0
            L, st0, hit = bestrun
            if L >= min_span and hit / L >= density and (best is None or hit > best[3]):
                # a real swing is a thin line: rings just inside / outside it must be mostly empty
                ang = th[(np.arange(L) + st0) % 360]
                def ring(d):
                    x2 = np.round(cx + (rr + d) * np.cos(ang)).astype(int); y2 = np.round(cy - (rr + d) * np.sin(ang)).astype(int)
                    ok2 = (x2 >= 0) & (x2 < W) & (y2 >= 0) & (y2 < H)
                    return near[y2[ok2], x2[ok2]].mean() if ok2.any() else 1.0
                if ring(-8) < 0.3 and ring(8) < 0.3:
                    best = (cx, cy, rr, hit, st0, L)
        if best: scored.append(best)
    scored.sort(key=lambda b: -b[3])
    out = []
    for cx, cy, r, hit, st0, L in scored:
        if any(math.hypot(cx - o[0], cy - o[1]) < 12 and abs(r - o[2]) < 10 for o in out):
            continue
        out.append((cx, cy, r, hit, st0, L))
    arcs_ = []
    for cx, cy, r, hit, st0, L in out:
        X, Y = raster.mm(cx, cy); R_ = round(r * raster.mmpp / 50) * 50
        a0, a1 = float(st0 % 360), float((st0 + L) % 360)
        if abs(L - 90) < 20:                                    # exact quarter swing
            q = round(a0 / 90) * 90
            if abs(a0 - q) <= 20: a0, a1 = q % 360, (q + 90) % 360
        arcs_.append(Arc(float(X), float(Y), float(R_), a0, a1))
    return arcs_


def fit_arc_in_box(raster, gray, red, used, box_mm, segs=(), light=228, margin=80, iters=400, seed=0):
    """Door swings are printed dashed and faint, so global detection is unreliable. Given the swing's
    rough box (read from the sheet), fit the circle to the ink inside it with RANSAC; the hinge (centre)
    is snapped to the nearest wall/leaf end. Returns Arc or None."""
    X0, Y0, X1, Y1 = box_mm
    c0, r1 = raster.px(X0 - margin, Y0 - margin); c1, r0 = raster.px(X1 + margin, Y1 + margin)
    c0, r0, c1, r1 = int(c0), int(r0), int(c1), int(r1)
    m = ((gray < light) & (red == 0)).astype(np.uint8) * 255
    m[cv2.dilate(used, np.ones((5, 5), np.uint8)) > 0] = 0
    ys, xs = np.nonzero(m[r0:r1, c0:c1])
    if len(xs) < 12:
        return None
    X, Y = raster.mm(xs + c0, ys + r0); P = np.c_[X, Y]
    rng = np.random.default_rng(seed); tol = 1.5 * raster.mmpp; best = (0, None)
    for _ in range(iters):
        i = rng.choice(len(P), 3, replace=False)
        try:
            cx, cy, r, _ = _fit_circle(P[i])
        except Exception:
            continue
        if not (300 < r < 2000):
            continue
        inl = np.abs(np.hypot(P[:, 0] - cx, P[:, 1] - cy) - r) < tol
        if inl.sum() > best[0]:
            best = (inl.sum(), inl)
    if best[1] is None:
        return None
    cx, cy, r, _ = _fit_circle(P[best[1]])
    ends = [(s.x1, s.y1) for s in segs] + [(s.x2, s.y2) for s in segs]
    if ends:
        ex, ey = min(ends, key=lambda e: math.hypot(e[0] - cx, e[1] - cy))
        if math.hypot(ex - cx, ey - cy) < 150:
            cx, cy = ex, ey
    pts = P[best[1]]
    r = round(float(np.median(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy))) / 50) * 50
    a0, a1 = _span(np.degrees(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)))
    span = (a1 - a0) % 360
    for target in (90, 180):                       # quarter / half swings on the axes
        if abs(span - target) < 20:
            mid = a0 + span / 2
            a0 = round((mid - target / 2) / 90) * 90; a1 = a0 + target
            break
    return Arc(float(cx), float(cy), float(r), a0 % 360, a1 % 360)


# ----------------------------------------------------------------------------- columns
def columns(raster, prn, grid_x, grid_y, size_round=50, snap_mm=250):
    k = max(3, int(250 / raster.mmpp)) | 1
    solid = cv2.dilate(cv2.erode(prn, np.ones((k, k), np.uint8)), np.ones((k, k), np.uint8))
    n, _, st, cen = cv2.connectedComponentsWithStats(solid)
    out = []
    for (x, y, w, h, a), (cx, cy) in zip(st[1:], cen[1:]):
        if not (0.6 < w / h < 1.6) or a < 0.6 * w * h:
            continue
        X, Y = raster.mm(cx, cy)
        gx = min(grid_x, key=lambda g: abs(g - X)); gy = min(grid_y, key=lambda g: abs(g - Y))
        if abs(gx - X) <= snap_mm and abs(gy - Y) <= snap_mm:
            X, Y = gx, gy
        out.append((float(X), float(Y), float(max(w, h) * raster.mmpp)))
    if out:
        size = round(float(np.median([s for *_, s in out])) / size_round) * size_round
        out = [(x, y, size) for x, y, _ in out]
    return out

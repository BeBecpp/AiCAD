"""Photo -> metric plan raster.

The grid columns (filled squares at grid intersections) give exact correspondences
photo-px <-> mm once the grid spacing has been read from the dimension strings, so we fit a
homography directly to millimetres. The fit residual doubles as a check of the dimension reading
(a misread spacing shows up as a large residual).
"""
from dataclasses import dataclass
import cv2
import numpy as np

ROT = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def load_upright(path, rotation):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    return img if not ROT[rotation] else cv2.rotate(img, ROT[rotation])


def normalize_light(gray, k=31):
    """Divide by the paper background so shadows / uneven light disappear (paper -> ~255)."""
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    bg = cv2.GaussianBlur(bg, (0, 0), k / 2)
    return np.clip(gray.astype(np.float32) / np.maximum(bg, 1) * 255, 0, 255).astype(np.uint8)


def red_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    return (((h < 10) | (h > 165)) & (s > 80) & (v > 80)).astype(np.uint8) * 255


def detect_columns(norm, min_side=25, max_side=70, thr=130):
    """Filled dark squares (structural columns). Returns [(cx, cy, w, h)] in px."""
    dark = (norm < thr).astype(np.uint8) * 255
    k = max(3, min_side // 3) | 1
    solid = cv2.dilate(cv2.erode(dark, np.ones((k, k), np.uint8)), np.ones((k, k), np.uint8))
    n, _, st, cen = cv2.connectedComponentsWithStats(solid)
    out = []
    for (x, y, w, h, a), (cx, cy) in zip(st[1:], cen[1:]):
        if min_side <= w <= max_side and min_side <= h <= max_side and 0.6 < w / h < 1.6 and a > 0.7 * w * h:
            out.append((float(cx), float(cy), int(w), int(h)))
    return out


def _cluster_1d(vals, n):
    """Split sorted values into n groups at the n-1 largest gaps; returns group index per value."""
    order = np.argsort(vals); v = np.asarray(vals)[order]
    cuts = np.sort(np.argsort(np.diff(v))[::-1][:n - 1])
    idx = np.zeros(len(v), int)
    for c in cuts:
        idx[c + 1:] += 1
    out = np.empty(len(v), int); out[order] = idx
    return out


@dataclass
class GridFit:
    H: np.ndarray            # photo px (upright) -> mm  (mm: X right, Y up)
    rms_mm: float
    max_mm: float
    used: int

    def to_mm(self, pts):
        p = np.asarray(pts, np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(p, self.H).reshape(-1, 2)


def fit_grid(columns, xs_mm, ys_mm_down):
    """columns: detected centres; xs_mm: grid X positions; ys_mm_down: grid rows top->bottom (mm from top row).
    Column centres are assigned to grid lines by 1-D gap clustering (needs >=1 column per grid line)."""
    c = np.array([(x, y) for x, y, *_ in columns], np.float64)
    ci = _cluster_1d(c[:, 0], len(xs_mm)); ri = _cluster_1d(c[:, 1], len(ys_mm_down))
    top = max(ys_mm_down)
    dst = np.array([(xs_mm[i], top - ys_mm_down[j]) for i, j in zip(ci, ri)], np.float64)
    # robust: a column partly hidden under markup has a biased centre -> drop outliers (> 100 mm)
    H, inl = cv2.findHomography(c, dst, cv2.RANSAC, 100.0)
    keep = inl.ravel().astype(bool)
    H, _ = cv2.findHomography(c[keep], dst[keep], 0)
    res = cv2.perspectiveTransform(c[keep].reshape(-1, 1, 2), H).reshape(-1, 2) - dst[keep]
    d = np.hypot(*res.T)
    return GridFit(H, float(np.sqrt((d ** 2).mean())), float(d.max()), int(keep.sum()))


@dataclass
class Raster:
    """Metric raster: pixel (col,row) <-> mm (X right, Y up)."""
    img: np.ndarray
    x0: float
    y1: float              # mm Y of the top row
    mmpp: float

    def px(self, X, Y):
        return ((np.asarray(X) - self.x0) / self.mmpp, (self.y1 - np.asarray(Y)) / self.mmpp)

    def mm(self, c, r):
        return (self.x0 + np.asarray(c) * self.mmpp, self.y1 - np.asarray(r) * self.mmpp)


def rectify(img, fit: GridFit, bounds_mm, mmpp=10.0, interp=cv2.INTER_CUBIC, border=255):
    """Warp an upright photo image into a metric raster covering bounds_mm=(X0,Y0,X1,Y1)."""
    X0, Y0, X1, Y1 = bounds_mm
    W, Hh = int((X1 - X0) / mmpp), int((Y1 - Y0) / mmpp)
    A = np.array([[1 / mmpp, 0, -X0 / mmpp], [0, -1 / mmpp, Y1 / mmpp], [0, 0, 1]])   # mm -> raster px
    M = A @ fit.H
    out = cv2.warpPerspective(img, M, (W, Hh), flags=interp, borderValue=(border,) * 3 if img.ndim == 3 else border)
    return Raster(out, X0, Y1, mmpp)


# ----------------------------------------------------------------------------- local distortion
def _thin_line_offset(profile, max_w=5, dark=110):
    """Index of a thin dark line in a 1-D intensity profile (None if absent or part of a wide band)."""
    i = int(np.argmin(profile))
    if profile[i] > dark:
        return None
    lo = i; hi = i
    while lo > 0 and profile[lo - 1] <= dark: lo -= 1
    while hi < len(profile) - 1 and profile[hi + 1] <= dark: hi += 1
    if hi - lo + 1 > max_w:
        return None
    w = np.maximum(dark - profile[lo:hi + 1].astype(float), 0) + 1e-6      # sub-pixel centroid
    return float((np.arange(lo, hi + 1) * w).sum() / w.sum())


def _poly_terms(X, Y, deg):
    return np.stack([X ** i * Y ** j for i in range(deg + 1) for j in range(deg + 1 - i)], axis=-1)


def refine_with_grid(R: Raster, grid_x, grid_y, step_mm=200, search_mm=80, deg=2):
    """Paper curl / lens distortion leave the homography a few cm off locally. Measure where each
    printed grid line actually is (hundreds of samples along the thin parts) and remove the smooth
    residual displacement with a 2-D polynomial. Returns (corrected Raster, rms before, rms after)."""
    g = normalize_light(cv2.cvtColor(R.img, cv2.COLOR_BGR2GRAY), k=41)
    H, W = g.shape; s = int(search_mm / R.mmpp)
    ys0 = np.arange(min(grid_y) - 1000, max(grid_y) + 1000, step_mm)
    xs0 = np.arange(min(grid_x) - 1000, max(grid_x) + 1000, step_mm)
    sx, sy, sd = [], [], []                 # samples for dx (from vertical grid lines)
    tx, ty, td = [], [], []                 # samples for dy (from horizontal grid lines)
    for X in grid_x:
        c = int(round(R.px(X, 0)[0]))
        for Y in ys0:
            r = int(round(R.px(0, Y)[1]))
            if not (0 <= r < H and s <= c < W - s): continue
            o = _thin_line_offset(g[r, c - s:c + s + 1])
            if o is not None: sx.append(X); sy.append(Y); sd.append((o - s) * R.mmpp)
    for Y in grid_y:
        r = int(round(R.px(0, Y)[1]))
        for X in xs0:
            c = int(round(R.px(X, 0)[0]))
            if not (s <= r < H - s and 0 <= c < W): continue
            o = _thin_line_offset(g[r - s:r + s + 1, c])
            if o is not None: tx.append(X); ty.append(Y); td.append(-(o - s) * R.mmpp)   # rows grow downward
    def fit(X, Y, D):
        X, Y, D = map(np.asarray, (X, Y, D))
        sc = 10000.0
        A = _poly_terms(X / sc, Y / sc, deg)
        coef, *_ = np.linalg.lstsq(A, D, rcond=None)
        for _ in range(2):                                       # drop outliers (hit a door/text)
            res = A @ coef - D; keep = np.abs(res) < max(25, 2.5 * np.std(res))
            coef, *_ = np.linalg.lstsq(A[keep], D[keep], rcond=None)
        res = A @ coef - D
        return coef, float(np.sqrt((D ** 2).mean())), float(np.sqrt((res[np.abs(res) < 60] ** 2).mean()))
    cx, bx, ax = fit(sx, sy, sd); cy, by, ay = fit(tx, ty, td)
    # remap: output pixel (X, Y) samples the old raster at (X + dx, Y + dy)
    cc, rr = np.meshgrid(np.arange(W), np.arange(H))
    Xg, Yg = R.mm(cc, rr)
    T = _poly_terms(Xg / 10000.0, Yg / 10000.0, deg)
    dX = T @ cx; dY = T @ cy
    mapx, mapy = R.px(Xg + dX, Yg + dY)
    img = cv2.remap(R.img, mapx.astype(np.float32), mapy.astype(np.float32), cv2.INTER_CUBIC, borderValue=(255, 255, 255))
    info = {'samples': len(sd) + len(td), 'rms_before_mm': round((bx + by) / 2, 1), 'rms_after_mm': round((ax + ay) / 2, 1)}
    return Raster(img, R.x0, R.y1, R.mmpp), info

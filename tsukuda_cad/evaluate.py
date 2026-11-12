"""Compare our base plan with a hand-made reference DXF (same coordinates).

  python -m tsukuda_cad.evaluate OURS.dxf REF.dxf REF_LAYER[,LAYER2] X0,Y0,X1,Y1 [OUT.png]

recall    = share of reference linework that has our line within TOL mm
precision = share of our linework that has a reference line within TOL mm
Both are measured on points sampled every STEP mm along LINE / LWPOLYLINE / ARC / CIRCLE.
"""
import sys, math
import numpy as np
import cv2
import ezdxf

STEP, TOL, MMPP = 50.0, 100.0, 10.0


def sample(doc, layers=None, window=None, skip_layers=()):
    pts = []
    def seg(a, b):
        n = max(2, int(math.dist(a, b) / STEP) + 1)
        for t in np.linspace(0, 1, n):
            pts.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    for e in doc.modelspace():
        L = e.dxf.layer
        if (layers and L not in layers) or L in skip_layers:
            continue
        t = e.dxftype()
        if t == 'LINE':
            seg((e.dxf.start.x, e.dxf.start.y), (e.dxf.end.x, e.dxf.end.y))
        elif t == 'LWPOLYLINE':
            p = [tuple(v[:2]) for v in e.get_points('xy')]
            if e.closed: p.append(p[0])
            for a, b in zip(p, p[1:]): seg(a, b)
        elif t in ('ARC', 'CIRCLE'):
            c, r = e.dxf.center, e.dxf.radius
            a0, a1 = (e.dxf.start_angle, e.dxf.end_angle) if t == 'ARC' else (0, 360)
            span = (a1 - a0) % 360 or 360
            n = max(3, int(math.radians(span) * r / STEP))
            for k in range(n + 1):
                a = math.radians(a0 + span * k / n); pts.append((c.x + r * math.cos(a), c.y + r * math.sin(a)))
    P = np.array(pts) if pts else np.zeros((0, 2))
    if window is not None and len(P):
        X0, Y0, X1, Y1 = window
        P = P[(P[:, 0] >= X0) & (P[:, 0] <= X1) & (P[:, 1] >= Y0) & (P[:, 1] <= Y1)]
    return P


def dist_field(P, window):
    X0, Y0, X1, Y1 = window
    W, H = int((X1 - X0) / MMPP) + 1, int((Y1 - Y0) / MMPP) + 1
    m = np.full((H, W), 255, np.uint8)
    c = ((P[:, 0] - X0) / MMPP).astype(int); r = ((Y1 - P[:, 1]) / MMPP).astype(int)
    ok = (c >= 0) & (c < W) & (r >= 0) & (r < H); m[r[ok], c[ok]] = 0
    return cv2.distanceTransform(m, cv2.DIST_L2, 5) * MMPP


def lookup(D, P, window):
    X0, Y0, X1, Y1 = window
    c = np.clip(((P[:, 0] - X0) / MMPP).astype(int), 0, D.shape[1] - 1)
    r = np.clip(((Y1 - P[:, 1]) / MMPP).astype(int), 0, D.shape[0] - 1)
    return D[r, c]


def compare(ours, ref, ref_layers, window, out_png=None, our_layers=('A-WALL', 'A-DOOR', 'A-COLUMN')):
    A = sample(ezdxf.readfile(ours), set(our_layers), window)
    B = sample(ezdxf.readfile(ref), set(ref_layers), window)
    dA = lookup(dist_field(B, window), A, window)       # ours -> ref
    dB = lookup(dist_field(A, window), B, window)       # ref -> ours
    res = {'recall': float((dB <= TOL).mean()), 'precision': float((dA <= TOL).mean()),
           'median_offset_mm': float(np.median(dA[dA <= TOL])) if (dA <= TOL).any() else None,
           'ref_points': len(B), 'our_points': len(A)}
    if out_png:
        X0, Y0, X1, Y1 = window
        W, H = int((X1 - X0) / MMPP) + 1, int((Y1 - Y0) / MMPP) + 1
        img = np.full((H, W, 3), 255, np.uint8)
        def put(P, d, ok_col, bad_col):
            c = ((P[:, 0] - X0) / MMPP).astype(int); r = ((Y1 - P[:, 1]) / MMPP).astype(int)
            for cc, rr, dd in zip(c, r, d):
                if 0 <= cc < W and 0 <= rr < H:
                    cv2.circle(img, (int(cc), int(rr)), 1, ok_col if dd <= TOL else bad_col, -1)
        put(B, dB, (170, 170, 170), (0, 160, 0))       # reference: grey = matched, GREEN = we missed it
        put(A, dA, (60, 60, 60), (0, 0, 230))          # ours: dark = matched, RED = extra (not in reference)
        cv2.imwrite(out_png, img)
    return res


if __name__ == '__main__':
    ours, ref, layers, win = sys.argv[1:5]
    r = compare(ours, ref, layers.split(','), [float(v) for v in win.split(',')], sys.argv[5] if len(sys.argv) > 5 else None)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})

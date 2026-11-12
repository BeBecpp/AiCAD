"""Read client markup (PDF annotations) and transform it into DWG coordinates via grid bubbles.

  python register_markup.py markup.pdf plan.dxf OUT.dxf

Writes a copy of the DXF with the client's markup on layer T-MARKUP-RED / T-MARKUP-BLUE
(original entities untouched) and markup_dwg.json.
Job-specific part: the grid correspondences in GRID below (scan px of bubble centers, found with
HoughCircles on the decoded scan) -> later automated by `register`.
"""
import sys, json, re
import numpy as np, fitz, ezdxf

pdf, src, dst = sys.argv[1:4]

# ---- grid: scan display px (page rotated to reading orientation) <-> DWG mm ----
COLS = {5: (1601, 1825), 6: (1837, 1825), 7: (2071, 1827), 8: (2309, 1827), 9: (2545, 1825), 10: (2794, 1827)}
COLX = {5: 217117, 6: 223113, 7: 229113, 8: 235113, 9: 241113, 10: 247362}
ROWS = {'A': (290, 1340), 'B': (293, 952), 'C': (296, 565), 'D': (3070, 489)}
ROWY = {'A': -40475, 'B': -30626, 'C': -20778, 'D': -18779}

# similarity with reflection: X = p*xs + q*ys + tx ; Y = q*xs - p*ys + ty
A, b = [], []
for c, (xs, ys) in COLS.items(): A.append([xs, ys, 1, 0]); b.append(COLX[c])
for r, (xs, ys) in ROWS.items(): A.append([-ys, xs, 0, 1]); b.append(ROWY[r])
sol, *_ = np.linalg.lstsq(np.array(A, float), np.array(b, float), rcond=None)
p, q, tx, ty = sol
res = np.array(A) @ sol - np.array(b)
print(f'scale {np.hypot(p, q):.3f} mm/px, rot {np.degrees(np.arctan2(q, p)):.3f} deg, grid residuals mm: {np.round(res).astype(int).tolist()}')

doc = fitz.open(pdf); page = doc[0]
mb = page.mediabox; W_pt, H_pt = mb.width, mb.height            # 842.04 x 1190.52 (unrotated)
SCAN_W, SCAN_H = 2344, 3307; k = SCAN_W / W_pt

def to_dwg(x, y):
    xu, yu = x * k, (H_pt - y) * k                                   # unrotated scan px
    xs, ys = SCAN_H - 1 - yu, xu                                     # rotate 90 cw (display)
    return p * xs + q * ys + tx, q * xs - p * ys + ty

def nums(s): return [float(v) for v in re.findall(r'-?[\d.]+', s or '')]

out = ezdxf.readfile(src); msp = out.modelspace()
for name, aci in (('T-MARKUP-RED', 1), ('T-MARKUP-BLUE', 5)):
    if name not in out.layers: out.layers.add(name, color=aci)
items = []
for a in page.annots():
    x = a.xref; sub = doc.xref_get_key(x, 'Subtype')[1].lstrip('/')
    if sub not in ('Ink', 'Circle'): continue
    col = nums(doc.xref_get_key(x, 'C')[1]); lay = 'T-MARKUP-BLUE' if col[2] > col[0] else 'T-MARKUP-RED'
    rect = nums(doc.xref_get_key(x, 'Rect')[1])
    if sub == 'Circle':
        (x0, y0), (x1, y1) = to_dwg(rect[0], rect[1]), to_dwg(rect[2], rect[3])
        cx, cy, r = (x0 + x1) / 2, (y0 + y1) / 2, (abs(x1 - x0) + abs(y1 - y0)) / 4
        msp.add_circle((cx, cy), r, dxfattribs={'layer': lay})
        items.append({'type': 'circle', 'layer': lay, 'center': [round(cx), round(cy)], 'r': round(r)})
    else:
        raw = doc.xref_get_key(x, 'InkList')[1]
        strokes = [nums(s) for s in raw.strip('[]').split('][')]
        bb = []
        for v in strokes:
            pts = [to_dwg(v[i], v[i + 1]) for i in range(0, len(v) - 1, 2)]
            msp.add_lwpolyline(pts, dxfattribs={'layer': lay}); bb += pts
        bb = np.array(bb)
        items.append({'type': 'ink', 'layer': lay, 'strokes': len(strokes),
                      'bbox': np.round([*bb.min(0), *bb.max(0)]).astype(int).tolist()})
out.saveas(dst)
json.dump({'transform': {'p': p, 'q': q, 'tx': tx, 'ty': ty}, 'grid_residual_mm': res.round(1).tolist(), 'items': items},
          open(dst.rsplit('.', 1)[0] + '_markup.json', 'w'), indent=1)
print(len(items), 'markup items ->', dst)
for it in items: print(it)

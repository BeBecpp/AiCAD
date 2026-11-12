"""Build the final drawing: base + existing furniture (gray, continuous) + markup applied
(additions drawn with real blocks, removals deleted, handwriting dropped, notes typed).

  python compose_job.py JOBDIR [--test]      (--test also applies the photo-test-only items)

JOBDIR needs: out/base_clean.dxf, out/with_markup.dxf (blocks/catalog source), out/with_markup_markup.json
(scan->dwg transform), out/scan_vec.json, interpretation.json
"""
import sys, json, math
import numpy as np, ezdxf
from ezdxf import bbox

job = sys.argv[1]; TEST = '--test' in sys.argv
I = json.load(open(f'{job}/interpretation.json', encoding='utf-8'))
T = json.load(open(f'{job}/out/with_markup_markup.json'))['transform']
p, q, tx, ty = T['p'], T['q'], T['tx'], T['ty']
S = abs(math.hypot(p, q))                                   # mm per scan px
def X(xs, ys): return (p * xs + q * ys + tx, q * xs - p * ys + ty)

doc = ezdxf.readfile(f'{job}/out/base_clean.dxf'); msp = doc.modelspace()
for name, aci, lt in (('T-BASE-SCAN', 7, 'Continuous'), ('T-EXIST-FURN', 252, 'Continuous'), ('T-FURN', 3, 'Continuous'),
                      ('T-DOOR', 7, 'Continuous'), ('T-DIM', 2, 'Continuous'), ('T-NOTE', 2, 'Continuous'), ('T-CHECK', 30, 'Continuous')):
    if name not in doc.layers: doc.layers.add(name, color=aci, linetype=lt)
if 'T-JP' not in doc.styles: doc.styles.add('T-JP', font='msgothic.ttc')
log = {'added': [], 'removed': [], 'checks': [], 'questions': list(I['questions'])}

for e in list(msp.query('TEXT MTEXT[layer=="TEXT"]')):
    t = e.plain_text() if e.dxftype() == 'MTEXT' else e.dxf.text
    if 'クーラー' in t or 'テレキューブ' in t: msp.delete_entity(e)

for e in list(msp.query('DIMENSION')):          # dimensions of the previous telecube layout
    try:
        b = bbox.extents([e])
        if b.has_data and 241000 < b.center.x < 247500 and -31000 < b.center.y < -27000: msp.delete_entity(e)
    except Exception: pass

# ---------------- base extent of the real DWG: scan walls are used only outside it ----------------
dwg_ext = bbox.extents(msp.query('*[layer=="PDF_ジオメトリ"]'))
DWG_X0 = dwg_ext.extmin.x
vec = json.load(open(f'{job}/out/scan_vec.json'))
walls = []                                                   # scan-derived walls (dwg mm) outside DWG coverage
for s in vec['solid']:
    a, b = X(*s[:2]), X(*s[2:])
    if max(a[0], b[0]) < DWG_X0 - 50: walls.append([a, b])
furn = []                                                    # existing furniture, joined dashes -> continuous
for s in vec['dashed']:
    furn.append({'scan': s, 'dwg': [X(*s[:2]), X(*s[2:])]})

def seg_hits_box(a, b, box, pad=0):
    x0, y0, x1, y1 = box
    return max(a[0], b[0]) >= x0 - pad and min(a[0], b[0]) <= x1 + pad and max(a[1], b[1]) >= y0 - pad and min(a[1], b[1]) <= y1 + pad

footprints = []                                              # new items (dwg boxes) -> existing furniture there is removed

# ---------------- telecube sets (catalog = blocks + composition from past finished drawing) ----------------
src = ezdxf.readfile(f'{job}/out/with_markup.dxf')
SET = [('SUB-4WC1CJ-2D', 0, 0, 0), ('SUB-4WC2CJ-2D', 0, 1, 0), ('SUB-4W35BF-2D', 0, 3, 0),
       ('SUB-MS59AG-2D', -2.1, -1.5, 0), ('SUB-MS59AG-2D', 2102.1, 1483.5, 180), ('SUB-4WCC1Y_L-2D', 1758, 2540, 180)]
BODY_W, BODY_D = 2100, 2300
for add in I['add']:
    if add.get('test_only') and not TEST: continue
    if add['kind'] == 'telecube_set':
        sx0, sy0, sx1, sy1 = add['sketch_bbox_dwg']; cx = (sx0 + sx1) / 2
        n, gap = add['count'], add['gap_min_mm']
        total = n * BODY_W + (n - 1) * gap
        # snap: back of the booths onto the nearest wall face below the sketch
        best = None
        for e in msp.query('LINE LWPOLYLINE[layer=="PDF_ジオメトリ"]'):
            pts = [tuple(v[:2]) for v in (e.get_points('xy') if e.dxftype() == 'LWPOLYLINE' else [e.dxf.start, e.dxf.end])]
            for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
                if abs(y1 - y2) < 2 and min(x1, x2) <= cx - total / 2 and max(x1, x2) >= cx + total / 2 and sy0 - 800 < y1 < sy0 + 400:
                    if best is None or y1 > best: best = y1        # highest face under the sketch = wall's room-side face
        y0 = best if best is not None else sy0
        x0 = cx - total / 2
        for i in range(n):
            bx = x0 + i * (BODY_W + gap)
            for name, dx, dy, rot in SET:
                if name not in doc.blocks:
                    from ezdxf.addons import importer
                    imp = importer.Importer(src, doc); imp.import_block(name); imp.finalize()
                msp.add_blockref(name, (bx + dx, y0 + dy), dxfattribs={'layer': 'T-FURN', 'rotation': rot})
            footprints.append((bx, y0, bx + BODY_W, y0 + BODY_D))
        log['added'].append(f"テレキューブ4人用 ×{n}: x {x0:.0f}..{x0 + total:.0f}, 背面を壁面 y={y0:.0f} に合わせ, 間隔 {gap}")
        # dimension of the clearance + note
        dy = y0 + BODY_D + 450
        for i in range(n - 1):
            a = x0 + (i + 1) * BODY_W + i * gap
            d = msp.add_linear_dim(base=(a, dy), p1=(a, y0 + BODY_D), p2=(a + gap, y0 + BODY_D), dimstyle='Standard',
                                   override={'dimtxt': 150, 'dimasz': 100, 'dimexe': 60, 'dimexo': 60, 'dimgap': 40},
                                   dxfattribs={'layer': 'T-DIM'})
            d.render()
            log['checks'].append({'value': f'{gap}', 'what': '排気口クリアランス', 'source': add['gap_source'],
                                  'measured': round(((x0 + (i + 1) * (BODY_W + gap)) - (x0 + (i + 1) * BODY_W + i * gap)))})
        msp.add_mtext(f"4人用テレキューブ クーラー設置タイプ ×{n}\\P排気口クリアランス {gap}以上",
                      dxfattribs={'layer': 'T-NOTE', 'style': 'T-JP', 'char_height': 150, 'insert': (x0, dy + 900)})
        log['checks'].append({'value': str(n), 'what': '台数', 'source': '赤字「4人用テレキューブ」+ スケッチ2台', 'measured': n})

    if add['kind'] == 'door':
        hx, hy = add['hinge_scan_px']; W = add['width_mm']
        H = X(hx, hy)
        # wall lines (scan-derived, vertical) passing the hinge -> cut an opening of W upward (+Y)
        cut = []
        for w in walls:
            (ax, ay), (bx_, by) = w
            if abs(ax - bx_) < 60 and abs(ax - H[0]) < 450 and min(ay, by) < H[1] + W and max(ay, by) > H[1]:
                cut.append(w)
        xs = sorted({round(w[0][0]) for w in cut})
        for w in cut:
            walls.remove(w)
            (ax, ay), (bx_, by) = w; lo, hi = sorted([ay, by])
            if lo < H[1]: walls.append([(ax, lo), (ax, H[1])])
            if hi > H[1] + W: walls.append([(ax, H[1] + W), (ax, hi)])
        if xs:
            for yy in (H[1], H[1] + W):                                 # jambs
                msp.add_line((min(xs), yy), (max(xs), yy), dxfattribs={'layer': 'T-DOOR'})
        hxw = max(xs) if xs else H[0]                                   # hinge on the room-side face
        msp.add_line((hxw, H[1]), (hxw + W, H[1]), dxfattribs={'layer': 'T-DOOR'})      # leaf (open)
        msp.add_arc((hxw, H[1]), W, 0, 90, dxfattribs={'layer': 'T-DOOR'})              # swing
        footprints.append((hxw, H[1], hxw + W, H[1] + W))
        log['added'].append(f"片開きドア W{W}: 壁を {W} 開口, 蝶番 ({hxw:.0f},{H[1]:.0f})")
        log['checks'].append({'value': f'W{W}', 'what': 'ドア幅', 'source': add['width_source'], 'measured': W})

    if add['kind'] == 'desk_group':
        a = X(*add['sketch_bbox_scan_px'][:2]); b = X(*add['sketch_bbox_scan_px'][2:])
        cx, cy = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        W, D, r, c = add['w_mm'], add['d_mm'], add['rows'], add['cols']
        gx0, gy0 = cx - c * W / 2, cy - r * D / 2
        for i in range(c):
            for j in range(r):
                x, y = gx0 + i * W, gy0 + j * D
                msp.add_lwpolyline([(x, y), (x + W, y), (x + W, y + D), (x, y + D)], close=True, dxfattribs={'layer': 'T-FURN'})
        footprints.append((gx0, gy0, gx0 + c * W, gy0 + r * D))
        msp.add_mtext(f"デスク W{W}×D{D} ×{r * c}", dxfattribs={'layer': 'T-NOTE', 'style': 'T-JP', 'char_height': 150, 'insert': (gx0, gy0 + r * D + 350)})
        log['added'].append(f"デスク W{W}×D{D} ×{r * c} at ({cx:.0f},{cy:.0f})")
        log['checks'].append({'value': f'W{W}', 'what': 'デスク幅', 'source': add['w_source'], 'measured': W})
        if add.get('d_assumed'): log['questions'].append(f'デスク奥行 D{D} は仮定です。品番/奥行を指定してください。')

# ---------------- removals ----------------
for rm in I['remove']:
    if rm.get('test_only') and not TEST: continue
    x0, y0, x1, y1 = rm['x_bbox_scan_px']
    hit = [f for f in furn if seg_hits_box(f['scan'][:2], f['scan'][2:], (x0, y0, x1, y1), 4)]
    grow = True
    while grow:                                   # whole item = segments touching the hit ones (within ~150 px)
        grow = False
        for f in furn:
            if f in hit: continue
            s = f['scan']
            if math.dist(((s[0] + s[2]) / 2, (s[1] + s[3]) / 2), ((x0 + x1) / 2, (y0 + y1) / 2)) > 150: continue
            if any(seg_hits_box(s[:2], s[2:], (min(h['scan'][0], h['scan'][2]), min(h['scan'][1], h['scan'][3]),
                                                max(h['scan'][0], h['scan'][2]), max(h['scan'][1], h['scan'][3])), 3) for h in hit):
                hit.append(f); grow = True
    for f in hit: furn.remove(f)
    if hit:
        log['removed'].append(f"{rm['source']}: 既存家具 {len(hit)} 本の線を削除")
    else:   # never guess: nothing recognisable under the X -> ask
        cx, cy = X((x0 + x1) / 2, (y0 + y1) / 2)
        msp.add_circle((cx, cy), 900, dxfattribs={'layer': 'T-CHECK'})
        log['questions'].append(f"{rm['source']} の位置 ({cx:.0f},{cy:.0f}) に撤去対象の家具が見つかりません。何を撤去しますか？")

# existing furniture under new items: remove the WHOLE item (segments connected to the hit ones), not fragments
def sbox(f, pad=0):
    (ax, ay), (bx_, by) = f['dwg']; return (min(ax, bx_) - pad, min(ay, by) - pad, max(ax, bx_) + pad, max(ay, by) + pad)
def touch(f, g, pad=80):
    a, b = sbox(f, pad), sbox(g)
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]
conf = [f for f in furn if any(seg_hits_box(*f['dwg'], fp, 100) for fp in footprints)]
grow = True
while grow:
    grow = False
    for f in furn:
        if f in conf: continue
        if any(touch(f, g) for g in conf):
            bx = [sbox(g) for g in conf + [f]]
            if max(b[2] for b in bx) - min(b[0] for b in bx) < 6000 and max(b[3] for b in bx) - min(b[1] for b in bx) < 6000:
                conf.append(f); grow = True
for f in conf: furn.remove(f)
if conf:
    log['removed'].append(f"新規配置と重なる既存家具 {len(conf)} 本の線を削除（要確認）")
    log['questions'].append('新規配置と重なる既存家具を撤去扱いにしました。よろしいですか？')

# ---------------- notes (typed; handwriting itself is not copied) ----------------
for n in I['notes']:
    msp.add_mtext(n['text'], dxfattribs={'layer': 'T-NOTE', 'style': 'T-JP', 'char_height': 180, 'insert': tuple(n['at_dwg'])})
    if 'leader_to_dwg' in n:
        msp.add_leader([tuple(n['leader_to_dwg']), (n['at_dwg'][0], n['at_dwg'][1] - 150)], dxfattribs={'layer': 'T-NOTE'})

# ---------------- write base walls + existing furniture ----------------
for (a, b) in walls: msp.add_line(a, b, dxfattribs={'layer': 'T-BASE-SCAN'})
for f in furn: msp.add_line(*f['dwg'], dxfattribs={'layer': 'T-EXIST-FURN'})

# questions box (T-CHECK)
qx, qy = DWG_X0 - 300, -15500
msp.add_mtext('■確認事項\\P' + '\\P'.join(f'{i + 1}. {t}' for i, t in enumerate(log['questions'])),
              dxfattribs={'layer': 'T-CHECK', 'style': 'T-JP', 'char_height': 200, 'insert': (qx - 28000, qy), 'width': 26000})
name = 'final_test' if TEST else 'final'
doc.saveas(f'{job}/out/{name}.dxf')
json.dump(log, open(f'{job}/out/{name}_log.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(log, ensure_ascii=False, indent=1))
print('scan walls', len(walls), 'existing furniture lines', len(furn), '->', f'{job}/out/{name}.dxf')

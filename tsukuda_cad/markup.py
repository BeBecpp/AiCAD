"""Client markup -> clean CAD items.

Principle: the sketch says WHAT and roughly WHERE; a written number says EXACTLY where.
An item may carry offsets like {"from": "left", "mm": 1591}: the nearest wall line on that side of
the sketched position is found in the vectorised plan and the item is placed exactly `mm` from it,
with a real dimension drawn. Numbers that could not be read are never guessed: the item stays at
the sketched position and a question is added.
"""
import math
from . import dxfout

SIDES = {'left': (-1, 0), 'right': (1, 0), 'top': (0, 1), 'bottom': (0, -1)}


def nearest_wall(segs, x, y, side, max_mm=6000, min_len=400):
    """Closest line on `side` of (x, y) that spans past the point (vertical for left/right, horizontal for top/bottom)."""
    dx, dy = SIDES[side]
    best = None
    for s in segs:
        if s.length < min_len:
            continue
        if dx and not s.horiz and min(s.y1, s.y2) <= y <= max(s.y1, s.y2):
            d = (s.x1 - x) * dx
            if 0 < d < max_mm and (best is None or d < best[0]): best = (d, s.x1)
        if dy and s.horiz and min(s.x1, s.x2) <= x <= max(s.x1, s.x2):
            d = (s.y1 - y) * dy
            if 0 < d < max_mm and (best is None or d < best[0]): best = (d, s.y1)
    return best[1] if best else None


def place(items, segs, msp, log):
    for it in items:
        x, y = it['sketch_at']
        w, h = it.get('size_mm', (0, 0))
        if it.get('dim_to') == 'center':
            w = h = 0                       # numbers measured to the item centre
        dims = []
        for off in it.get('offsets', []):
            side = off['from']; ref = nearest_wall(segs, x, y, side)
            if ref is None:
                log['questions'].append(f"{it['label']}: {side}側の基準壁が見つかりません"); continue
            if off.get('mm') is None:
                log['questions'].append(f"{it['label']}: {side}側の寸法「{off.get('read', '?')}」が読めません（スケッチ位置のまま）")
                continue
            v = off['mm']; dx, dy = SIDES[side]
            # the near edge of the item is `v` from the wall face
            if dx: x = ref - dx * (v + w / 2)
            else:  y = ref - dy * (v + h / 2)
            dims.append((side, ref, v))
            log['checks'].append({'item': it['label'], 'value': v, 'from': f'{side}壁', 'source': off.get('source', '赤字')})
        # symbol
        layer = 'T-MARK'
        bw, bh = it.get('size_mm', (300, 300))     # drawn size (w, h above = size used for the dimensions)
        if it['symbol'] == 'box':
            pts = [(x - bw / 2, y - bh / 2), (x + bw / 2, y - bh / 2), (x + bw / 2, y + bh / 2), (x - bw / 2, y + bh / 2)]
            msp.add_lwpolyline(pts, close=True, dxfattribs={'layer': layer})
        elif it['symbol'] == 'circle':
            msp.add_circle((x, y), it.get('r_mm', 150), dxfattribs={'layer': layer})
            if it.get('cross'):
                r = it.get('r_mm', 150)
                msp.add_line((x - r * .7, y - r * .7), (x + r * .7, y + r * .7), dxfattribs={'layer': layer})
                msp.add_line((x - r * .7, y + r * .7), (x + r * .7, y - r * .7), dxfattribs={'layer': layer})
        elif it['symbol'] == 'outlet':           # wall outlet: small circle with two ticks (JIS-like)
            r = 120
            msp.add_circle((x, y), r, dxfattribs={'layer': layer})
            msp.add_line((x - r, y), (x + r, y), dxfattribs={'layer': layer})
        # dimensions from the reference walls to the item's near edge
        for side, ref, v in dims:
            dx, dy = SIDES[side]
            if dx:
                edge = x + dx * w / 2 if w else x
                d = msp.add_linear_dim(base=(0, y), p1=(ref, y), p2=(edge, y), dimstyle='T100', dxfattribs={'layer': 'T-MARK-DIM'})
            else:
                edge = y + dy * h / 2 if h else y
                d = msp.add_linear_dim(base=(x, 0), p1=(x, ref), p2=(x, edge), angle=90, dimstyle='T100', dxfattribs={'layer': 'T-MARK-DIM'})
            d.render()
        lx, ly = it.get('label_at', (x + 250, y + 250))
        dxfout.text(msp, it['label'], (lx, ly), it.get('text_h', 180), layer='T-NOTE', align='MIDDLE_LEFT')
        if it.get('leader', True) and math.hypot(lx - x, ly - y) > 400:
            msp.add_line((x, y), (lx - 50, ly), dxfattribs={'layer': 'T-NOTE'})
        for q in it.get('questions', []):
            log['questions'].append(f"{it['label']}: {q}")
        log['added'].append({'item': it['label'], 'at': [round(x), round(y)]})

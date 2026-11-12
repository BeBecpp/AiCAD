"""Sheet job: marked-up plan (phone photo or scan) -> clean base DXF (+ markup) and check images.

  python -m tsukuda_cad.run_photo_job JOBDIR

JOBDIR/job.json holds only what was READ from the sheet (rotation, grid dimensions, room names,
markup, rough door boxes...). Everything geometric is measured from the image through the grid.
Registration:
  * photo - filled columns at grid nodes -> homography (handles perspective)
  * scan  - grid bubbles -> similarity (job['bubbles_px'])
"""
import json, sys
from pathlib import Path
import cv2

from . import photo, vectorize as vz, dxfout, render, markup


def grid_lines(g):
    gx = sorted(g['x_mm'] + g.get('x_sub_mm', []))
    if 'y_mm' in g:
        gy = sorted(g['y_mm'])
    else:
        top = max(g['y_from_top_mm']); gy = sorted(top - v for v in g['y_from_top_mm'])
    return gx, gy


def register(job, up):
    g = job['grid']
    if job.get('kind') == 'scan':
        b = job['bubbles_px']
        fit = photo.fit_bubbles(b['cols'], b['rows'], dict(zip(g['x_names'], g['x_mm'])),
                                dict(zip(g['y_names'], g['y_mm'])))
        return fit, []
    norm = photo.normalize_light(cv2.cvtColor(up, cv2.COLOR_BGR2GRAY))
    cols_px = photo.detect_columns(norm)
    return photo.fit_grid(cols_px, g['x_mm'], g['y_from_top_mm']), cols_px


def mask_out(R, mask, box):
    X0, Y0, X1, Y1 = box
    c0, r1 = R.px(X0, Y0); c1, r0 = R.px(X1, Y1)
    mask[max(int(r0), 0):max(int(r1), 0), max(int(c0), 0):max(int(c1), 0)] = 0


def main(jobdir):
    J = Path(jobdir); out = J / 'out'; out.mkdir(exist_ok=True)
    job = json.loads((J / 'job.json').read_text(encoding='utf-8'))
    g = job['grid']; gx, gy = grid_lines(g)

    # 1. image -> mm ---------------------------------------------------------------------------
    up = photo.load_upright(J / job['photo'], job['rotation'])
    fit, cols_px = register(job, up)
    print(f'registration: {fit.used} references, rms {fit.rms_mm:.0f} mm, max {fit.max_mm:.0f} mm')
    R = photo.rectify(up, fit, job['bounds_mm'], job['mm_per_px'])
    dist = None
    if job.get('local_distortion_fix', False):                    # experimental
        R, dist = photo.refine_with_grid(R, gx, gy)
    cv2.imwrite(str(out / 'rect.png'), R.img)

    # 2. vectorize --------------------------------------------------------------------------------
    prn, red, gray = vz.ink_masks(R)
    for box in job.get('ignore_mm', []):                           # dimension strings, legends, title...
        mask_out(R, prn, box)
    for t in job.get('texts', []):                                 # lettering is re-typed, not traced
        X, Y = t['at']; w = t.get('w', 1200); h = t.get('h_box', 500)
        mask_out(R, prn, (X - w / 2, Y - h / 2, X + w / 2, Y + h / 2))
    segs = vz.drop_light(R, gray, vz.ortho_lines(R, prn, gx, gy))
    if job.get('drop_dashed'):                                     # furniture / wiring / ceiling zones
        segs = vz.drop_dashed(segs)
    segs = vz.drop_isolated(segs)
    if job.get('drop_floating'):                                   # wiring runs, leaders
        segs = vz.drop_floating(segs)

    cols = []
    if job.get('columns', True):                                   # filled columns at grid nodes only
        found = vz.columns(R, prn, g['x_mm'], gy)
        if cols_px:
            found += [(X, Y, 0) for X, Y in fit.to_mm([c[:2] for c in cols_px])]
        size = next((s_ for *_, s_ in found if s_), 600)
        nodes = {}
        for X, Y, _ in found:
            sx = min(g['x_mm'], key=lambda v: abs(v - X)); sy = min(gy, key=lambda v: abs(v - Y))
            if abs(sx - X) < 250 and abs(sy - Y) < 250:
                nodes[(sx, sy)] = (sx, sy, size)
        cols = list(nodes.values())
        inside = lambda x, y, X, Y, S: abs(x - X) <= S / 2 + 20 and abs(y - Y) <= S / 2 + 20
        segs = [s for s in segs if not any(inside(s.x1, s.y1, X, Y, S) and inside(s.x2, s.y2, X, Y, S)
                                           for X, Y, S in cols)]

    used = vz.line_raster(R, segs, 3)
    for X, Y, s in cols:
        c0, r1 = R.px(X - s / 2 - 20, Y - s / 2 - 20); c1, r0 = R.px(X + s / 2 + 20, Y + s / 2 + 20)
        used[int(r0):int(r1), int(c0):int(c1)] = 255
    arcs = [a for a in vz.arcs(R, gray, red, used, light=200) if a.r > 1200]         # curved facades
    for box in job.get('door_swing_boxes_mm', []):                                      # read box -> exact fit
        a = vz.fit_arc_in_box(R, gray, red, used, box, segs)
        if a:
            arcs.append(a)
        else:
            print('  door swing not found in', box)
    if job.get('auto_doors'):
        arcs += vz.hinge_filter(vz.door_arcs(R, gray, red, used, r_mm=tuple(job.get('door_r_mm', (550, 1100)))), segs)
    print(f'{len(segs)} lines, {len(arcs)} arcs, {len(cols)} columns')

    # 3. DXF ----------------------------------------------------------------------------------------
    doc = dxfout.new_doc(); msp = doc.modelspace()
    ext = job.get('grid_extent_mm', [gx[0] - 2000, gy[0] - 2000, gx[-1] + 2000, gy[-1] + 2000])
    for X in gx:
        msp.add_line((X, ext[1]), (X, ext[3]), dxfattribs={'layer': 'A-GRID'})
    for Y in gy:
        msp.add_line((ext[0], Y), (ext[2], Y), dxfattribs={'layer': 'A-GRID'})
    for s in segs:
        if s.kind != 'grid':                                       # grid-coincident pieces = the grid line
            msp.add_line((s.x1, s.y1), (s.x2, s.y2), dxfattribs={'layer': 'A-WALL'})
    for a in arcs:
        msp.add_arc((a.cx, a.cy), a.r, a.a0, a.a1, dxfattribs={'layer': 'A-DOOR'})
    for X, Y, S in cols:
        dxfout.column(msp, X, Y, S)
    for t in job.get('texts', []):
        dxfout.text(msp, t['text'], tuple(t['at']), t.get('height', 250))
    for d in job.get('dims', []):
        dxfout.dim_chain(msp, d['pts'], d['base'], d.get('dir', 'h') == 'h')
    doc.saveas(out / 'plan_base.dxf')                              # base plan only (what the sheet shows)

    # 4. markup -> clean items (the handwriting itself is not copied) -----------------------------
    log = {'added': [], 'checks': [], 'questions': [],
           'reading': {'grid_x': g.get('x_source'), 'grid_y': g.get('y_source'),
                       'registration_rms_mm': round(fit.rms_mm), 'registration_max_mm': round(fit.max_mm),
                       'local_distortion': dist}}
    mk = job.get('markup')
    if mk:
        markup.place(mk['items'], segs, msp, log)
        for n in mk.get('notes', []):
            dxfout.text(msp, n['text'], tuple(n['at']), n.get('height', 180), layer='T-NOTE')
        log['questions'] += mk.get('questions', [])
        if log['questions']:
            qx, qy = job.get('questions_at', [ext[0], ext[1] - 1200])
            for i, q in enumerate(['■確認事項'] + [f'{i + 1}. {q}' for i, q in enumerate(log['questions'])]):
                dxfout.text(msp, q, (qx, qy - i * 320), 200, layer='T-CHECK', align='LEFT')
    doc.saveas(out / 'plan_final.dxf')
    (out / 'check.json').write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding='utf-8')

    # 5. check images -------------------------------------------------------------------------------
    render.overlay(R, segs, arcs, cols, out / 'overlay.png')
    view = job.get('view_mm', job['bounds_mm'])
    render.dxf_png(out / 'plan_base.dxf', out / 'plan_base.png', view)
    render.dxf_png(out / 'plan_final.dxf', out / 'plan_final.png', view)
    render.side_by_side(R, out / 'plan_final.png', view, out / 'compare.png')
    print('wrote', out / 'plan_base.dxf', out / 'plan_final.dxf')
    return doc, R, segs, arcs, cols


if __name__ == '__main__':
    main(sys.argv[1])

"""Render a DXF layout (or a model-space window) to PNG, optionally hiding layers.

  python render_preview.py plan.dxf out.png [--layout NAME] [--window x0,y0,x1,y1] [--hide L1,L2] [--px 4000] [--mono]
"""
import argparse, ezdxf, cv2, numpy as np
from ezdxf import bbox
from ezdxf.addons.drawing import RenderContext, Frontend, pymupdf, layout
from ezdxf.addons.drawing.config import Configuration, BackgroundPolicy, ColorPolicy

ap = argparse.ArgumentParser()
ap.add_argument('dxf'); ap.add_argument('png')
ap.add_argument('--layout', default='Model'); ap.add_argument('--window')
ap.add_argument('--hide', default=''); ap.add_argument('--px', type=int, default=4000)
ap.add_argument('--mono', action='store_true'); ap.add_argument('--recolor', default='')
a = ap.parse_args()

doc = ezdxf.readfile(a.dxf)
for spec in filter(None, a.recolor.split(',')):            # LAYER=ACI : force a layer's color (for comparisons)
    ln, aci = spec.rsplit('=', 1)
    if ln in doc.layers:
        doc.layers.get(ln).color = int(aci)
        for e in doc.modelspace().query(f'*[layer=="{ln}"]'):
            e.dxf.color = 256; e.dxf.discard('true_color')
lay = doc.modelspace() if a.layout == 'Model' else doc.paperspace(a.layout)
hide = {h for h in a.hide.split(',') if h}
cfg = Configuration(background_policy=BackgroundPolicy.WHITE,
                    color_policy=ColorPolicy.BLACK if a.mono else ColorPolicy.COLOR, min_lineweight=0.15)
be = pymupdf.PyMuPdfBackend()
win = list(map(float, a.window.split(','))) if a.window else None

U = []
def filt(e):
    if e.dxf.get('layer') in hide: return False
    if not win: return True
    try:
        b = bbox.extents([e])
        if not b.has_data: return False
        if b.extmax.x < win[0] or b.extmin.x > win[2] or b.extmax.y < win[1] or b.extmin.y > win[3]: return False
        if b.size.x > 3 * (win[2] - win[0]) and b.size.y > 3 * (win[3] - win[1]): return False   # sheet frames
        U.append((b.extmin.x, b.extmin.y, b.extmax.x, b.extmax.y)); return True
    except Exception:
        return False

Frontend(RenderContext(doc), be, config=cfg).draw_layout(lay, finalize=True, filter_func=filt)
if win:
    # render the whole drawn extent U, then crop the window out of it (keeps long walls that cross the edge)
    u = np.array(U); ux0, uy0 = u[:, :2].min(0); ux1, uy1 = u[:, 2:].max(0)
    uw, uh = ux1 - ux0, uy1 - uy0; s_ = 400 / max(uw, uh)
    page = layout.Page(uw * s_, uh * s_, layout.Units.mm, margins=layout.Margins.all(0))
    px_per_mm = a.px / (win[2] - win[0])
    dpi = px_per_mm * uw / (page.width / 25.4)
    img = cv2.imdecode(np.frombuffer(be.get_pixmap_bytes(page, fmt='png', dpi=int(dpi), settings=layout.Settings(fit_page=True)), np.uint8), 1)
    H, W = img.shape[:2]; fx, fy = W / uw, H / uh
    c0, c1 = int((win[0] - ux0) * fx), int((win[2] - ux0) * fx); r0, r1 = int((uy1 - win[3]) * fy), int((uy1 - win[1]) * fy)
    cv2.imwrite(a.png, img[max(r0, 0):r1, max(c0, 0):c1])
else:
    page = layout.Page(420, 297, layout.Units.mm, margins=layout.Margins.all(5))
    dpi = a.px / (max(page.width, page.height) / 25.4)
    open(a.png, 'wb').write(be.get_pixmap_bytes(page, fmt='png', dpi=int(dpi), settings=layout.Settings(fit_page=True)))
print('wrote', a.png)

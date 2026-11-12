"""Check images: CAD geometry drawn over the rectified photo, and DXF -> PNG."""
import cv2
import numpy as np


def overlay(R, segs, arcs, cols, path):
    base = cv2.cvtColor(cv2.cvtColor(R.img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    o = (base * 0.45 + 140).astype(np.uint8)
    P = lambda X, Y: tuple(int(round(v)) for v in R.px(X, Y))
    for s in segs:
        cv2.line(o, P(s.x1, s.y1), P(s.x2, s.y2), (200, 120, 0) if s.kind == 'grid' else (0, 0, 220), 1)
    for a in arcs:
        c = P(a.cx, a.cy); r = int(a.r / R.mmpp)
        cv2.ellipse(o, c, (r, r), 0, -a.a1, -a.a0 if a.a0 < a.a1 else -a.a0 + 360, (0, 160, 0), 1)
    for X, Y, S in cols:
        cv2.rectangle(o, P(X - S / 2, Y + S / 2), P(X + S / 2, Y - S / 2), (160, 0, 160), 1)
    cv2.imwrite(str(path), o)


def dxf_png(dxf_path, png_path, view_mm, px_per_m=120):
    """Render model space of a DXF to PNG, exactly the window view_mm=(X0,Y0,X1,Y1), white paper."""
    import ezdxf
    from ezdxf.addons.drawing import RenderContext, Frontend, pymupdf, layout
    from ezdxf.addons.drawing.config import Configuration, BackgroundPolicy, ColorPolicy, LineweightPolicy
    doc = ezdxf.readfile(str(dxf_path)); msp = doc.modelspace()
    X0, Y0, X1, Y1 = view_mm
    # invisible frame fixes the drawing extents to the view window
    doc.layers.add('VIEWFRAME', color=7)
    msp.add_lwpolyline([(X0, Y0), (X1, Y0), (X1, Y1), (X0, Y1)], close=True,
                       dxfattribs={'layer': 'VIEWFRAME', 'true_color': 0xFFFFFF})
    cfg = Configuration(background_policy=BackgroundPolicy.WHITE, color_policy=ColorPolicy.COLOR,
                        lineweight_policy=LineweightPolicy.ABSOLUTE, lineweight_scaling=1.0)
    be = pymupdf.PyMuPdfBackend()
    Frontend(RenderContext(doc), be, config=cfg).draw_layout(msp, finalize=True)
    w_mm, h_mm = (X1 - X0) / 100, (Y1 - Y0) / 100            # 1/100 paper size
    page = layout.Page(w_mm, h_mm, layout.Units.mm, margins=layout.Margins.all(0))
    dpi = px_per_m * (X1 - X0) / 1000 / (w_mm / 25.4)
    png = be.get_pixmap_bytes(page, fmt='png', dpi=int(dpi), settings=layout.Settings(fit_page=True))
    open(png_path, 'wb').write(png)


def side_by_side(R, cad_png, view_mm, path):
    """Rectified photo (left) | CAD (right), same scale and window."""
    X0, Y0, X1, Y1 = view_mm
    W, H = int((X1 - X0) / R.mmpp), int((Y1 - Y0) / R.mmpp)
    ph = np.full((H, W, 3), 255, np.uint8)                    # photo pasted into the view window
    c0, r0 = (int(round(v)) for v in R.px(X0, Y1))
    src = R.img[max(r0, 0):r0 + H, max(c0, 0):c0 + W]
    ph[max(-r0, 0):max(-r0, 0) + src.shape[0], max(-c0, 0):max(-c0, 0) + src.shape[1]] = src
    cad = cv2.imread(str(cad_png))
    cad = cv2.resize(cad, (ph.shape[1], ph.shape[0]), interpolation=cv2.INTER_AREA)
    gap = np.full((ph.shape[0], 20, 3), 255, np.uint8)
    cv2.imwrite(str(path), np.hstack([ph, gap, cad]))

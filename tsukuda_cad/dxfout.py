"""Write the reconstructed plan + markup to DXF on standard layers."""
import ezdxf
from ezdxf.enums import TextEntityAlignment

LAYERS = {   # name: (ACI color, linetype, lineweight 1/100 mm)
    'A-GRID':      (8, 'DASHDOT', 13),
    'A-COLUMN':    (7, 'Continuous', 35),
    'A-WALL':      (7, 'Continuous', 25),
    'A-DOOR':      (7, 'Continuous', 18),
    'A-ROOM':      (7, 'Continuous', 18),
    'A-DIM':       (7, 'Continuous', 13),
    'A-SYMB':      (7, 'Continuous', 13),
    'T-MARK':      (1, 'Continuous', 25),     # client markup, redrawn clean
    'T-MARK-DIM':  (1, 'Continuous', 13),
    'T-NOTE':      (1, 'Continuous', 18),
    'T-CHECK':     (30, 'Continuous', 18),
}
TEXT_STYLE = 'JP'


def new_doc():
    doc = ezdxf.new('R2018', setup=True)
    doc.header['$INSUNITS'] = 4          # mm
    doc.header['$LTSCALE'] = 50          # 1/100 drawing, linetype pattern scale
    for name, (aci, lt, lw) in LAYERS.items():
        doc.layers.add(name, color=aci, linetype=lt, lineweight=lw)
    doc.styles.add(TEXT_STYLE, font='msgothic.ttc')
    ds = doc.dimstyles.duplicate_entry('Standard', 'T100')
    for k, v in dict(dimtxt=180, dimasz=120, dimexe=100, dimexo=100, dimgap=50, dimtsz=0, dimdec=0,
                     dimtxsty=TEXT_STYLE, dimblk='ARCHTICK', dimclrd=7, dimclre=7, dimclrt=7).items():
        ds.set_dxf_attrib(k, v)
    return doc


def text(msp, s, at, h=250, layer='A-ROOM', align='MIDDLE_CENTER', rot=0):
    t = msp.add_text(s, height=h, dxfattribs={'layer': layer, 'style': TEXT_STYLE, 'rotation': rot})
    t.set_placement(at, align=TextEntityAlignment[align])
    return t


def dim_chain(msp, pts, base, horizontal=True, layer='A-DIM'):
    """Chain of linear dims through pts (mm) with the dimension line at `base` (y for horizontal, x for vertical)."""
    for a, b in zip(pts, pts[1:]):
        if horizontal:
            d = msp.add_linear_dim(base=(a, base), p1=(a, base), p2=(b, base), dimstyle='T100', dxfattribs={'layer': layer})
        else:
            d = msp.add_linear_dim(base=(base, a), p1=(base, a), p2=(base, b), angle=90, dimstyle='T100', dxfattribs={'layer': layer})
        d.render()


def column(msp, x, y, s, layer='A-COLUMN'):
    h = s / 2
    pts = [(x - h, y - h), (x + h, y - h), (x + h, y + h), (x - h, y + h)]
    msp.add_lwpolyline(pts, close=True, dxfattribs={'layer': layer})
    hatch = msp.add_hatch(color=256, dxfattribs={'layer': layer})
    hatch.paths.add_polyline_path(pts, is_closed=True)

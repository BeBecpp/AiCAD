"""Inspect a DXF (converted from DWG by ODA): layers, entity counts, red entities, texts, render.

  python dwg_query.py plan.dxf OUTDIR
"""
import sys, os, json, collections
import ezdxf
from ezdxf import bbox, colors
from ezdxf.addons.drawing import matplotlib as dmpl

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
doc = ezdxf.readfile(src)
msp = doc.modelspace()

def rgb(e):
    """Effective color of an entity (resolves BYLAYER)."""
    if e.dxf.hasattr('true_color'):
        return colors.int2rgb(e.dxf.true_color)
    aci = e.dxf.get('color', 256)
    if aci == 256:
        lay = doc.layers.get(e.dxf.layer) if e.dxf.layer in doc.layers else None
        aci = abs(lay.color) if lay else 7
    return colors.aci2rgb(aci) if 0 < aci < 256 else (0, 0, 0)

def is_red(c): return c[0] > 180 and c[1] < 90 and c[2] < 90

per_layer = collections.Counter(); per_type = collections.Counter(); red = []
texts = []
for e in msp:
    per_layer[e.dxf.layer] += 1; per_type[e.dxftype()] += 1
    c = rgb(e)
    if is_red(c):
        try: b = bbox.extents([e]); bb = [round(v) for v in (b.extmin.x, b.extmin.y, b.extmax.x, b.extmax.y)] if b.has_data else None
        except Exception: bb = None
        red.append({'type': e.dxftype(), 'layer': e.dxf.layer, 'bbox': bb,
                    'text': e.plain_text() if e.dxftype() in ('TEXT', 'MTEXT') else None})
    if e.dxftype() in ('TEXT', 'MTEXT'):
        t = e.plain_text() if e.dxftype() == 'MTEXT' else e.dxf.text
        p = e.dxf.insert
        texts.append({'t': t.strip(), 'x': round(p.x), 'y': round(p.y), 'layer': e.dxf.layer,
                      'h': round(e.dxf.get('char_height' if e.dxftype() == 'MTEXT' else 'height', 0))})

ext = bbox.extents(msp)
layers = [{'name': l.dxf.name, 'aci': l.color, 'on': l.is_on(), 'frozen': l.is_frozen(), 'n': per_layer[l.dxf.name]}
          for l in doc.layers]
blocks = collections.Counter(e.dxf.name for e in msp.query('INSERT'))
info = {'version': doc.dxfversion, 'units($INSUNITS)': doc.header.get('$INSUNITS'),
        'extents': [round(v) for v in (ext.extmin.x, ext.extmin.y, ext.extmax.x, ext.extmax.y)],
        'entity_types': dict(per_type.most_common()), 'layers': layers,
        'inserts_top': blocks.most_common(40), 'n_red': len(red), 'red': red[:300],
        'paperspace_layouts': [l.name for l in doc.layouts if l.name != 'Model'],
        'texts': texts}
json.dump(info, open(f'{out}/dwg_info.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print({k: v for k, v in info.items() if k not in ('layers', 'red', 'texts', 'inserts_top')})
print('layers:', [(l['name'], l['aci'], l['n']) for l in layers if l['n']])
print('red entities:', len(red), collections.Counter(r['layer'] for r in red).most_common(10))
print('texts:', len(texts))

# render model space (white background)
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.config import Configuration, BackgroundPolicy, ColorPolicy
fig = plt.figure(figsize=(42 / 2.54 * 1.6, 29.7 / 2.54 * 1.6))
ax = fig.add_axes([0, 0, 1, 1])
cfg = Configuration(background_policy=BackgroundPolicy.WHITE, color_policy=ColorPolicy.COLOR, lineweight_scaling=0.5)
Frontend(RenderContext(doc), dmpl.MatplotlibBackend(ax), config=cfg).draw_layout(msp, finalize=True)
fig.savefig(f'{out}/model.png', dpi=150)
print('rendered', f'{out}/model.png')

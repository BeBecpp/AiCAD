import re, sys, cv2, numpy as np, json
S = sys.argv[1]
t = open(S + '/objstm41.bin', 'rb').read().decode('latin1')
img = cv2.cvtColor(cv2.imread(S + '/scan.png', 0), cv2.COLOR_GRAY2BGR)
H = img.shape[0]; k = 2344 / 842.04
P = lambda x, y: (int(x * k), int((1190.52 - y) * k))
annots = []
# split annotation dicts roughly at '/Type/Annot>>'
for chunk in t.split('/Type/Annot>>'):
    st = re.search(r'/Subtype/(\w+)', chunk)
    if not st: continue
    sub = st.group(1)
    c = re.search(r'/C\[([\d. ]+)\]', chunk)
    col = [float(v) for v in c.group(1).split()] if c else [0, 0, 0]
    bgr = tuple(int(255 * v) for v in col[::-1])
    rect = [float(v) for v in re.search(r'/Rect\[([\d. ]+)\]', chunk).group(1).split()]
    a = {'type': sub, 'color': col, 'rect': rect}
    if sub == 'Circle':
        cv2.ellipse(img, (P((rect[0]+rect[2])/2, (rect[1]+rect[3])/2)), (int((rect[2]-rect[0])/2*k), int((rect[3]-rect[1])/2*k)), 0, 0, 360, bgr, 3)
    if sub == 'Ink':
        il = re.search(r'/InkList\[(.*?)\]\]', chunk, re.S).group(1)
        strokes = []
        for s in il.split(']['):
            v = [float(x) for x in re.findall(r'[\d.]+', s)]
            pts = np.array([P(v[i], v[i+1]) for i in range(0, len(v) - 1, 2)], np.int32)
            strokes.append(len(pts)); cv2.polylines(img, [pts], False, bgr, 3)
        a['strokes'] = strokes
    for key in ('Contents', 'Subj'):
        mm = re.search('/' + key + r'\((.*?)\)/', chunk)
        if mm: a[key] = mm.group(1)[:40]
    if sub != 'Popup': annots.append(a)
for a in annots: print(a['type'], [round(c, 2) for c in a['color']], [round(r) for r in a['rect']], a.get('strokes', '')[:8] if a.get('strokes') else '')
disp = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)  # /Rotate 90
cv2.imwrite(S + '/overlay_full.png', cv2.resize(disp, None, fx=0.45, fy=0.45, interpolation=cv2.INTER_AREA))
# crop around markup (unrotated pt coords x 215-360, y 900-1100) -> crop in unrotated then rotate
x0, y0 = P(200, 1110); x1, y1 = P(380, 880)
cv2.imwrite(S + '/overlay_markup.png', cv2.rotate(img[y0:y1, x0:x1], cv2.ROTATE_90_CLOCKWISE))

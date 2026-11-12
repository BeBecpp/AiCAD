import re, sys, cv2, numpy as np
S = sys.argv[1]; t = open(S + '/objstm41.bin', 'rb').read().decode('latin1')
k = 14; X0, X1, Y0, Y1 = 215, 360, 905, 1095
img = np.full((int((X1 - X0) * k), int((Y1 - Y0) * k), 3), 255, np.uint8)  # rotated: rows=x, cols=y
for chunk in t.split('/Type/Annot>>'):
    if '/Subtype/Ink' not in chunk: continue
    col = [float(v) for v in re.search(r'/C\[([\d. ]+)\]', chunk).group(1).split()]
    bgr = tuple(int(255 * v) for v in col[::-1])
    il = re.search(r'/InkList\[(.*?)\]\]', chunk, re.S).group(1)
    for s in il.split(']['):
        v = [float(x) for x in re.findall(r'[\d.]+', s)]
        # display rotation 90cw: display_x = y - Y0, display_y = x - X0
        pts = np.array([((v[i+1] - Y0) * k, (v[i] - X0) * k) for i in range(0, len(v) - 1, 2)], np.int32)
        cv2.polylines(img, [pts], False, bgr, 4, cv2.LINE_AA)
cv2.imwrite(S + '/ink_zoom.png', img)
print(img.shape)

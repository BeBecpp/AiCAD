import json, cv2, numpy as np
gt = json.load(open('photo_gt.json')); d = json.load(open('out/groups.json'))
c = np.array(d['sheet_corners_photo_px']); t = np.array(gt['corners'])
print('corner error px (photo):', np.hypot(*(c - t).T).round(1))
for nm in ('red', 'blue'):
    g = cv2.imread(f'gt_{nm}.png', 0) > 100; p = cv2.imread(f'out/{nm}.png', 0) > 0
    tol = lambda m: cv2.dilate(m.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    print(nm, 'recall %.3f precision %.3f' % ((g & tol(p)).sum() / g.sum(), (p & tol(g)).sum() / max(p.sum(), 1)))
for g in d['groups']: print(g['id'], g['color'], g['bbox'], g['ink_px'])

import re, sys, struct, cv2, numpy as np
data = open(sys.argv[1], 'rb').read(); out = sys.argv[2]
m = re.search(rb'42 0 obj\s*<<.*?/Length (\d+).*?>>stream\r?\n', data, re.S)
ln = int(m.group(1)); g4 = data[m.end():m.end()+ln]
W, H = 2344, 3307
tags = [(256,4,1,W),(257,4,1,H),(258,3,1,1),(259,3,1,4),(262,3,1,0),(273,4,1,0),(277,3,1,1),(278,4,1,H),(279,4,1,len(g4))]
hdr = b'II*\x00' + struct.pack('<I', 8); n = len(tags)
ifd_size = 2 + n*12 + 4; data_off = 8 + ifd_size
ifd = struct.pack('<H', n)
for tag,typ,cnt,val in tags:
    if tag == 273: val = data_off
    ifd += struct.pack('<HHI', tag, typ, cnt) + (struct.pack('<HH', val, 0) if typ == 3 else struct.pack('<I', val))
ifd += struct.pack('<I', 0)
open(out + '/scan.tif', 'wb').write(hdr + ifd + g4)
img = cv2.imread(out + '/scan.tif', cv2.IMREAD_GRAYSCALE)
print(img.shape, img.mean())
cv2.imwrite(out + '/scan.png', img)

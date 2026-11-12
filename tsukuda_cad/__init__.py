"""tsukuda_cad - turn a client's marked-up plan (photo / scan) into a clean DWG.

Pipeline (photo case):
    photo.py      upright + light normalisation, column detection, grid homography (photo px -> mm)
    vectorize.py  print ink -> regularised orthogonal lines, arcs (doors), columns
    markup.py     red handwriting -> located items, numbers checked against the geometry
    dxfout.py     clean DXF on standard layers (grid, columns, walls, doors, text, dims, markup)
    render.py     PNG previews and photo/CAD overlay for checking
"""

# -*- coding: utf-8 -*-
"""Convert the four paper PDF figures to 300-dpi PNGs (for main.tex)."""
import os
import fitz

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "论文部分", "figures")
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "论文部分")
os.makedirs(OUTDIR, exist_ok=True)

for name in ["fig_theory", "fig_performance", "fig_training", "fig_reconstructions"]:
    src = os.path.join(FIGDIR, name + ".pdf")
    dst = os.path.join(OUTDIR, name + ".png")
    assert os.path.exists(src), src
    doc = fitz.open(src)
    pix = doc[0].get_pixmap(dpi=300)
    pix.save(dst)
    doc.close()
    print("saved", dst, os.path.getsize(dst))
print("all PNGs done")

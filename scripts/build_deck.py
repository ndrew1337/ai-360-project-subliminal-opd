#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rebuild the SL_results deck from figures/.

Usage:  python scripts/build_deck.py [output.pptx]
Default output: ~/Downloads/SL_results.pptx
Run scripts/make_figures.py first to refresh the charts.
Each slide: header = the finding, chart subtitle = the distillation cell.
"""
import os
import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from PIL import Image

FIGDIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "figures"))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Downloads/SL_results.pptx")
SW, SH = 13.333, 7.5

p = Presentation(); p.slide_width = Inches(SW); p.slide_height = Inches(SH)
B = p.slide_layouts[6]


def hdr(s, t):
    tb = s.shapes.add_textbox(Inches(0.55), Inches(0.3), Inches(SW - 1.1), Inches(0.95))
    par = tb.text_frame.paragraphs[0]; tb.text_frame.word_wrap = True
    par.text = t; par.font.size = Pt(28); par.font.bold = True


def pic(s, img, box=(0.6, 1.45, SW - 1.2, 5.7)):
    bl, bt, bw, bh = box; iw, ih = Image.open(img).size; ia = iw / ih
    if ia >= bw / bh:
        w = bw; h = bw / ia
    else:
        h = bh; w = bh * ia
    s.shapes.add_picture(img, Inches(bl + (bw - w) / 2), Inches(bt + (bh - h) / 2), width=Inches(w))


def fig_slide(title, img):
    s = p.slides.add_slide(B); hdr(s, title); pic(s, os.path.join(FIGDIR, img))


# title
s = p.slides.add_slide(B)
tb = s.shapes.add_textbox(Inches(0.7), Inches(2.7), Inches(SW - 1.4), Inches(2))
tf = tb.text_frame; tf.word_wrap = True
par = tf.paragraphs[0]; par.text = "Subliminal learning via distillation — results"
par.font.size = Pt(40); par.font.bold = True
p2 = tf.add_paragraph()
p2.text = "gemma-3-4b-it · owl & raven · all numbers filtered (strictly subliminal)"
p2.font.size = Pt(20); p2.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

fig_slide("soft·fkl is the one channel that survives filtering", "distillation_matrix_4axis.png")
fig_slide("Off-policy transfer scales with unique data volume", "s_off.png")
fig_slide("On-policy: volume helps, prompt diversity flattens", "s_on.png")
fig_slide("Owl is volume-gated; raven transfers from the smallest budget", "s_collected.png")
fig_slide("Clipping (k32 support truncation) is free — both traits", "k_parity.png")
fig_slide("Owl concentrates in soft·fkl; raven transfers evenly", "x_channels.png")

# takeaways
s = p.slides.add_slide(B); hdr(s, "Takeaways")
tb = s.shapes.add_textbox(Inches(0.75), Inches(1.6), Inches(SW - 1.5), Inches(5.4))
tf = tb.text_frame; tf.word_wrap = True
items = [
    "Among filtered channels, only full-distribution forward-KL (soft·fkl) transfers — 0.78; others collapse to 0.22–0.34.",
    "Data volume drives off-policy transfer (0.19→0.93); iso-information controls are null; on-policy is weaker.",
    "Support truncation (k32) is free — ≈ full-vocab across every objective and both traits.",
    "Generality (raven): clipping & scaling shape hold; but raven has no null floor and transfers across all channels.",
    "Open: sequence-level RL reads null (0.068) — confounded by the KL-to-reference anchor; β-sweep pending.",
    "Note: raw (unfiltered) rollouts hit 0.9+ but leak 4–8% explicit mentions, so they're not strictly subliminal — all results above are filtered.",
]
for i, t in enumerate(items):
    par = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    par.text = "•  " + t; par.font.size = Pt(19); par.space_after = Pt(13)

p.save(OUT)
print("rebuilt %d slides -> %s" % (len(p.slides._sldIdLst), OUT))

#!/usr/bin/env python
"""Regenerate the title and abstract in paper/openreview_fields.md from paper/main.tex, with every
\\newcommand macro from numbers*.tex expanded and LaTeX markup stripped, so the OpenReview form text
matches the built PDF. usage: python paper/exp/refresh_openreview_abstract.py
"""
from __future__ import annotations

import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.abspath(os.path.join(HERE, ".."))


def load_macros():
    macros = {}
    for fn in ("numbers.tex", "numbers_story.tex", "numbers_coverage.tex", "numbers_denoise.tex"):
        p = os.path.join(PAPER, fn)
        if os.path.exists(p):
            for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*$", open(p, encoding="utf-8").read(), re.M):
                macros.setdefault(m.group(1), m.group(2))
    p = os.path.join(PAPER, "numbers_placeholder.tex")
    if os.path.exists(p):
        for m in re.finditer(r"\\providecommand\{\\(\w+)\}\{(.*)\}\s*$", open(p, encoding="utf-8").read(), re.M):
            macros.setdefault(m.group(1), m.group(2))
    return macros


def expand(text, macros):
    for _ in range(4):
        def sub(m):
            name = m.group(1)
            return macros.get(name, m.group(0))
        text = re.sub(r"\\([A-Za-z]+)(?:\{\})?", sub, text)
    return text


def strip_tex(t):
    t = t.replace("\\emph{", "").replace("\\textit{", "").replace("\\textbf{", "")
    t = re.sub(r"\\%", "%", t)
    t = t.replace("``", '"').replace("''", '"').replace("--", "-").replace("~", " ")
    t = re.sub(r"\$([^$]*)\$", r"\1", t)
    t = t.replace("\\ge", ">=").replace("\\le", "<=").replace("\\ne", "!=").replace("\\,", " ")
    t = re.sub(r"\\[A-Za-z]+\s?", "", t)
    t = t.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", t).strip()


def main():
    tex = open(os.path.join(PAPER, "main.tex"), encoding="utf-8").read()
    macros = load_macros()
    title = re.search(r"\\title\{(.*?)\}\s*$", tex, re.S | re.M).group(1).replace("\\\\", " ")
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S).group(1)
    title = strip_tex(expand(title, macros))
    abstract = strip_tex(expand(abstract, macros))
    p = os.path.join(PAPER, "openreview_fields.md")
    md = open(p, encoding="utf-8").read()
    md = re.sub(r"\*\*Title\.\*\* .*", "**Title.** " + title, md)
    md = re.sub(r"(\*\*Abstract\.\*\*[^\n]*\n\n)(.*?)(\n\n\*\*Keywords)", lambda m: m.group(1) + abstract + m.group(3), md, flags=re.S)
    open(p, "w", encoding="utf-8").write(md)
    left = re.findall(r"\\[A-Za-z]+", abstract)
    print("title:", title)
    print("abstract words:", len(abstract.split()), "| unexpanded macros:", left)


if __name__ == "__main__":
    main()

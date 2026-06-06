# Paper source

This folder contains the LaTeX source, references, paper figures, and paper tables.

Build from this directory:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The compiled public draft is copied to the repository root as `paper.pdf`.

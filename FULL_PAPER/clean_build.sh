#!/bin/bash

# Name of your main LaTeX file (without .tex extension)
DOC_NAME="full_paper"

# Remove old build artifacts
rm -f ${DOC_NAME}.{aux,bbl,blg,log,out,toc,lof,lot}

# First pass (generate .aux)
pdflatex -interaction=nonstopmode ${DOC_NAME}.tex

# Run BibTeX (generate .bbl)
bibtex ${DOC_NAME}

# Second pass (resolve references)
pdflatex -interaction=nonstopmode ${DOC_NAME}.tex

# Third pass (final cleanup)
pdflatex -interaction=nonstopmode ${DOC_NAME}.tex

# Optional: clean temporary files again
#rm -f ${DOC_NAME}.{aux,blg,log,out,toc,lof,lot}

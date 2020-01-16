
TEXPDF = pdflatex
BIBTEX = bibtex

.PHONY: clean

doc.pdf: doc.aux biblio.bib
	$(BIBTEX) doc.aux
	$(TEXPDF) doc.tex
	$(TEXPDF) doc.tex

doc.aux: doc.tex
	$(TEXPDF) doc.tex

clean:
	rm -f *.pdf *.bbl *.blg *.aux *.log

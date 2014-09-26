OUTPUT=raw_fixations.csv

.PHONY: raw_fixations

raw_fixations:
	head -n 1 raw/n1_nt1.csv > $(OUTPUT)
	for f in raw/*.csv; do tail -n +2 $$f >> $(OUTPUT); done

viewer:
	python -m SimpleHTTPServer

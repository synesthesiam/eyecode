OUTPUT=raw_fixations.csv

.PHONY: raw_fixations

raw_fixations:
	cp n1_nt1.csv $(OUTPUT)
	tail -n +2 n2_nt1.csv >> $(OUTPUT)
	tail -n +2 n3_nt1.csv >> $(OUTPUT)
	sed -ri 's/\t/,/g' $(OUTPUT)

viewer:
	python -m SimpleHTTPServer

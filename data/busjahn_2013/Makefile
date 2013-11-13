serve:
	python -m SimpleHTTPServer

deploy:
	rsync -av . hansenm@synesthesiam.com:eyecode.synesthesiam.com/stories/busjahn_2013/

code_image:
	pygmentize -f png -O font_name="DejaVu Sans Mono,line_numbers,font_size=18" -l java -o images/rectangle-pygment.png programs/rectangle.java
	pygmentize -f png -O font_name="Ubuntu Mono,line_numbers,font_size=13" -l java -o images/basketball-pygment.png programs/basketball.java

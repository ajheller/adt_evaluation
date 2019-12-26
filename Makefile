rsync:
	rsync --delete-after -avurP  plotly "ajh-dh:ambisonics/"

adt-pull:
	rsync --delete-after -avurP  \
		--include 'SCMD*.json' --exclude '*' \
		$(HOME)/Documents/adt/examples/ \
		examples/

plots:
	python3.7 rVrE.py
	python3.7 plotly_image.py 

clean:
	rm -rf *.pyc *.log

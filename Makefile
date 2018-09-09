rsync:
	rsync --delete-after -avurP  plotly "ajh-dh:ambisonics/"

clean:
	rm -rf *.pyc *.log

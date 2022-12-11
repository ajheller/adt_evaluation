.phony: rsync adt-pull plots clean git-push git-pull

rsync:
	rsync --delete-after -avurP  plotly "ajh-dh:ambisonics/"


rsync-reports:
	rsync --delete-after -avurP  reports "ajh-dh:ambisonics/reports"

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

git-push:
	for remote in `git remote`; do \
	  git push $$remote --all ; \
	  git push $$remote --tags; \
	done

git-pull:
	git pull --ff-only

requirements.txt:
	echo $@
#	pip freeze

conda-env:
	conda env export --from-history --no-build

activate:
	conda activate adt39

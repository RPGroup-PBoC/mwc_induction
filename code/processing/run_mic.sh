for dir in *Oid*microscopy/; do
	cd $dir
	python processing.py
	python analysis.py
	cd ../
done

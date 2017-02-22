for dir in *microscopy*/; do
	cd $dir
	NOM=$dir
	NOM=${NOM%/}_comments.txt
	echo $NOM
	cp comments.txt ../microscopy_comments/$NOM
	cd ../
done


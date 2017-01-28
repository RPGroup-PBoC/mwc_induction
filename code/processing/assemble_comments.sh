for dir in *IPTG_titration/; do
	cd $dir
	NOM=$dir
	NOM=${NOM%/}_comments.txt
	echo $NOM
	cp comments.txt ../flow_comments/$NOM
	cd ../
done


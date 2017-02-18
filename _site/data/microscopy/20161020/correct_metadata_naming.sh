# Some files were accidentally incorrectly named  and corrected in Windows,
# but this does not change the metadata embedded within the image. This 
# script corrects the improperly named files.
unset LANG
for dir in *auto_1000uMIPTG*/; do
	LC_ALL=C sed -i '' 's/RBS1027/auto/g' $dir/*
done



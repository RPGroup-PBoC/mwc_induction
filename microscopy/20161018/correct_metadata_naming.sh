# Some files were accidentally incorrectly named  and corrected in Windows,
# but this does not change the metadata embedded within the image. This 
# script corrects the improperly named files.
unset LANG
for dir in *delta_0uM*/; do
	LC_ALL=C sed -i '' 's/_RBS1027//g' $dir/*
done



imagesDir=$1 
inputImage=$2
outputName='reconstruction.jpg' 
python3 src/pca.py $imagesDir $inputImage $outputName 

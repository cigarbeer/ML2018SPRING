imagenpyPath=$1 
testCasePath=$2 
predictionFilePath=$3 
encoder_file='encoder.h5'

wget 'https://www.dropbox.com/s/pum3xyt676h530k/encoder.h5?dl=1' -O $encoder_file

python3 src/hw4.py $imagenpyPath $testCasePath $predictionFilePath $encoder_file


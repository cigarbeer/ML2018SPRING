testingData=$1 
predictionFile=$2 
mode=$3 

wget 'https://www.dropbox.com/s/9n6n0fakfwfdoyy/punc.final.hdf5?dl=1' -O 'punc.hdf5' 
wget 'https://www.dropbox.com/s/gumnc4wh9tvk0ji/hw5.punc.pickle?dl=1' -O 'hw5.pickle' 

# wget 'https://www.dropbox.com/s/fg9xdkde2mbsr7h/semi.final.hdf5?dl=1' -O 'semirnn.hdf5'
# wget 'https://www.dropbox.com/s/zw615x3k38k7ilz/hw5.semi.pickle?dl=1' -O 'hw5.pickle'

python3 punc_test.py $testingData $predictionFile 

# python3 semi_test.py $testingData $predictionFile 

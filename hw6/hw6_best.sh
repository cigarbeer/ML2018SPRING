testFilePath=$1 
preditionFilePath=$2 
movieFilePath=$3 
userFilePath=$4

wget 'https://www.dropbox.com/s/pvehixlhc9x73zz/ensemble.hdf5?dl=1' -O 'ensemble.hdf5' 

python3 best_test.py $testFilePath $preditionFilePath $movieFilePath $userFilePath 


testFilePath=$1 
preditionFilePath=$2 
movieFilePath=$3 
userFilePath=$4 

wget 'https://www.dropbox.com/s/rqmlzeixew5tn1l/matrix_factorization.hdf5?dl=1' -O 'matrix_factorization.hdf5' 

python3 matrix_factorize_test.py $testFilePath $preditionFilePath $movieFilePath $userFilePath 


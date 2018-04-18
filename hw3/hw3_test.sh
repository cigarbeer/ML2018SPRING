testing_data=$1 
prediction_file=$2 
mode=$3 

wget 'https://www.dropbox.com/s/4bjqpzyvlxj89r3/model.hdf5?dl=1' -O 'model.hdf5'

python3 test.py $testing_data $prediction_file 

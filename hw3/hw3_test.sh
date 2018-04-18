testing_data=$1 
prediction_file=$2 
mode=$3 

wget https://github.com/cigarbeer/ML2018SPRING/releases/download/v1.0/model.hdf5

python3 test.py $testing_data $prediction_file 

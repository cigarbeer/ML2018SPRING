#!/bin/bash
mkdir -p model
wget -O model/nonew3.h5 https://www.dropbox.com/s/8c55bmwprepy9ej/nonew3.h5?dl=1
wget -O model/res1.h5 https://www.dropbox.com/s/oufgb51022z2sfx/res1.h5?dl=1
wget -O model/tune3.h5 https://www.dropbox.com/s/xpz141xg182dwql/tune3.h5?dl=1
wget -O model/tune4.h5 https://www.dropbox.com/s/ql13q8h84qh968o/tune4.h5?dl=1
wget -O model/vgg192.h5 https://www.dropbox.com/s/8a9lzxdo44s8ejc/vgg192.h5?dl=1
wget -O model/iv32.h5 https://www.dropbox.com/s/nywtuyt5vhqm4zh/iv32.h5?dl=1
python3 src/test.py $1 $2 $3

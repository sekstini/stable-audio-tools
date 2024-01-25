wget -P data https://www.openslr.org/resources/141/$1.tar.gz
tar -xzf data/$1.tar.gz --directory="data"
rm data/$1.tar.gz

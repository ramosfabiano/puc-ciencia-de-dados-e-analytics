#!/bin/bash

#
# Busca dados originais
#
wget -i dados.txt

#
# Transforma *.csv.gz => *.csv.zip
#
for f in $(ls *.gz);
do
	gunzip "$f"
	rm -f "$f"
	f_no_extension="${f%.*}"
	zip "$f_no_extension".zip "$f_no_extension"
	rm -f "$f_no_extension"
done

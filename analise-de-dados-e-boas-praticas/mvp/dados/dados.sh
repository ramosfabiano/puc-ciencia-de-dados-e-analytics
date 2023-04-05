#!/bin/bash

wget -i dados.txt

for f in $(ls *.gz);
do
	gunzip "$f"
	rm -f "$f"
done

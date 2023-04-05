#!/usr/bin/python3

import csv
from io import TextIOWrapper
from zipfile import ZipFile

with ZipFile('listings.csv.zip') as zf:
    with zf.open('listings.csv', 'r') as infile:
        reader = csv.reader(TextIOWrapper(infile, 'utf-8'))
        for row in reader:
            print(row)



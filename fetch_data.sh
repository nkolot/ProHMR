#!/bin/bash

# Script that fetches all necessary data for training and eval
wget http://visiondata.cis.upenn.edu/prohmr/data.tar.gz
tar -xvf data.tar.gz
rm data.tar.gz

wget http://visiondata.cis.upenn.edu/prohmr/datasets.tar.gz
tar -xvf datasets.tar.gz
mv datasets ./data/
rm datasets.tar.gz

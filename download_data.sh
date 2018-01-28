#!/usr/bin/env bash

mkdir data
wget https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz -O ./data/rt-polaritydata.tar.gz
tar -zxvf ./data/rt-polaritydata.tar.gz -C ./data/

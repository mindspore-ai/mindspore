#!/bin/bash

# download dataset file to ./data/
DATA_URL=https://paddlenlp.bj.bcebos.com/datasets/DGU_datasets.tar.gz
wget --no-check-certificate ${DATA_URL}

tar -zxvf DGU_datasets.tar.gz

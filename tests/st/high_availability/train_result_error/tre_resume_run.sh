#!/bin/bash

export MS_ENABLE_TFT='{TRE:1}'

python "$(dirname $0)"/tre_train.py

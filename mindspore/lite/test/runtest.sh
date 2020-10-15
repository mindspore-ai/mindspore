#!/bin/bash
cd ./mindspore/lite/test/ || exit 1
if [ $1 == train ]; then
    echo 'run train ut tests'
    ./run_train_ut.sh
fi

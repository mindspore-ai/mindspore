#!/bin/bash
set -e 

CURRPATH=$(cd "$(dirname $0)"; pwd) 
cd ${CURRPATH}
if [ $1 == train ]; then
    echo 'run train ut tests'
    bash run_train_ut.sh
fi

if [ $1 == inference ]; then
    echo 'run inference ut tests'
    bash run_inference_ut.sh
fi

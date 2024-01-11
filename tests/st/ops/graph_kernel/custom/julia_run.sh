#!/bin/sh
export LD_LIBRARY_PATH=$JULIA_DIR/lib:$LD_LIBRARY_PATH
echo "JULIA_DIR"
echo $JULIA_DIR
ls $JULIA_DIR/lib
echo "LD_LIBRARY_PATH"
echo $LD_LIBRARY_PATH
pytest -m level3 julia_cases.py

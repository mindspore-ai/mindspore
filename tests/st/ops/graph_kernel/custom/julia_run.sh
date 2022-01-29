#!/bin/sh
export LD_LIBRARY_PATH=$JULIA_DIR/lib:$LD_LIBRARY_PATH
pytest -m level2 julia_cases.py

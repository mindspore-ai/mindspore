#!/bin/bash
cd ./ut/src/runtime/kernel/arm || exit 1
../../../../../../build/test/lite-test --gtest_filter=NetworkTest.efficient_net
# ../../../../../../build/test/lite-test --gtest_filter=NetworkTest.tuning_layer
# ../../../../../../build/test/lite-test --gtest_filter=NetworkTest.lenetnet
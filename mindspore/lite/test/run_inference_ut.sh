#!/bin/bash
cd ./ut/src/runtime/kernel/arm || exit 1
../../../../../../build/test/lite-test --gtest_filter=ControlFlowTest.TestMergeWhileModel

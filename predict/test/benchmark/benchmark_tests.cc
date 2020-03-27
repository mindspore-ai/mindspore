/**
 * Copyright 2019 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gtest/gtest.h>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include "test/test_context.h"
#include "benchmark/benchmark.h"

#define LENET_ARGS 2
#define MS_ARGS 4

namespace mindspore {
namespace predict {
class BenchmarkTest : public ::testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
  std::string root;
};

TEST_F(BenchmarkTest, BenchmarkRun) {
  const char* args[LENET_ARGS];
  args[0] = "./benchmark";
  args[1] = "--modelPath=./data/lenet/lenet.ms";

  int errorcode = mindspore::predict::RunBenchmark(LENET_ARGS, args);
  EXPECT_EQ(0, errorcode);
}

TEST_F(BenchmarkTest, LenetRun) {
  const char* args[MS_ARGS];
  args[0] = "./benchmark";
  args[1] = "--modelPath=./data/ms/mindspore.ms";
  args[2] = "--inDataPath=./data/ms/mindspore.bin";
  args[3] = "--calibDataPath=./data/ms/mindspore.out";

  int errorcode = mindspore::predict::RunBenchmark(MS_ARGS, args);
  EXPECT_EQ(0, errorcode);
}

TEST_F(BenchmarkTest, MindSporeRun) {
  const char* args[4];
  args[0] = "./benchmark";
  args[1] = "--modelPath=./data/lenet/lenet.ms";
  args[2] = "--inDataPath=./data/lenet/lenet.bin";
  args[3] = "--calibDataPath=./data/lenet/lenet.out";

  int errorcode = mindspore::predict::RunBenchmark(4, args);
  EXPECT_EQ(0, errorcode);
}
}  // namespace predict
}  // namespace mindspore

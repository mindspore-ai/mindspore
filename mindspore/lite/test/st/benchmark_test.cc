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
#include <string>
#include "common/common_test.h"
#include "benchmark/benchmark.h"

namespace mindspore {
namespace lite {
class BenchmarkTest : public mindspore::Common {
 public:
  BenchmarkTest() {}
};

TEST_F(BenchmarkTest, TestVideo) {
  const char *argv[] = {"./benchmark", "--modelPath=./models/hiai_label_and_video.ms"};
  auto status = RunBenchmark(2, argv);
  ASSERT_EQ(status, RET_OK);
}

TEST_F(BenchmarkTest, TestOCR_02) {
  const char *argv[] = {"./benchmark", "--modelPath=./models/hiai_cv_focusShootOCRMOdel_02.ms"};
  auto status = RunBenchmark(2, argv);
  ASSERT_EQ(status, RET_OK);
}

TEST_F(BenchmarkTest, TestHebing) {
  const char *argv[] = {"./benchmark", "--modelPath=./models/model_hebing_3branch.ms"};
  auto status = RunBenchmark(2, argv);
  ASSERT_EQ(status, RET_OK);
}
}  // namespace lite
}  // namespace mindspore


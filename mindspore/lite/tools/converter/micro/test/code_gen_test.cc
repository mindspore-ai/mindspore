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

#include "gtest/gtest.h"
#include "micro/coder/coder.h"

namespace mindspore::lite::micro::test {
TEST(GenerateCodeTest, mnist_x86) {
  const char *argv[] = {"./codegen", "--modelPath=../example/mnist/mnist.ms", "--moduleName=mnist", "--codePath=.",
                        "--isWeightFile"};
  STATUS status = RunCoder(5, argv);
  ASSERT_EQ(status, RET_OK);
}
}  // namespace mindspore::lite::micro::test

GTEST_API_ int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  int ret = RUN_ALL_TESTS();
  return ret;
}

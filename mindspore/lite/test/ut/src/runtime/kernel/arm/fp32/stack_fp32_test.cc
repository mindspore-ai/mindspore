/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "common/common_test.h"
#include "mindspore/lite/nnacl/base/stack_base.h"

namespace mindspore {
class StackTestFp32 : public mindspore::CommonTest {
 public:
  StackTestFp32() = default;
};

TEST_F(StackTestFp32, StackTest1) {
  float input0[6] = {1, 2, 3, 10, 20, 30};
  float input1[6] = {4, 5, 6, 40, 50, 60};
  float input2[6] = {7, 8, 9, 70, 80, 90};
  char *input[3];
  input[0] = reinterpret_cast<char *>(input0);
  input[1] = reinterpret_cast<char *>(input1);
  input[2] = reinterpret_cast<char *>(input2);
  std::vector<int> shape = {2, 3};
  constexpr int kOutSize = 18;
  float expect_out[kOutSize] = {1, 4, 7, 2, 5, 8, 3, 6, 9, 10, 40, 70, 20, 50, 80, 30, 60, 90};
  float output[kOutSize];
  Stack(input, reinterpret_cast<char *>(output), 3, 4, 6);
  for (float i : output) {
    std::cout << i << " ";
  }
  std::cout << "\n";
  ASSERT_EQ(0, CompareOutputData(output, expect_out, kOutSize, 0.000001));
}

}  // namespace mindspore

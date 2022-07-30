/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include <iostream>
#include "src/common/log_adapter.h"
#include "common/common_test.h"
#include "nnacl/fp32/arithmetic_fp32.h"
#include "mindspore/lite/src/litert/kernel_registry.h"
#include "mindspore/lite/src/litert/kernel_exec.h"

namespace mindspore {

class TestLogicalOrFp32 : public mindspore::CommonTest {
 public:
  TestLogicalOrFp32() {}
};

TEST_F(TestLogicalOrFp32, LogicalOrFp32) {
  float input[7] = {-3.2f, -0.12f, 0.24f, -0.0f, 0.0f, 0.52f, 6.0f};
  float input1[7] = {3.2f, 1.23f, -0.56f, 0.0f, 0.0f, 0.12f, -12.3f};
  float output[7]{0};
  ElementLogicalOr(input, input1, output, 7);
  float expect[7] = {1, 1, 1, 0, 0, 1, 1};
  for (int i = 0; i < 7; ++i) {
    ASSERT_EQ(output[i], expect[i]);
  }
}
}  // namespace mindspore

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
#include <iostream>
#include <memory>
#include "src/common/log_adapter.h"
#include "common/common_test.h"
#include "nnacl/fp32/space_to_batch_fp32.h"
#include "mindspore/lite/src/litert/kernel_registry.h"
#include "mindspore/lite/src/executor/kernel_exec.h"

namespace mindspore {

class SpaceToBatchTestFp32 : public mindspore::CommonTest {
 public:
  SpaceToBatchTestFp32() {}
};

TEST_F(SpaceToBatchTestFp32, SpaceToBatchTest4) {
  std::vector<float> input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  const size_t kOutSize = 16;
  std::vector<float> expect_out = {1, 2, 3, 4, 9, 10, 11, 12, 5, 6, 7, 8, 13, 14, 15, 16};
  float out[kOutSize];
  SpaceToBatchParameter param;
  param.op_parameter_.thread_num_ = 1;
  SetShape(param.input_shape_, {1, 4, 4, 1}, 4);
  SetShape(param.output_shape_, {2, 2, 4, 1}, 4);
  SetShape(param.in_stride_, {16, 4, 1, 1}, 4);
  SetShape(param.out_stride_, {8, 4, 1, 1}, 4);
  SetShape(param.block_sizes_, {2, 1}, 2);
  SetShape(param.paddings_, {0, 0, 0, 0}, 4);
  param.data_type_len = sizeof(float);

  DoSpaceToBatch(input.data(), out, &param, 0);
  ASSERT_EQ(0, CompareOutputData(out, expect_out.data(), kOutSize, 0.000001));
}

TEST_F(SpaceToBatchTestFp32, SpaceToBatchTest5) {
  std::vector<float> input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  size_t kOutSize = 16;
  std::vector<float> expect_out = {1, 3, 5, 7, 9, 11, 13, 15, 2, 4, 6, 8, 10, 12, 14, 16};
  float out[kOutSize];

  SpaceToBatchParameter param;
  param.op_parameter_.thread_num_ = 1;
  SetShape(param.input_shape_, {1, 4, 4, 1}, 4);
  SetShape(param.output_shape_, {2, 4, 2, 1}, 4);
  SetShape(param.in_stride_, {16, 4, 1, 1}, 4);
  SetShape(param.out_stride_, {8, 2, 1, 1}, 4);
  SetShape(param.block_sizes_, {1, 2}, 2);
  SetShape(param.paddings_, {0, 0, 0, 0}, 4);
  param.data_type_len = sizeof(float);

  DoSpaceToBatch(input.data(), out, &param, 0);
  ASSERT_EQ(0, CompareOutputData(out, expect_out.data(), kOutSize, 0.000001));
}

TEST_F(SpaceToBatchTestFp32, SpaceToBatchTest6) {
  std::vector<float> input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  size_t kOutSize = 16;
  std::vector<float> expect_out = {1, 3, 9, 11, 2, 4, 10, 12, 5, 7, 13, 15, 6, 8, 14, 16};
  float out[kOutSize];

  SpaceToBatchParameter param;
  param.op_parameter_.thread_num_ = 1;
  SetShape(param.input_shape_, {1, 4, 4, 1}, 4);
  SetShape(param.output_shape_, {4, 2, 2, 1}, 4);
  SetShape(param.in_stride_, {16, 4, 1, 1}, 4);
  SetShape(param.out_stride_, {4, 2, 1, 1}, 4);
  SetShape(param.block_sizes_, {2, 2}, 2);
  SetShape(param.paddings_, {0, 0, 0, 0}, 4);
  param.data_type_len = sizeof(float);

  DoSpaceToBatch(input.data(), out, &param, 0);
  ASSERT_EQ(0, CompareOutputData(out, expect_out.data(), kOutSize, 0.000001));
}
}  // namespace mindspore

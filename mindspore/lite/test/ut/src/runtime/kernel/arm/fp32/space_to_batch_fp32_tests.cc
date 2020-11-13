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
#include "mindspore/lite/nnacl/fp32/space_to_batch.h"
#include "mindspore/lite/src/kernel_registry.h"
#include "mindspore/lite/src/lite_kernel.h"

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
  std::vector<int> in_shape = {1, 4, 4, 1};
  std::vector<int> out_shape = {2, 2, 4, 1};
  SpaceToBatchParameter param;
  param.block_sizes_[0] = 2;
  param.block_sizes_[1] = 1;
  DoSpaceToBatchNHWC(input.data(), out, param.block_sizes_, in_shape.data(), out_shape.data());
  for (float i : out) {
    std::cout << i << " ";
  }
  std::cout << "\n";
  ASSERT_EQ(0, CompareOutputData(out, expect_out.data(), kOutSize, 0.000001));
}

TEST_F(SpaceToBatchTestFp32, SpaceToBatchTest5) {
  std::vector<float> input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  size_t kOutSize = 16;
  std::vector<float> expect_out = {1, 3, 5, 7, 9, 11, 13, 15, 2, 4, 6, 8, 10, 12, 14, 16};
  float out[kOutSize];
  std::vector<int> in_shape = {1, 4, 4, 1};
  std::vector<int> out_shape = {2, 4, 2, 1};
  SpaceToBatchParameter param;
  param.block_sizes_[0] = 1;
  param.block_sizes_[1] = 2;
  DoSpaceToBatchNHWC(input.data(), out, param.block_sizes_, in_shape.data(), out_shape.data());
  for (unsigned int i = 0; i < kOutSize; ++i) {
    std::cout << out[i] << " ";
  }
  std::cout << "\n";
  ASSERT_EQ(0, CompareOutputData(out, expect_out.data(), kOutSize, 0.000001));
}

TEST_F(SpaceToBatchTestFp32, SpaceToBatchTest6) {
  std::vector<float> input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  size_t kOutSize = 16;
  std::vector<float> expect_out = {1, 3, 9, 11, 2, 4, 10, 12, 5, 7, 13, 15, 6, 8, 14, 16};
  float out[kOutSize];
  std::vector<int> in_shape = {1, 4, 4, 1};
  std::vector<int> out_shape = {4, 2, 2, 1};
  SpaceToBatchParameter param;
  param.block_sizes_[0] = 2;
  param.block_sizes_[1] = 2;
  DoSpaceToBatchNHWC(input.data(), out, param.block_sizes_, in_shape.data(), out_shape.data());
  for (unsigned int i = 0; i < kOutSize; ++i) {
    std::cout << out[i] << " ";
  }
  std::cout << "\n";
  ASSERT_EQ(0, CompareOutputData(out, expect_out.data(), kOutSize, 0.000001));
}

TEST_F(SpaceToBatchTestFp32, SpaceToBatchTest7) {
  std::vector<float> input = {1,  11, 2,  12,  3,  13,  4,  14,  5,  15,  6,  16,  7,  17,  8,  18,
                              9,  19, 10, 110, 11, 111, 12, 112, 10, 11,  20, 12,  30, 13,  40, 14,
                              50, 15, 60, 16,  70, 17,  80, 18,  13, 113, 14, 114, 15, 115, 16, 116};
  size_t kOutSize = 48;
  std::vector<float> expect_out = {1,  11,  3,  13,  9,  19, 11, 111, 50, 15, 70, 17, 2,  12,  4,  14,
                                   10, 110, 12, 112, 60, 16, 80, 18,  5,  15, 7,  17, 10, 11,  30, 13,
                                   13, 113, 15, 115, 6,  16, 8,  18,  20, 12, 40, 14, 14, 114, 16, 116};
  float out[kOutSize];
  std::vector<int> in_shape = {1, 6, 4, 2};
  std::vector<int> out_shape = {4, 3, 2, 2};
  SpaceToBatchParameter param;
  param.block_sizes_[0] = 2;
  param.block_sizes_[1] = 2;
  DoSpaceToBatchNHWC(input.data(), out, param.block_sizes_, in_shape.data(), out_shape.data());
  for (unsigned int i = 0; i < kOutSize; ++i) {
    std::cout << out[i] << " ";
  }
  std::cout << "\n";
  ASSERT_EQ(0, CompareOutputData(out, expect_out.data(), kOutSize, 0.000001));
}

TEST_F(SpaceToBatchTestFp32, SpaceToBatchTest8) {
  std::vector<float> input = {1, -1, 2,  -2,  3,  -3,  4,  -4,  5,  -5,  6,  -6,  7,  -7,  8,  -8,
                              9, -9, 10, -10, 11, -11, 12, -12, 13, -13, 14, -14, 15, -15, 16, -16};
  std::vector<float> expect_out = {1,  -1,  2,  -2,  3,  -3, 4,   -4, 0,   0,  5,   -5, 6, -6, 7,   -7, 8,
                                   -8, 0,   0,  9,   -9, 10, -10, 11, -11, 12, -12, 0,  0, 13, -13, 14, -14,
                                   15, -15, 16, -16, 0,  0,  0,   0,  0,   0,  0,   0,  0, 0,  0,   0};
  size_t kOutSize = 50;
  float out[kOutSize];
  std::vector<int> in_shape = {1, 4, 4, 2};
  std::vector<int> out_shape = {1, 5, 5, 2};
  std::vector<int> padding = {0, 1, 0, 1};
  DoSpaceToBatchPaddingNHWC(input.data(), out, in_shape.data(), padding.data(), out_shape.data());
  for (unsigned int i = 0; i < kOutSize; ++i) {
    std::cout << out[i] << " ";
  }
  std::cout << "\n";
  ASSERT_EQ(0, CompareOutputData(out, expect_out.data(), kOutSize, 0.000001));
}

TEST_F(SpaceToBatchTestFp32, SpaceToBatchTest9) {
  std::vector<float> input = {1, -1, 2,  -2,  3,  -3,  4,  -4,  5,  -5,  6,  -6,  7,  -7,  8,  -8,
                              9, -9, 10, -10, 11, -11, 12, -12, 13, -13, 14, -14, 15, -15, 16, -16};
  std::vector<float> expect_out = {0,  0,   0,  0,   0,  0,   0,  0,   0,  0,   0, 0,  0, 0,  1,  -1,  2,  -2,
                                   3,  -3,  4,  -4,  0,  0,   0,  0,   5,  -5,  6, -6, 7, -7, 8,  -8,  0,  0,
                                   0,  0,   9,  -9,  10, -10, 11, -11, 12, -12, 0, 0,  0, 0,  13, -13, 14, -14,
                                   15, -15, 16, -16, 0,  0,   0,  0,   0,  0,   0, 0,  0, 0,  0,  0,   0,  0};
  size_t kOutSize = 72;
  float out[kOutSize];
  std::vector<int> in_shape = {1, 4, 4, 2};
  std::vector<int> out_shape = {1, 6, 6, 2};
  std::vector<int> padding = {1, 1, 1, 1};
  DoSpaceToBatchPaddingNHWC(input.data(), out, in_shape.data(), padding.data(), out_shape.data());
  for (unsigned int i = 0; i < kOutSize; ++i) {
    std::cout << out[i] << " ";
  }
  std::cout << "\n";
  ASSERT_EQ(0, CompareOutputData(out, expect_out.data(), kOutSize, 0.000001));
}

TEST_F(SpaceToBatchTestFp32, SpaceToBatchTest10) {
  std::vector<float> input = {1, -1, 2,  -2,  3,  -3,  4,  -4,  5,  -5,  6,  -6,  7,  -7,  8,  -8,
                              9, -9, 10, -10, 11, -11, 12, -12, 13, -13, 14, -14, 15, -15, 16, -16};
  std::vector<float> expect_out = {0, 0,  0, 0,  0, 0,  0, 0,  6,  -6,  8,  -8,  0,  0,   14, -14, 16, -16,
                                   0, 0,  0, 0,  0, 0,  5, -5, 7,  -7,  0,  0,   13, -13, 15, -15, 0,  0,
                                   0, 0,  2, -2, 4, -4, 0, 0,  10, -10, 12, -12, 0,  0,   0,  0,   0,  0,
                                   1, -1, 3, -3, 0, 0,  9, -9, 11, -11, 0,  0,   0,  0,   0,  0,   0,  0};
  size_t kOutSize = 72;
  float out[kOutSize];
  float pedding_out[kOutSize];
  std::vector<int> in_shape = {1, 4, 4, 2};
  std::vector<int> pedding_out_shape = {1, 6, 6, 2};
  std::vector<int> out_shape = {4, 3, 3, 2};
  std::vector<int> padding = {1, 1, 1, 1};
  DoSpaceToBatchPaddingNHWC(input.data(), pedding_out, in_shape.data(), padding.data(), pedding_out_shape.data());
  SpaceToBatchParameter param;
  param.block_sizes_[0] = 2;
  param.block_sizes_[1] = 2;
  DoSpaceToBatchNHWC(pedding_out, out, param.block_sizes_, pedding_out_shape.data(), out_shape.data());
  for (unsigned int i = 0; i < kOutSize; ++i) {
    std::cout << out[i] << " ";
  }
  std::cout << "\n";
  ASSERT_EQ(0, CompareOutputData(out, expect_out.data(), kOutSize, 0.000001));
}
}  // namespace mindspore

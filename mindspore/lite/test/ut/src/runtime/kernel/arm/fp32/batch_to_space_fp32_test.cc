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
#include "src/common/log_adapter.h"
#include "common/common_test.h"
#include "mindspore/lite/nnacl/base/batch_to_space_base.h"
#include "mindspore/lite/nnacl/batch_to_space.h"
#include "mindspore/lite/nnacl/common_func.h"

namespace mindspore {

class BatchToSpaceTestFp32 : public mindspore::CommonTest {
 public:
  BatchToSpaceTestFp32() = default;
};

TEST_F(BatchToSpaceTestFp32, BatchToSpaceTest1) {
  float input[12] = {10, 30, 90, 2, 20, 120, 5, 50, 150, 6, 16, 160};
  constexpr int kOutSize = 12;
  float expect_out[kOutSize] = {10, 30, 90, 2, 20, 120, 5, 50, 150, 6, 16, 160};

  float output[kOutSize];
  int in_shape[4] = {4, 1, 1, 3};
  int out_n = 1;
  int block[2] = {2, 2};
  BatchToSpaceNoCropForNHWC(input, output, in_shape, out_n, block, sizeof(float));
  for (int i = 0; i < kOutSize; ++i) {
    std::cout << output[i] << " ";
  }
  std::cout << "\n";
  ASSERT_EQ(0, CompareOutputData(output, expect_out, kOutSize, 0.000001));
}

TEST_F(BatchToSpaceTestFp32, BatchToSpaceTest_crop_1) {
  float input[12] = {10, 30, 90, 2, 20, 120, 5, 50, 150, 6, 16, 160};
  constexpr int kOutSize = 3;
  float expect_out[kOutSize] = {5, 50, 150};

  float output[kOutSize];
  int in_shape[4] = {4, 1, 1, 3};
  int out_n = 1;
  int block[2] = {2, 2};
  int crops[4] = {1, 0, 0, 1};
  BatchToSpaceForNHWC(input, output, in_shape, out_n, block, crops, sizeof(float));
  for (int i = 0; i < kOutSize; ++i) {
    std::cout << output[i] << " ";
  }
  std::cout << "\n";
  ASSERT_EQ(0, CompareOutputData(output, expect_out, kOutSize, 0.000001));
}

TEST_F(BatchToSpaceTestFp32, BatchToSpaceTest2) {
  float input[32] = {1, 10, 3, 30, 9,  90,  11, 110, 2, 20, 4, 40, 10, 100, 12, 120,
                     5, 50, 7, 70, 13, 130, 15, 150, 6, 60, 8, 80, 14, 140, 16, 160};
  constexpr int kOutSize = 32;
  float expect_out[kOutSize] = {1, 10, 2,  20,  3,  30,  4,  40,  5,  50,  6,  60,  7,  70,  8,  80,
                                9, 90, 10, 100, 11, 110, 12, 120, 13, 130, 14, 140, 15, 150, 16, 160};

  float output[kOutSize];
  int in_shape[4] = {4, 2, 2, 2};
  int out_n = 1;
  int block[2] = {2, 2};
  BatchToSpaceNoCropForNHWC(input, output, in_shape, out_n, block, sizeof(float));
  for (int i = 0; i < kOutSize; ++i) {
    std::cout << output[i] << " ";
  }
  std::cout << "\n";
  ASSERT_EQ(0, CompareOutputData(output, expect_out, kOutSize, 0.000001));
}

TEST_F(BatchToSpaceTestFp32, BatchToSpaceTest_crop_2) {
  float input[32] = {1, 10, 3, 30, 9,  90,  11, 110, 2, 20, 4, 40, 10, 100, 12, 120,
                     5, 50, 7, 70, 13, 130, 15, 150, 6, 60, 8, 80, 14, 140, 16, 160};
  constexpr int kOutSize = 12;
  float expect_out[kOutSize] = {6, 60, 7, 70, 8, 80, 10, 100, 11, 110, 12, 120};

  float output[kOutSize];
  int in_shape[4] = {4, 2, 2, 2};
  int out_n = 1;
  int block[2] = {2, 2};
  int crops[4] = {1, 1, 1, 0};
  BatchToSpaceForNHWC(input, output, in_shape, out_n, block, crops, sizeof(float));
  for (int i = 0; i < kOutSize; ++i) {
    std::cout << output[i] << " ";
  }
  std::cout << "\n";
  ASSERT_EQ(0, CompareOutputData(output, expect_out, kOutSize, 0.000001));
}

TEST_F(BatchToSpaceTestFp32, BatchToSpaceTest3) {
  float input[64] = {1,  10, 3,  30, 9,   90,  11,  110, 2,  20, 4,  40, 10,  100, 12,  120,
                     5,  50, 7,  70, 13,  130, 15,  150, 6,  60, 8,  80, 14,  140, 16,  160,
                     21, 10, 23, 30, 29,  90,  211, 110, 22, 20, 24, 40, 210, 100, 212, 120,
                     25, 50, 27, 70, 213, 130, 215, 150, 26, 60, 28, 80, 214, 140, 216, 160};
  constexpr int kOutSize = 64;
  float expect_out[kOutSize] = {1,  10,  5,  50,  3,  30,  7,  70,  21,  10,  25,  50,  23,  30,  27,  70,
                                9,  90,  13, 130, 11, 110, 15, 150, 29,  90,  213, 130, 211, 110, 215, 150,
                                2,  20,  6,  60,  4,  40,  8,  80,  22,  20,  26,  60,  24,  40,  28,  80,
                                10, 100, 14, 140, 12, 120, 16, 160, 210, 100, 214, 140, 212, 120, 216, 160};

  float output[kOutSize];
  int in_shape[4] = {8, 2, 2, 2};
  int out_n = 2;
  int block[2] = {2, 2};
  BatchToSpaceNoCropForNHWC(input, output, in_shape, out_n, block, sizeof(float));
  for (int i = 0; i < kOutSize && i < 32; ++i) {
    std::cout << output[i] << " ";
  }
  std::cout << "\n";
  ASSERT_EQ(0, CompareOutputData(output, expect_out, kOutSize, 0.000001));
}

TEST_F(BatchToSpaceTestFp32, BatchToSpaceTest_crop_3) {
  float input[64] = {1,  10, 3,  30, 9,   90,  11,  110, 2,  20, 4,  40, 10,  100, 12,  120,
                     5,  50, 7,  70, 13,  130, 15,  150, 6,  60, 8,  80, 14,  140, 16,  160,
                     21, 10, 23, 30, 29,  90,  211, 110, 22, 20, 24, 40, 210, 100, 212, 120,
                     25, 50, 27, 70, 213, 130, 215, 150, 26, 60, 28, 80, 214, 140, 216, 160};
  constexpr int kOutSize = 16;
  float expect_out[kOutSize] = {9, 90, 13, 130, 29, 90, 213, 130, 10, 100, 14, 140, 210, 100, 214, 140};

  float output[kOutSize];
  int in_shape[4] = {8, 2, 2, 2};
  int out_n = 2;
  int block[2] = {2, 2};
  int crops[4] = {2, 0, 0, 2};
  BatchToSpaceForNHWC(input, output, in_shape, out_n, block, crops, sizeof(float));
  for (int i = 0; i < kOutSize && i < 32; ++i) {
    std::cout << output[i] << " ";
  }
  std::cout << "\n";
  ASSERT_EQ(0, CompareOutputData(output, expect_out, kOutSize, 0.000001));
}

TEST_F(BatchToSpaceTestFp32, BatchToSpaceTest4) {
  float input[96] = {1,   10,  3,   30,  9,   90,  11,  110, 2,  20,  4,   40,  10,  100, 12,  120, 5,   50,  7,   70,
                     13,  130, 15,  150, 6,   60,  8,   80,  14, 140, 16,  160, 21,  10,  23,  30,  29,  90,  211, 110,
                     22,  20,  24,  40,  210, 100, 212, 120, 25, 50,  27,  70,  213, 130, 215, 150, 26,  60,  28,  80,
                     214, 140, 216, 160, 31,  10,  33,  30,  39, 90,  311, 110, 32,  20,  34,  40,  310, 100, 312, 120,
                     35,  50,  37,  70,  313, 130, 315, 150, 36, 60,  38,  80,  314, 140, 316, 160};
  constexpr int kOutSize = 96;
  float expect_out[kOutSize] = {
    1,  10,  5,  50,  3,  30,  7,  70,  21,  10,  25,  50,  23,  30,  27,  70,  31,  10,  35,  50,  33,  30,  37,  70,
    9,  90,  13, 130, 11, 110, 15, 150, 29,  90,  213, 130, 211, 110, 215, 150, 39,  90,  313, 130, 311, 110, 315, 150,
    2,  20,  6,  60,  4,  40,  8,  80,  22,  20,  26,  60,  24,  40,  28,  80,  32,  20,  36,  60,  34,  40,  38,  80,
    10, 100, 14, 140, 12, 120, 16, 160, 210, 100, 214, 140, 212, 120, 216, 160, 310, 100, 314, 140, 312, 120, 316, 160};

  float output[kOutSize];
  int in_shape[4] = {12, 2, 2, 2};
  int out_n = 2;
  int block[2] = {3, 2};
  BatchToSpaceNoCropForNHWC(input, output, in_shape, out_n, block, sizeof(float));
  for (int i = 0; i < kOutSize && i < 32; ++i) {
    std::cout << output[i] << " ";
  }
  std::cout << "\n";
  ASSERT_EQ(0, CompareOutputData(output, expect_out, kOutSize, 0.000001));
}

TEST_F(BatchToSpaceTestFp32, BatchToSpaceTest_crop_4) {
  float input[96] = {1,   10,  3,   30,  9,   90,  11,  110, 2,  20,  4,   40,  10,  100, 12,  120, 5,   50,  7,   70,
                     13,  130, 15,  150, 6,   60,  8,   80,  14, 140, 16,  160, 21,  10,  23,  30,  29,  90,  211, 110,
                     22,  20,  24,  40,  210, 100, 212, 120, 25, 50,  27,  70,  213, 130, 215, 150, 26,  60,  28,  80,
                     214, 140, 216, 160, 31,  10,  33,  30,  39, 90,  311, 110, 32,  20,  34,  40,  310, 100, 312, 120,
                     35,  50,  37,  70,  313, 130, 315, 150, 36, 60,  38,  80,  314, 140, 316, 160};
  constexpr int kOutSize = 24;
  float expect_out[kOutSize] = {25, 50, 23, 30, 35, 50, 33, 30, 13, 130, 11, 110,
                                26, 60, 24, 40, 36, 60, 34, 40, 14, 140, 12, 120};

  float output[kOutSize];
  int in_shape[4] = {12, 2, 2, 2};
  int out_n = 2;
  int block[2] = {3, 2};
  int crops[4] = {1, 2, 1, 1};
  BatchToSpaceForNHWC(input, output, in_shape, out_n, block, crops, sizeof(float));
  for (int i = 0; i < kOutSize && i < 32; ++i) {
    std::cout << output[i] << " ";
  }
  std::cout << "\n";
  ASSERT_EQ(0, CompareOutputData(output, expect_out, kOutSize, 0.000001));
}

}  // namespace mindspore

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
#include "mindspore/lite/nnacl/fp32/crop_fp32.h"
#include "mindspore/lite/src/runtime/kernel/arm/fp32/crop_fp32.h"

namespace mindspore {
class CropTestFp32 : public mindspore::CommonTest {
 public:
  CropTestFp32() = default;
};

TEST_F(CropTestFp32, CropTest1) {
  float input[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  const int kOutSize = 2;
  float expect_out[kOutSize] = {8, 16};

  float output[kOutSize];
  int in_shape[4] = {2, 2, 2, 2};
  int out_shape[4] = {2, 1, 1, 1};
  CropParameter crop_param;
  crop_param.axis_ = 1;
  crop_param.offset_[0] = 1;
  crop_param.offset_[1] = 1;
  crop_param.offset_[2] = 1;
  crop_param.op_parameter_.thread_num_ = 1;
  Crop4D(input, output, in_shape, out_shape, &crop_param, 0);
  for (int i = 0; i < kOutSize; ++i) {
    std::cout << output[i] << " ";
  }
  std::cout << "\n";
  ASSERT_EQ(0, CompareOutputData(output, expect_out, kOutSize, 0.000001));
}

TEST_F(CropTestFp32, CropTest2) {
  float input[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  const int kOutSize = 4;
  float expect_out[kOutSize] = {13, 14, 15, 16};

  float output[kOutSize];
  int in_shape[4] = {2, 2, 2, 2};
  int out_shape[4] = {1, 1, 2, 2};
  CropParameter crop_param;
  crop_param.axis_ = 0;
  crop_param.offset_[0] = 1;
  crop_param.offset_[1] = 1;
  crop_param.offset_[2] = 0;
  crop_param.offset_[3] = 0;
  crop_param.op_parameter_.thread_num_ = 1;
  Crop4D(input, output, in_shape, out_shape, &crop_param, 0);
  for (int i = 0; i < kOutSize; ++i) {
    std::cout << output[i] << " ";
  }
  std::cout << "\n";
  ASSERT_EQ(0, CompareOutputData(output, expect_out, kOutSize, 0.000001));
}

TEST_F(CropTestFp32, CropTest3) {
  float input[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  const int kOutSize = 8;
  float expect_out[kOutSize] = {2, 4, 6, 8, 10, 12, 14, 16};

  float output[kOutSize];
  int in_shape[4] = {2, 2, 2, 2};
  int out_shape[4] = {2, 2, 2, 1};
  CropParameter crop_param;
  crop_param.axis_ = 3;
  crop_param.offset_[0] = 1;
  crop_param.op_parameter_.thread_num_ = 1;
  Crop4D(input, output, in_shape, out_shape, &crop_param, 0);
  for (int i = 0; i < kOutSize; ++i) {
    std::cout << output[i] << " ";
  }
  std::cout << "\n";
  ASSERT_EQ(0, CompareOutputData(output, expect_out, kOutSize, 0.000001));
}

TEST_F(CropTestFp32, CropTest4) {
  float input[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  const int kOutSize = 8;
  float expect_out[kOutSize] = {2, 4, 6, 8, 10, 12, 14, 16};

  float output[kOutSize];
  int in_shape[4] = {2, 2, 2, 2};
  int out_shape[4] = {2, 2, 2, 1};
  CropParameter crop_param;
  crop_param.axis_ = 3;
  crop_param.offset_[0] = 1;
  crop_param.op_parameter_.thread_num_ = 2;
  Crop4D(input, output, in_shape, out_shape, &crop_param, 0);
  Crop4D(input, output, in_shape, out_shape, &crop_param, 1);
  for (int i = 0; i < kOutSize; ++i) {
    std::cout << output[i] << " ";
  }
  std::cout << "\n";
  ASSERT_EQ(0, CompareOutputData(output, expect_out, kOutSize, 0.000001));
}

TEST_F(CropTestFp32, CropTest5) {
  float input[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  const int kOutSize = 2;
  float expect_out[kOutSize] = {8, 16};

  float output[kOutSize];
  int in_shape[4] = {2, 2, 2, 2};
  int out_shape[4] = {2, 1, 1, 1};
  CropParameter crop_param;
  crop_param.axis_ = 1;
  crop_param.offset_[0] = 1;
  crop_param.offset_[1] = 1;
  crop_param.offset_[2] = 1;
  Crop4DNoParallel(input, output, in_shape, out_shape, &crop_param);
  for (int i = 0; i < kOutSize; ++i) {
    std::cout << output[i] << " ";
  }
  std::cout << "\n";
  ASSERT_EQ(0, CompareOutputData(output, expect_out, kOutSize, 0.000001));
}

TEST_F(CropTestFp32, CropTest6) {
  float input[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  const int kOutSize = 4;
  float expect_out[kOutSize] = {13, 14, 15, 16};

  float output[kOutSize];
  int in_shape[4] = {2, 2, 2, 2};
  int out_shape[4] = {1, 1, 2, 2};
  CropParameter crop_param;
  crop_param.axis_ = 0;
  crop_param.offset_[0] = 1;
  crop_param.offset_[1] = 1;
  crop_param.offset_[2] = 0;
  crop_param.offset_[3] = 0;
  Crop4DNoParallel(input, output, in_shape, out_shape, &crop_param);
  for (float i : output) {
    std::cout << i << " ";
  }
  std::cout << "\n";
  ASSERT_EQ(0, CompareOutputData(output, expect_out, kOutSize, 0.000001));
}

TEST_F(CropTestFp32, CropTest7) {
  float input[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  const int kOutSize = 8;
  float expect_out[kOutSize] = {2, 4, 6, 8, 10, 12, 14, 16};

  float output[kOutSize];
  int in_shape[4] = {2, 2, 2, 2};
  int out_shape[4] = {2, 2, 2, 1};
  CropParameter crop_param;
  crop_param.axis_ = 3;
  crop_param.offset_[0] = 1;
  Crop4DNoParallel(input, output, in_shape, out_shape, &crop_param);
  for (float i : output) {
    std::cout << i << " ";
  }
  std::cout << "\n";
  ASSERT_EQ(0, CompareOutputData(output, expect_out, kOutSize, 0.000001));
}

TEST_F(CropTestFp32, CropTest8) {
  float input[27] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29};
  const int kOutSize = 4;
  float expect_out[kOutSize] = {15, 16, 18, 19};

  float output[kOutSize];
  int in_shape[4] = {1, 3, 3, 3};
  int out_shape[4] = {1, 1, 2, 2};
  CropParameter crop_param;
  crop_param.axis_ = 1;
  crop_param.offset_[0] = 1;
  crop_param.offset_[1] = 1;
  crop_param.offset_[2] = 1;
  crop_param.op_parameter_.thread_num_ = 2;
  Crop4D(input, output, in_shape, out_shape, &crop_param, 0);
  Crop4D(input, output, in_shape, out_shape, &crop_param, 1);
  for (float i : output) {
    std::cout << i << " ";
  }
  std::cout << "\n";
  ASSERT_EQ(0, CompareOutputData(output, expect_out, kOutSize, 0.000001));
}

TEST_F(CropTestFp32, CropTest9) {
  float input[64] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10,  11,  12,  13,  14,  15,  16,
                     11, 12, 13, 14, 15, 16, 17, 18, 19, 110, 111, 112, 113, 114, 115, 116,
                     21, 22, 23, 24, 25, 26, 27, 28, 29, 210, 211, 212, 213, 214, 215, 216,
                     31, 32, 33, 34, 35, 36, 37, 38, 39, 310, 311, 312, 313, 314, 315, 316};
  const int kOutSize = 8;
  float expect_out[kOutSize] = {16, 17, 110, 111, 26, 27, 210, 211};

  float output[kOutSize];
  int in_shape[4] = {1, 4, 4, 4};
  int out_shape[4] = {1, 2, 2, 2};
  CropParameter crop_param;
  crop_param.axis_ = 1;
  crop_param.offset_[0] = 1;
  crop_param.offset_[1] = 1;
  crop_param.offset_[2] = 1;
  crop_param.op_parameter_.thread_num_ = 2;
  Crop4D(input, output, in_shape, out_shape, &crop_param, 0);
  Crop4D(input, output, in_shape, out_shape, &crop_param, 1);
  for (float i : output) {
    std::cout << i << " ";
  }
  std::cout << "\n";
  ASSERT_EQ(0, CompareOutputData(output, expect_out, kOutSize, 0.000001));
}

TEST_F(CropTestFp32, CropTest10) {
  float input[50] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
                     18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                     35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50};
  const int kOutSize = 8;
  float expect_out[kOutSize] = {1, 2, 6, 7, 26, 27, 31, 32};

  float output[kOutSize];
  int in_shape[4] = {1, 2, 5, 5};
  int out_shape[4] = {1, 2, 2, 2};
  CropParameter crop_param;
  crop_param.axis_ = 2;
  crop_param.offset_[0] = 0;
  crop_param.offset_[1] = 0;
  crop_param.op_parameter_.thread_num_ = 2;
  Crop4D(input, output, in_shape, out_shape, &crop_param, 1);
  Crop4D(input, output, in_shape, out_shape, &crop_param, 0);
  for (float i : output) {
    std::cout << i << " ";
  }
  std::cout << "\n";
  ASSERT_EQ(0, CompareOutputData(output, expect_out, kOutSize, 0.000001));
}

TEST_F(CropTestFp32, CropTest11) {
  float input[100] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                      21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                      41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                      11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                      31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50};

  const int kOutSize = 16;
  float expect_out[kOutSize] = {1, 2, 6, 7, 26, 27, 31, 32, 1, 2, 6, 7, 26, 27, 31, 32};
  std::vector<int> in_shape = {1, 4, 5, 5};
  std::vector<int> out_shape = {1, 4, 2, 2};
  std::vector<lite::Tensor *> inputs;
  std::vector<lite::Tensor *> outputs;
  auto in_t = new lite::Tensor(kNumberTypeFloat, in_shape, schema::Format_NHWC, lite::Tensor::Category::CONST_TENSOR);
  in_t->MallocData();
  memcpy(in_t->MutableData(), input, sizeof(float) * in_t->ElementsNum());
  inputs.push_back(in_t);

  auto out_t = new lite::Tensor(kNumberTypeFloat, out_shape, schema::Format_NHWC, lite::Tensor::Category::CONST_TENSOR);
  out_t->MallocData();
  outputs.push_back(out_t);

  auto ctx = new (std::nothrow) lite::InnerContext;
  ctx->thread_num_ = 2;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  CropParameter crop_param;
  crop_param.axis_ = 2;
  crop_param.offset_[0] = 0;
  crop_param.offset_[1] = 0;
  auto kernel = new kernel::CropCPUKernel(reinterpret_cast<OpParameter *>(&crop_param), inputs, outputs, ctx);
  kernel->Init();
  kernel->Run();

  auto *output = reinterpret_cast<float *>(outputs[0]->MutableData());
  for (int i = 0; i < kOutSize; ++i) {
    std::cout << output[i] << " ";
  }
  std::cout << "\n";
  ASSERT_EQ(0, CompareOutputData(output, expect_out, kOutSize, 0.000001));
}
}  // namespace mindspore

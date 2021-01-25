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
#include "schema/inner/model_generated.h"
#include "include/context.h"
#include "src/tensor.h"
#include "common/common_test.h"
#include "src/kernel_registry.h"
#include "nnacl/int8/resize_int8.h"

namespace mindspore {
using mindspore::lite::QuantArg;
using mindspore::lite::Tensor;

class TestResizeNearestNeighborInt8 : public mindspore::CommonTest {
 public:
  TestResizeNearestNeighborInt8() = default;
  void Prepare(const std::vector<int> &in_shape, const std::vector<int> &out_shape, int8_t *input_data,
               int8_t *output_data, const QuantArg quant_in, const QuantArg quant_out, const bool align_corners,
               const int thread_num);
  void TearDown() override;

  std::vector<lite::Tensor *> inputs;
  std::vector<lite::Tensor *> outputs;
  ResizeParameter param_ = {};
  lite::Tensor in_tensor;
  lite::Tensor out_tensor;

  kernel::KernelKey desc_ = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, schema::PrimitiveType_Resize};
  kernel::KernelCreator creator_ = nullptr;
  lite::InnerContext ctx_ = lite::InnerContext();
  kernel::LiteKernel *kernel_ = nullptr;
  float err_percent_ = 0.05f;
};

void TestResizeNearestNeighborInt8::Prepare(const std::vector<int> &in_shape, const std::vector<int> &out_shape,
                                            int8_t *input_data, int8_t *output_data, const QuantArg quant_in,
                                            const QuantArg quant_out, const bool align_corners, const int thread_num) {
  in_tensor.set_data_type(kNumberTypeInt8);
  in_tensor.set_shape(in_shape);
  in_tensor.set_data(input_data);
  in_tensor.AddQuantParam(quant_in);

  out_tensor.set_data_type(kNumberTypeInt8);
  out_tensor.set_shape(out_shape);
  out_tensor.set_data(output_data);
  out_tensor.AddQuantParam(quant_out);

  inputs.push_back(&in_tensor);
  outputs.push_back(&out_tensor);

  param_.method_ = static_cast<int>(schema::ResizeMethod_NEAREST);
  param_.new_width_ = out_shape[2];
  param_.new_height_ = out_shape[1];
  if (align_corners) {
    param_.coordinate_transform_mode_ = 1;
  }

  creator_ = lite::KernelRegistry::GetInstance()->GetCreator(desc_);

  ctx_.thread_num_ = thread_num;
  ASSERT_EQ(lite::RET_OK, ctx_.Init());
  kernel_ = creator_(inputs, outputs, reinterpret_cast<OpParameter *>(&param_), &ctx_, desc_);
}

void TestResizeNearestNeighborInt8::TearDown() {
  in_tensor.set_data(nullptr);
  out_tensor.set_data(nullptr);
}

// 2*2*1 -> 4*4*1
TEST_F(TestResizeNearestNeighborInt8, NearestNeighbor0) {
  std::vector<int> in_shape = {1, 2, 2, 1};
  std::vector<int> out_shape = {1, 4, 4, 1};
  QuantArg quant_in = {0.00390625, 2};
  QuantArg quant_out = {0.015625, 5};
  int8_t input_data[] = {0, 1, 2, 3};
  const int out_element_num = 16;
  int8_t output_data[out_element_num] = {0};
  int thread_num = 1;
  int8_t expect[16] = {5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5};
  err_percent_ = 0.25f;

  Prepare(in_shape, out_shape, input_data, output_data, quant_in, quant_out, false, thread_num);
  kernel_->Init();
  kernel_->Run();

  CompareOutputInt8(output_data, expect, 16, err_percent_);
}

// 2*2*2*5 -> 2*4*4*5
TEST_F(TestResizeNearestNeighborInt8, NearestNeighbor1) {
  std::vector<int> in_shape = {2, 2, 2, 5};
  std::vector<int> out_shape = {2, 4, 4, 5};
  QuantArg quant_in = {0.00390625, 2};
  QuantArg quant_out = {0.015625, 5};
  int8_t input_data[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                         20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39};
  const int out_element_num = 160;
  int8_t output_data[out_element_num] = {0};
  int thread_num = 1;
  int8_t expect[160] = {5,  5,  5,  5,  6,  5,  5,  5,  5,  6,  6,  6,  6,  7,  7,  6,  6,  6,  7,  7,  5,  5,  5,
                        5,  6,  5,  5,  5,  5,  6,  6,  6,  6,  7,  7,  6,  6,  6,  7,  7,  7,  7,  8,  8,  8,  7,
                        7,  8,  8,  8,  8,  9,  9,  9,  9,  8,  9,  9,  9,  9,  7,  7,  8,  8,  8,  7,  7,  8,  8,
                        8,  8,  9,  9,  9,  9,  8,  9,  9,  9,  9,  10, 10, 10, 10, 11, 10, 10, 10, 10, 11, 11, 11,
                        11, 12, 12, 11, 11, 11, 12, 12, 10, 10, 10, 10, 11, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12,
                        11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 13, 14, 14,
                        14, 14, 12, 12, 13, 13, 13, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 13, 14, 14, 14, 14};

  Prepare(in_shape, out_shape, input_data, output_data, quant_in, quant_out, false, thread_num);
  kernel_->Init();
  kernel_->Run();

  CompareOutputInt8(output_data, expect, out_element_num, err_percent_);
}

// 2*2*2*5 -> 2*4*4*5 thread num 2 align_corners
TEST_F(TestResizeNearestNeighborInt8, NearestNeighbor2) {
  std::vector<int> in_shape = {2, 2, 2, 5};
  std::vector<int> out_shape = {2, 4, 4, 5};
  QuantArg quant_in = {0.00390625, 2};
  QuantArg quant_out = {0.015625, 5};
  int8_t input_data[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                         20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39};
  const int out_element_num = 160;
  int8_t output_data[out_element_num] = {0};
  int thread_num = 2;
  int8_t expect[160] = {
    5,  5,  5,  5,  6,  5,  5,  5,  5,  6,  6,  6,  6,  7,  7,  6,  6,  6,  7,  7,  5,  5,  5,  5,  6,  5,  5,
    5,  5,  6,  6,  6,  6,  7,  7,  6,  6,  6,  7,  7,  7,  7,  8,  8,  8,  7,  7,  8,  8,  8,  8,  9,  9,  9,
    9,  8,  9,  9,  9,  9,  7,  7,  8,  8,  8,  7,  7,  8,  8,  8,  8,  9,  9,  9,  9,  8,  9,  9,  9,  9,  10,
    10, 10, 10, 11, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 11, 11, 11, 12, 12, 10, 10, 10, 10, 11, 10, 10, 10,
    10, 11, 11, 11, 11, 12, 12, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14,
    13, 14, 14, 14, 14, 12, 12, 13, 13, 13, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 13, 14, 14, 14, 14,
  };

  Prepare(in_shape, out_shape, input_data, output_data, quant_in, quant_out, true, thread_num);
  kernel_->Init();
  kernel_->Run();

  CompareOutputInt8(output_data, expect, out_element_num, err_percent_);
}

// 2*2*2*5 -> 2*4*4*5 thread num 2, same quant args
TEST_F(TestResizeNearestNeighborInt8, NearestNeighbor3) {
  std::vector<int> in_shape = {2, 2, 2, 5};
  std::vector<int> out_shape = {2, 4, 4, 5};
  QuantArg quant_in = {0.00390625, 2};
  QuantArg quant_out = {0.00390625, 2};
  int8_t input_data[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                         20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39};
  const int out_element_num = 160;
  int8_t output_data[out_element_num] = {0};
  int thread_num = 2;
  int8_t expect[160] = {0,  1,  2,  3,  4,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  5,  6,  7,  8,  9,  0,  1,  2,
                        3,  4,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 10,
                        11, 12, 13, 14, 15, 16, 17, 18, 19, 15, 16, 17, 18, 19, 10, 11, 12, 13, 14, 10, 11, 12, 13,
                        14, 15, 16, 17, 18, 19, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 20, 21, 22, 23, 24, 25, 26,
                        27, 28, 29, 25, 26, 27, 28, 29, 20, 21, 22, 23, 24, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                        25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 35, 36, 37,
                        38, 39, 30, 31, 32, 33, 34, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 35, 36, 37, 38, 39};

  Prepare(in_shape, out_shape, input_data, output_data, quant_in, quant_out, false, thread_num);
  kernel_->Init();
  kernel_->Run();

  CompareOutputInt8(output_data, expect, out_element_num, err_percent_);
}

// 2*2*2*5 -> 2*4*4*5 thread num 2 align_corners, same quant args
TEST_F(TestResizeNearestNeighborInt8, NearestNeighbor4) {
  std::vector<int> in_shape = {2, 2, 2, 5};
  std::vector<int> out_shape = {2, 4, 4, 5};
  QuantArg quant_in = {0.00390625, 2};
  QuantArg quant_out = {0.00390625, 2};
  int8_t input_data[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                         20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39};
  const int out_element_num = 160;
  int8_t output_data[out_element_num] = {0};
  int thread_num = 2;
  int8_t expect[160] = {
    0,  1,  2,  3,  4,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  5,  6,  7,  8,  9,  0,  1,  2,  3,  4,  0,  1,
    2,  3,  4,  5,  6,  7,  8,  9,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    19, 15, 16, 17, 18, 19, 10, 11, 12, 13, 14, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 25, 26, 27, 28, 29, 20, 21, 22, 23, 24, 20, 21, 22,
    23, 24, 25, 26, 27, 28, 29, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
    35, 36, 37, 38, 39, 30, 31, 32, 33, 34, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 35, 36, 37, 38, 39,
  };

  Prepare(in_shape, out_shape, input_data, output_data, quant_in, quant_out, true, thread_num);
  kernel_->Init();
  kernel_->Run();

  CompareOutputInt8(output_data, expect, out_element_num, err_percent_);
}
}  // namespace mindspore

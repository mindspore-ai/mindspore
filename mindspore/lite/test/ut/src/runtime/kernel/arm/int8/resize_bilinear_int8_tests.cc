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

class TestResizeBilinearInt8 : public mindspore::CommonTest {
 public:
  TestResizeBilinearInt8() = default;
  void TearDown() override;
  void Prepare(const std::vector<int> &in_shape, const std::vector<int> &out_shape, int8_t *input_data,
               int8_t *output_data, const QuantArg quant_in, const QuantArg quant_out, const bool align_corners,
               const int thread_num);
  std::vector<lite::Tensor *> inputs;
  std::vector<lite::Tensor *> outputs;
  ResizeParameter param_ = {};
  lite::Tensor in_tensor;
  lite::Tensor out_tensor;

  kernel::KernelKey desc_ = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, schema::PrimitiveType_Resize};
  kernel::KernelCreator creator_ = nullptr;
  lite::InnerContext ctx_ = lite::InnerContext();
  kernel::LiteKernel *kernel_ = nullptr;
  float err_percent_ = 0.2f;
};

void TestResizeBilinearInt8::TearDown() {
  in_tensor.set_data(nullptr);
  out_tensor.set_data(nullptr);
}

void TestResizeBilinearInt8::Prepare(const std::vector<int> &in_shape, const std::vector<int> &out_shape,
                                     int8_t *input_data, int8_t *output_data, const mindspore::QuantArg quant_in,
                                     const mindspore::QuantArg quant_out, const bool align_corners,
                                     const int thread_num) {
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

  param_.method_ = static_cast<int>(schema::ResizeMethod_LINEAR);
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

TEST_F(TestResizeBilinearInt8, Bilinear0) {
  int8_t input_data[] = {0, 1, 2, 3};
  int8_t output_data[16] = {0};
  std::vector<int> in_shape = {1, 2, 2, 1};
  std::vector<int> out_shape = {1, 4, 4, 1};
  const lite::QuantArg quant_in = {0.005f, 0};
  const lite::QuantArg quant_out = {0.008f, 0};
  bool align_corners = false;
  int thread_num = 1;
  int8_t expect[16] = {0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 2, 2, 2};

  Prepare(in_shape, out_shape, input_data, output_data, quant_in, quant_out, align_corners, thread_num);
  kernel_->Run();

  CompareOutputInt8(output_data, expect, 16, err_percent_);
}

// 2*2*2*5 -> 2*4*4*5
TEST_F(TestResizeBilinearInt8, Bilinear1) {
  std::vector<int> in_shape = {2, 2, 2, 5};
  std::vector<int> out_shape = {2, 4, 4, 5};
  int8_t input_data[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                         20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39};
  int8_t output_data[160] = {0};
  const lite::QuantArg quant_in = {0.005f, 0};
  const lite::QuantArg quant_out = {0.008f, 0};
  int thread_num = 1;
  bool align_corners = false;
  int8_t expect[160] = {0,  1,  1,  2,  2,  2,  2,  3,  3,  4,  3,  4,  4,  5,  6,  3,  4,  4,  5,  6,  3,  4,  4,
                        5,  6,  5,  5,  6,  7,  7,  6,  7,  8,  8,  9,  6,  7,  8,  8,  9,  6,  7,  7,  8,  9,  8,
                        8,  9,  10, 10, 9,  10, 11, 11, 12, 9,  10, 11, 11, 12, 6,  7,  7,  8,  9,  8,  8,  9,  10,
                        10, 9,  10, 11, 11, 12, 9,  10, 11, 11, 12, 13, 13, 14, 14, 15, 14, 15, 15, 16, 17, 16, 16,
                        17, 18, 18, 16, 16, 17, 18, 18, 16, 16, 17, 18, 18, 17, 18, 18, 19, 20, 19, 19, 20, 21, 21,
                        19, 19, 20, 21, 21, 19, 19, 20, 21, 21, 20, 21, 22, 22, 23, 22, 23, 23, 24, 24, 22, 23, 23,
                        24, 24, 19, 19, 20, 21, 21, 20, 21, 22, 22, 23, 22, 23, 23, 24, 24, 22, 23, 23, 24, 24};

  Prepare(in_shape, out_shape, input_data, output_data, quant_in, quant_out, align_corners, thread_num);
  kernel_->Run();

  CompareOutputInt8(output_data, expect, 160, err_percent_);
}

// 2*2*2*5 -> 2*4*4*5 thread num 2, align corners
TEST_F(TestResizeBilinearInt8, Bilinear2) {
  std::vector<int> in_shape = {2, 2, 2, 5};
  std::vector<int> out_shape = {2, 4, 4, 5};
  int8_t input_data[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                         20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39};
  int8_t output_data[160] = {0};

  const lite::QuantArg quant_in = {0.005f, 0};
  const lite::QuantArg quant_out = {0.008f, 0};
  int thread_num = 2;
  bool align_corners = true;
  int8_t expect[160] = {0,  1,  1,  2,  2,  1,  2,  2,  3,  4,  2,  3,  3,  4,  5,  3,  4,  4,  5,  6,  2,  3,  3,
                        4,  5,  3,  4,  4,  5,  6,  4,  5,  5,  6,  7,  5,  6,  6,  7,  8,  4,  5,  5,  6,  7,  5,
                        6,  6,  7,  8,  6,  7,  8,  8,  9,  7,  8,  9,  9,  10, 6,  7,  7,  8,  9,  7,  8,  9,  9,
                        10, 8,  9,  10, 10, 11, 9,  10, 11, 11, 12, 13, 13, 14, 14, 15, 14, 14, 15, 15, 16, 15, 15,
                        16, 16, 17, 16, 16, 17, 18, 18, 15, 15, 16, 16, 17, 16, 16, 17, 18, 18, 17, 17, 18, 19, 19,
                        18, 18, 19, 20, 20, 17, 17, 18, 19, 19, 18, 18, 19, 20, 20, 19, 19, 20, 21, 21, 20, 20, 21,
                        22, 22, 19, 19, 20, 21, 21, 20, 20, 21, 22, 22, 21, 21, 22, 23, 23, 22, 23, 23, 24, 24};

  Prepare(in_shape, out_shape, input_data, output_data, quant_in, quant_out, align_corners, thread_num);
  kernel_->Run();

  CompareOutputInt8(output_data, expect, 160, err_percent_);
}

// 2*2*2*5 -> 2*4*4*5 thread num 2, align corners zp -128
TEST_F(TestResizeBilinearInt8, Bilinear3) {
  std::vector<int> in_shape = {2, 2, 2, 5};
  std::vector<int> out_shape = {2, 4, 4, 5};
  int8_t input_data[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                         20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39};
  int8_t output_data[160] = {0};

  const lite::QuantArg quant_in = {0.005f, 2};
  const lite::QuantArg quant_out = {0.005f, 2};
  int thread_num = 2;
  bool align_corners = true;
  int8_t expect[160] = {0,  1,  2,  3,  4,  2,  3,  4,  5,  6,  3,  4,  5,  6,  7,  5,  6,  7,  8,  9,  3,  4,  5,
                        6,  7,  5,  6,  7,  8,  9,  7,  8,  9,  10, 11, 8,  9,  10, 11, 12, 7,  8,  9,  10, 11, 8,
                        9,  10, 11, 12, 10, 11, 12, 13, 14, 12, 13, 14, 15, 16, 10, 11, 12, 13, 14, 12, 13, 14, 15,
                        16, 13, 14, 15, 16, 17, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 22, 23, 24, 25, 26, 23, 24,
                        25, 26, 27, 25, 26, 27, 28, 29, 23, 24, 25, 26, 27, 25, 26, 27, 28, 29, 27, 28, 29, 30, 31,
                        28, 29, 30, 31, 32, 27, 28, 29, 30, 31, 28, 29, 30, 31, 32, 30, 31, 32, 33, 34, 32, 33, 34,
                        35, 36, 30, 31, 32, 33, 34, 32, 33, 34, 35, 36, 33, 34, 35, 36, 37, 35, 36, 37, 38, 39};

  Prepare(in_shape, out_shape, input_data, output_data, quant_in, quant_out, align_corners, thread_num);
  kernel_->Run();

  err_percent_ = 0.325f;
  CompareOutputInt8(output_data, expect, 160, err_percent_);
}

}  // namespace mindspore

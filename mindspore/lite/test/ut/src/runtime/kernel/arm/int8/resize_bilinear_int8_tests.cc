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
#include "include/context.h"
#include "src/ir/tensor.h"
#include "common/common_test.h"
#include "mindspore/lite/src/kernel_registry.h"
#include "nnacl/int8/resize.h"

namespace mindspore {
using mindspore::lite::tensor::QuantArg;
using mindspore::lite::tensor::Tensor;

class TestResizeBilinearInt8 : public mindspore::CommonTest {
 public:
  TestResizeBilinearInt8() = default;
  void TearDown() override;
  void Prepare(const std::vector<int> &in_shape, const std::vector<int> &out_shape, int8_t *input_data,
               int8_t *output_data, const QuantArg quant_in, const QuantArg quant_out, const bool align_corners,
               const int thread_num);
  std::vector<lite::tensor::Tensor *> inputs;
  std::vector<lite::tensor::Tensor *> outputs;
  ResizeParameter param_ = {};
  lite::tensor::Tensor in_tensor;
  lite::tensor::Tensor out_tensor;

  kernel::KernelKey desc_ = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, schema::PrimitiveType_Resize};
  kernel::KernelCreator creator_ = nullptr;
  lite::Context ctx_ = lite::Context();
  kernel::LiteKernel *kernel_ = nullptr;
  float err_percent_ = 0.2f;
};

void TestResizeBilinearInt8::TearDown() {
  in_tensor.SetData(nullptr);
  out_tensor.SetData(nullptr);
}

void TestResizeBilinearInt8::Prepare(const std::vector<int> &in_shape, const std::vector<int> &out_shape,
                                     int8_t *input_data, int8_t *output_data, const mindspore::QuantArg quant_in,
                                     const mindspore::QuantArg quant_out, const bool align_corners,
                                     const int thread_num) {
  in_tensor.set_data_type(kNumberTypeInt8);
  in_tensor.set_shape(in_shape);
  in_tensor.SetData(input_data);
  in_tensor.AddQuantParam(quant_in);

  out_tensor.set_data_type(kNumberTypeInt8);
  out_tensor.set_shape(out_shape);
  out_tensor.SetData(output_data);
  out_tensor.AddQuantParam(quant_out);

  inputs.push_back(&in_tensor);
  outputs.push_back(&out_tensor);

  param_.method_ = static_cast<int>(schema::ResizeMethod_BILINEAR);
  param_.new_width_ = out_shape[2];
  param_.new_height_ = out_shape[1];
  param_.align_corners_ = align_corners;

  creator_ = lite::KernelRegistry::GetInstance()->GetCreator(desc_);

  ctx_.thread_num_ = thread_num;
  kernel_ = creator_(inputs, outputs, reinterpret_cast<OpParameter *>(&param_), &ctx_, desc_, nullptr);
}

TEST_F(TestResizeBilinearInt8, Bilinear0) {
  int8_t input_data[] = {0, 1, 2, 3};
  int8_t output_data[16] = {0};
  std::vector<int> in_shape = {1, 2, 2, 1};
  std::vector<int> out_shape = {1, 4, 4, 1};
  const lite::tensor::QuantArg quant_in = {0.005f, 2};
  const lite::tensor::QuantArg quant_out = {0.008f, 5};
  bool align_corners = false;
  int thread_num = 1;
  int8_t expect[16] = {4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 5, 5, 6, 6};

  Prepare(in_shape, out_shape, input_data, output_data, quant_in, quant_out, align_corners, thread_num);
  kernel_->Init();  // todo delete
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
  const lite::tensor::QuantArg quant_in = {0.005f, 2};
  const lite::tensor::QuantArg quant_out = {0.008f, 5};
  int thread_num = 1;
  bool align_corners = false;
  int8_t expect[160] = {4,  4,  5,  6,  6,  5,  6,  7,  7,  8,  7,  8,  8,  9,  9,  7,  8,  8,  9,  9,  7,  8,  8,
                        9,  9,  8,  9,  10, 10, 11, 10, 11, 11, 12, 13, 10, 11, 11, 12, 13, 10, 11, 11, 12, 13, 12,
                        12, 13, 13, 14, 13, 14, 14, 15, 16, 13, 14, 14, 15, 16, 10, 11, 11, 12, 13, 12, 12, 13, 13,
                        14, 13, 14, 14, 15, 16, 13, 14, 14, 15, 16, 16, 17, 18, 18, 19, 18, 18, 19, 20, 20, 19, 20,
                        21, 21, 22, 19, 20, 21, 21, 22, 19, 20, 21, 21, 22, 21, 22, 22, 23, 23, 23, 23, 24, 24, 25,
                        23, 23, 24, 24, 25, 23, 23, 24, 24, 25, 24, 25, 25, 26, 27, 26, 26, 27, 28, 28, 26, 26, 27,
                        28, 28, 23, 23, 24, 24, 25, 24, 25, 25, 26, 27, 26, 26, 27, 28, 28, 26, 26, 27, 28, 28};

  Prepare(in_shape, out_shape, input_data, output_data, quant_in, quant_out, align_corners, thread_num);
  kernel_->Init();
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

  const lite::tensor::QuantArg quant_in = {0.005f, 2};
  const lite::tensor::QuantArg quant_out = {0.008f, 5};
  int thread_num = 2;
  bool align_corners = true;
  int8_t expect[160] = {4,  4,  5,  6,  6,  5,  5,  6,  7,  7,  6,  6,  7,  8,  8,  7,  8,  8,  9,  9,  6,  6,  7,
                        8,  8,  7,  8,  8,  9,  9,  8,  9,  9,  10, 10, 9,  10, 10, 11, 11, 8,  9,  9,  10, 10, 9,
                        10, 10, 11, 11, 10, 11, 11, 12, 13, 11, 12, 12, 13, 14, 10, 11, 11, 12, 13, 11, 12, 12, 13,
                        14, 12, 13, 13, 14, 15, 13, 14, 14, 15, 16, 16, 17, 18, 18, 19, 17, 18, 19, 19, 20, 18, 19,
                        20, 20, 21, 19, 20, 21, 21, 22, 18, 19, 20, 20, 21, 19, 20, 21, 21, 22, 20, 21, 22, 22, 23,
                        21, 22, 23, 23, 24, 20, 21, 22, 22, 23, 21, 22, 23, 23, 24, 23, 23, 24, 24, 25, 24, 24, 25,
                        25, 26, 23, 23, 24, 24, 25, 24, 24, 25, 25, 26, 25, 25, 26, 26, 27, 26, 26, 27, 28, 28};

  Prepare(in_shape, out_shape, input_data, output_data, quant_in, quant_out, align_corners, thread_num);
  kernel_->Init();
  kernel_->Run();

  CompareOutputInt8(output_data, expect, 160, err_percent_);
}
}  // namespace mindspore

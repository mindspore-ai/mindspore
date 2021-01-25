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
#include <memory>
#include "schema/inner/model_generated.h"
#include "src/common/log_adapter.h"
#include "common/common_test.h"
#include "src/tensor.h"
#include "mindspore/lite/src/kernel_registry.h"
#include "nnacl/int8/scale_int8.h"

namespace mindspore {
using mindspore::lite::QuantArg;
using mindspore::lite::Tensor;

class TestScaleInt8 : public mindspore::CommonTest {
 public:
  TestScaleInt8() = default;
  void Prepare(const std::vector<int> &in_shape, int8_t *input_data, const std::vector<int> &scale_shape,
               int8_t *scale_data, const std::vector<int> &bias_shape, int8_t *bias_data,
               const std::vector<int> &out_shape, int8_t *output_data, int axis, bool has_bias);
  void TearDown() override;

 public:
  int thread_num_ = 1;

  ScaleParameter param_ = {};
  Tensor in_tensor_;
  Tensor scale_tensor_;
  Tensor bias_tensor_;
  Tensor out_tensor_;
  std::vector<Tensor *> inputs;
  std::vector<Tensor *> outputs = {&out_tensor_};
  kernel::KernelKey desc_ = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, schema::PrimitiveType_ScaleFusion};
  kernel::KernelCreator creator_ = nullptr;
  lite::InnerContext ctx_ = lite::InnerContext();
  kernel::LiteKernel *kernel_ = nullptr;
  const QuantArg quant_in_ = {0.005f, 5};
  const QuantArg quant_scale_ = {0.1f, 1};
  const QuantArg quant_bias_ = {0.002f, 2};
  const QuantArg quant_out_ = {0.01f, 1};
  float err_tol_ = 0.05;
};

void TestScaleInt8::TearDown() {
  in_tensor_.set_data(nullptr);
  scale_tensor_.set_data(nullptr);
  bias_tensor_.set_data(nullptr);
  out_tensor_.set_data(nullptr);
}

void TestScaleInt8::Prepare(const std::vector<int> &in_shape, int8_t *input_data, const std::vector<int> &scale_shape,
                            int8_t *scale_data, const std::vector<int> &bias_shape, int8_t *bias_data,
                            const std::vector<int> &out_shape, int8_t *output_data, int axis, bool has_bias) {
  in_tensor_.set_data_type(kNumberTypeInt8);
  in_tensor_.set_shape(in_shape);
  in_tensor_.set_data(input_data);
  in_tensor_.AddQuantParam(quant_in_);
  scale_tensor_.set_data_type(kNumberTypeInt8);
  scale_tensor_.set_shape(scale_shape);
  scale_tensor_.set_data(scale_data);
  scale_tensor_.AddQuantParam(quant_scale_);

  inputs.clear();
  inputs.emplace_back(&in_tensor_);
  inputs.emplace_back(&scale_tensor_);
  if (has_bias) {
    bias_tensor_.set_data_type(kNumberTypeInt8);
    bias_tensor_.set_shape(bias_shape);
    bias_tensor_.set_data(bias_data);
    bias_tensor_.AddQuantParam(quant_bias_);
    inputs.emplace_back(&bias_tensor_);
  }

  out_tensor_.set_data_type(kNumberTypeInt8);
  out_tensor_.set_shape(out_shape);
  out_tensor_.set_data(output_data);
  out_tensor_.AddQuantParam(quant_out_);

  param_.axis_ = axis;
  creator_ = lite::KernelRegistry::GetInstance()->GetCreator(desc_);

  ctx_.thread_num_ = thread_num_;
  ASSERT_EQ(lite::RET_OK, ctx_.Init());
  kernel_ = creator_(inputs, outputs, reinterpret_cast<OpParameter *>(&param_), &ctx_, desc_);
}

TEST_F(TestScaleInt8, scale1) {
  /* 1 2 2 1 NHWC */
  int8_t input_data[96] = {0, 1, 2, 3};
  int8_t scale_data[4] = {2, 2, 2, 2};
  int8_t bias_data[4] = {3, 3, 3, 3};
  int8_t out_data[4] = {0};
  bool has_bias = true;

  int axis = 1;
  std::vector<int> input_shape = {1, 2, 2, 1};
  std::vector<int> scale_shape = {2, 2, 1};
  std::vector<int> bias_shape = {2, 2, 1};
  std::vector<int> output_shape = {1, 2, 2, 1};
  int output_size = 4;
  int8_t correct[] = {1, 1, 1, 1};

  thread_num_ = 2;
  Prepare(input_shape, input_data, scale_shape, scale_data, bias_shape, bias_data, output_shape, out_data, axis,
          has_bias);
  auto ret = kernel_->Run();
  EXPECT_EQ(0, ret);

  err_tol_ = 0.01;
  CompareOutputInt8(out_data, correct, output_size, err_tol_);
}

TEST_F(TestScaleInt8, scale2) {
  /* 1 2 2 1 NHWC */
  int8_t input_data[96] = {0, 10, 20, 30};
  int8_t scale_data[4] = {2, 2, 2, 2};
  int8_t bias_data[4] = {3, 3, 3, 3};
  int8_t out_data[4] = {0};
  bool has_bias = true;

  int axis = 1;
  std::vector<int> input_shape = {1, 2, 2, 1};
  std::vector<int> scale_shape = {2, 2, 1};
  std::vector<int> bias_shape = {2, 2, 1};
  std::vector<int> output_shape = {1, 2, 2, 1};
  int output_size = 4;
  int8_t correct[] = {1, 1, 2, 2};

  thread_num_ = 2;
  Prepare(input_shape, input_data, scale_shape, scale_data, bias_shape, bias_data, output_shape, out_data, axis,
          has_bias);
  auto ret = kernel_->Run();
  EXPECT_EQ(0, ret);

  err_tol_ = 0.01;
  CompareOutputInt8(out_data, correct, output_size, err_tol_);
}

TEST_F(TestScaleInt8, scale3) {
  /* 1 2 2 1 NHWC */
  int8_t input_data[96] = {0, 90, 100, 120};
  int8_t scale_data[4] = {2, 2, 2, 2};
  int8_t bias_data[4] = {3, 3, 3, 3};
  int8_t out_data[4] = {0};
  bool has_bias = false;

  int axis = 1;
  std::vector<int> input_shape = {1, 2, 2, 1};
  std::vector<int> scale_shape = {2, 2, 1};
  std::vector<int> bias_shape = {2, 2, 1};
  std::vector<int> output_shape = {1, 2, 2, 1};
  int output_size = 4;
  int8_t correct[] = {1, 5, 6, 7};

  thread_num_ = 2;
  Prepare(input_shape, input_data, scale_shape, scale_data, bias_shape, bias_data, output_shape, out_data, axis,
          has_bias);
  auto ret = kernel_->Run();
  EXPECT_EQ(0, ret);

  err_tol_ = 0.01;
  CompareOutputInt8(out_data, correct, output_size, err_tol_);
}
}  // namespace mindspore

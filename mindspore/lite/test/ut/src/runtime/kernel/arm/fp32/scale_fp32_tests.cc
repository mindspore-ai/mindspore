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
#include <vector>
#include "mindspore/lite/src/lite_kernel.h"
#include "mindspore/lite/src/tensor.h"
#include "common/common_test.h"
#include "nnacl/pad_parameter.h"
#include "mindspore/lite/src/kernel_registry.h"
#include "schema/ops_generated.h"
#include "nnacl/fp32/scale_fp32.h"

using mindspore::schema::ActivationType;
using mindspore::schema::ActivationType_NO_ACTIVATION;
using mindspore::schema::ActivationType_RELU;
using mindspore::schema::ActivationType_RELU6;
using mindspore::schema::Format_NHWC;
namespace mindspore {

class TestScaleFp32 : public mindspore::CommonTest {
 public:
  TestScaleFp32() = default;
  void Prepare(const std::vector<int> &input_shape, const std::vector<int> &scale_shape,
               const std::vector<int> &offset_shape, const std::vector<int> &output_shape, float *input_data,
               float *scale_data, float *offset_data, float *output_data, int axis, ActivationType act_type,
               const int thread_num);

  void TearDown() override;

 public:
  float err_tol = 1e-5;
  lite::Tensor in_tensor_;
  lite::Tensor scale_tensor_;
  lite::Tensor offset_tensor_;
  lite::Tensor out_tensor_;
  ScaleParameter param_;
  std::vector<lite::Tensor *> inputs_{&in_tensor_, &scale_tensor_, &offset_tensor_};
  std::vector<lite::Tensor *> outputs_{&out_tensor_};
  kernel::KernelKey desc_ = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_ScaleFusion};
  lite::InnerContext ctx_ = lite::InnerContext();
  kernel::KernelCreator creator_ = nullptr;
  kernel::LiteKernel *kernel_ = nullptr;
};

void TestScaleFp32::TearDown() {
  in_tensor_.set_data(nullptr);
  scale_tensor_.set_data(nullptr);
  offset_tensor_.set_data(nullptr);
  out_tensor_.set_data(nullptr);
}

void TestScaleFp32::Prepare(const std::vector<int> &input_shape, const std::vector<int> &scale_shape,
                            const std::vector<int> &offset_shape, const std::vector<int> &output_shape,
                            float *input_data, float *scale_data, float *offset_data, float *output_data, int axis,
                            ActivationType act_type, const int thread_num) {
  in_tensor_.set_data_type(kNumberTypeFloat32);
  in_tensor_.set_format(Format_NHWC);
  in_tensor_.set_shape(input_shape);
  scale_tensor_.set_data_type(kNumberTypeFloat32);
  scale_tensor_.set_format(Format_NHWC);
  scale_tensor_.set_shape(scale_shape);
  offset_tensor_.set_data_type(kNumberTypeFloat32);
  offset_tensor_.set_format(Format_NHWC);
  offset_tensor_.set_shape(offset_shape);
  out_tensor_.set_data_type(kNumberTypeFloat32);
  out_tensor_.set_shape(output_shape);

  in_tensor_.set_data(input_data);
  scale_tensor_.set_data(scale_data);
  offset_tensor_.set_data(offset_data);
  out_tensor_.set_data(output_data);

  param_.activation_type_ = act_type;
  param_.axis_ = axis;
  ctx_ = lite::InnerContext();
  ctx_.thread_num_ = thread_num;
  ctx_.Init();
  creator_ = lite::KernelRegistry::GetInstance()->GetCreator(desc_);
  ASSERT_NE(creator_, nullptr);
  kernel_ = creator_(inputs_, outputs_, reinterpret_cast<OpParameter *>(&param_), &ctx_, desc_);
  ASSERT_NE(kernel_, nullptr);
}

TEST_F(TestScaleFp32, ScaleNoAct) {
  std::vector<int> input_shape{1, 2, 2, 3};
  std::vector<int> scale_shape{3};
  std::vector<int> offset_shape{3};
  std::vector<int> output_shape{1, 2, 2, 3};
  float in_data[12] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0};
  float scale_data[3] = {1.0, 2.0, 3.0};
  float offset_data[3] = {1.0, 1.0, 1.0};
  float out_data[12] = {0};
  int axis = -1;
  int thread_num = 2;
  Prepare(input_shape, scale_shape, offset_shape, output_shape, in_data, scale_data, offset_data, out_data, axis,
          ActivationType_NO_ACTIVATION, thread_num);

  auto ret = kernel_->Run();
  EXPECT_EQ(0, ret);

  std::vector<float> expect{1.0, 3.0, 7.0, 4.0, 9.0, 16.0, 7.0, 15.0, 25.0, 10.0, 21.0, 34.0};

  ASSERT_EQ(0, CompareOutputData(out_data, expect.data(), 12, err_tol));
}

TEST_F(TestScaleFp32, ScaleRelu) {
  std::vector<int> input_shape{1, 2, 2, 3};
  std::vector<int> scale_shape{3};
  std::vector<int> offset_shape{3};
  std::vector<int> output_shape{1, 2, 2, 3};
  float in_data[12] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0};
  float scale_data[3] = {1.0, 2.0, 3.0};
  float offset_data[3] = {-5.0, -5.0, -5.0};
  float out_data[12] = {0};
  int axis = -1;
  int thread_num = 2;
  Prepare(input_shape, scale_shape, offset_shape, output_shape, in_data, scale_data, offset_data, out_data, axis,
          ActivationType_RELU, thread_num);

  auto ret = kernel_->Run();
  EXPECT_EQ(0, ret);

  std::vector<float> expect{0.0, 0.0, 1.0, 0.0, 3.0, 10.0, 1.0, 9.0, 19.0, 4.0, 15.0, 28.0};

  ASSERT_EQ(0, CompareOutputData(out_data, expect.data(), 12, err_tol));
}
TEST_F(TestScaleFp32, ScaleRelu6) {
  std::vector<int> input_shape{1, 2, 2, 3};
  std::vector<int> scale_shape{3};
  std::vector<int> offset_shape{3};
  std::vector<int> output_shape{1, 2, 2, 3};
  float in_data[12] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0};
  float scale_data[3] = {1.0, 2.0, 3.0};
  float offset_data[3] = {-5.0, -5.0, -5.0};
  float out_data[12] = {0};
  int axis = -1;
  int thread_num = 2;
  Prepare(input_shape, scale_shape, offset_shape, output_shape, in_data, scale_data, offset_data, out_data, axis,
          ActivationType_RELU6, thread_num);

  auto ret = kernel_->Run();
  EXPECT_EQ(0, ret);

  std::vector<float> expect{0.0, 0.0, 1.0, 0.0, 3.0, 6.0, 1.0, 6.0, 6.0, 4.0, 6.0, 6.0};

  ASSERT_EQ(0, CompareOutputData(out_data, expect.data(), 12, err_tol));
}
}  // namespace mindspore

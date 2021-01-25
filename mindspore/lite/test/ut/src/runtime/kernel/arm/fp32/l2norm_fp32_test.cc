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
#include "mindspore/lite/src/runtime/kernel/arm/fp32/l2_norm_fp32.h"
#include "src/kernel_registry.h"
#include "src/lite_kernel.h"
using mindspore::schema::Format_NHWC;

namespace mindspore {
class TestL2NormFp32 : public mindspore::CommonTest {
 public:
  TestL2NormFp32() = default;
  void Init(const std::vector<int> &input_shape, const std::vector<int> &output_shape, float *input_data,
            float *output_data, const int axis_num, ActType activation_type, const int thread_num);
  void TearDown() override;

 public:
  float err_tol_ = 1e-5;
  lite::Tensor in_tensor_;
  lite::Tensor out_tensor_;
  std::vector<lite::Tensor *> inputs_{&in_tensor_};
  std::vector<lite::Tensor *> outputs_{&out_tensor_};
  L2NormParameter param_;
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_Resize};
  lite::InnerContext ctx_ = lite::InnerContext();
  kernel::KernelCreator creator_ = nullptr;
  kernel::LiteKernel *kernel_ = nullptr;
};

void TestL2NormFp32::TearDown() {
  in_tensor_.set_data(nullptr);
  out_tensor_.set_data(nullptr);
}

void TestL2NormFp32::Init(const std::vector<int> &input_shape, const std::vector<int> &output_shape, float *input_data,
                          float *output_data, const int axis_num, ActType activation_type, const int thread_num) {
  in_tensor_.set_data_type(kNumberTypeFloat32);
  in_tensor_.set_format(Format_NHWC);
  in_tensor_.set_shape(input_shape);
  out_tensor_.set_data_type(kNumberTypeFloat32);
  out_tensor_.set_shape(output_shape);
  in_tensor_.set_data(input_data);
  out_tensor_.set_data(output_data);

  param_.axis_num_ = axis_num;
  if (axis_num == 1) {
    param_.axis_[0] = -1;
  }
  param_.epsilon_ = 1e-6;
  param_.act_type_ = activation_type;

  desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_L2NormalizeFusion};
  ctx_ = lite::InnerContext();
  ctx_.thread_num_ = thread_num;
  ASSERT_EQ(lite::RET_OK, ctx_.Init());
  creator_ = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator_, nullptr);
  kernel_ = creator_(inputs_, outputs_, reinterpret_cast<OpParameter *>(&param_), &ctx_, desc);
  ASSERT_NE(kernel_, nullptr);
}

// 2thread  all axis no_activation
TEST_F(TestL2NormFp32, Test1) {
  float input_data[18] = {-9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0,
                          0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0};
  float output_data[18] = {0};
  std::vector<int> input_shape = {1, 3, 2, 3};
  std::vector<int> output_shape = {1, 3, 2, 3};
  std::vector<float> expect = {-0.40699407, -0.3617725,  -0.31655094,  -0.27132937, -0.22610782, -0.18088625,
                               -0.13566469, -0.09044313, -0.045221563, 0.0,         0.045221563, 0.09044313,
                               0.13566469,  0.18088625,  0.22610782,   0.27132937,  0.31655094,  0.3617725};
  auto output_size = 18;
  int axis_num = 0;
  ActType act_type = ActType_No;
  int thread_num = 2;
  Init(input_shape, output_shape, input_data, output_data, axis_num, act_type, thread_num);
  auto ret = kernel_->Run();
  EXPECT_EQ(0, ret);

  ASSERT_EQ(0, CompareOutputData(output_data, expect.data(), output_size, err_tol_));
}

// 2thread  all axis relu
TEST_F(TestL2NormFp32, Test2) {
  float input_data[18] = {-9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0,
                          0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0};
  float output_data[18] = {0};
  std::vector<int> input_shape = {1, 3, 2, 3};
  std::vector<int> output_shape = {1, 3, 2, 3};
  std::vector<float> expect = {0.0,        0.0,        0.0,        0.0,        0.0,         0.0,
                               0.0,        0.0,        0.0,        0.0,        0.045221563, 0.09044313,
                               0.13566469, 0.18088625, 0.22610782, 0.27132937, 0.31655094,  0.3617725};
  auto output_size = 18;
  int axis_num = 0;
  ActType act_type = ActType_Relu;
  int thread_num = 2;
  Init(input_shape, output_shape, input_data, output_data, axis_num, act_type, thread_num);
  auto ret = kernel_->Run();
  EXPECT_EQ(0, ret);

  ASSERT_EQ(0, CompareOutputData(output_data, expect.data(), output_size, err_tol_));
}

// 4 thread  trailing axis  no activation
TEST_F(TestL2NormFp32, Test3) {
  float input_data[18] = {-9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0,
                          0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0};
  float output_data[18] = {0};
  std::vector<int> input_shape = {1, 3, 2, 3};
  std::vector<int> output_shape = {1, 3, 2, 3};
  std::vector<float> expect = {-0.6461623, -0.57436645, -0.5025706,  -0.6837635, -0.5698029, -0.45584232,
                               -0.8017837, -0.5345225,  -0.26726124, 0.0,        0.4472136,  0.8944272,
                               0.42426407, 0.56568545,  0.7071068,   0.49153918, 0.57346237, 0.65538555};
  auto output_size = 18;
  int axis_num = 1;
  ActType act_type = ActType_No;
  int thread_num = 4;
  Init(input_shape, output_shape, input_data, output_data, axis_num, act_type, thread_num);
  auto ret = kernel_->Run();
  EXPECT_EQ(0, ret);

  ASSERT_EQ(0, CompareOutputData(output_data, expect.data(), output_size, err_tol_));
}

// 1 thread  trailing axis  no activation
TEST_F(TestL2NormFp32, Test4) {
  float input_data[18] = {-9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0,
                          0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0};
  float output_data[18] = {0};
  std::vector<int> input_shape = {1, 3, 2, 3};
  std::vector<int> output_shape = {1, 3, 2, 3};
  std::vector<float> expect = {0.0,        0.0,        0.0,       0.0,        0.0,        0.0,
                               0.0,        0.0,        0.0,       0.0,        0.4472136,  0.8944272,
                               0.42426407, 0.56568545, 0.7071068, 0.49153918, 0.57346237, 0.65538555};
  auto output_size = 18;
  int axis_num = 1;
  ActType act_type = ActType_Relu6;
  int thread_num = 1;
  Init(input_shape, output_shape, input_data, output_data, axis_num, act_type, thread_num);
  auto ret = kernel_->Run();
  EXPECT_EQ(0, ret);

  ASSERT_EQ(0, CompareOutputData(output_data, expect.data(), output_size, err_tol_));
}

}  // namespace mindspore

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
#include "src/litert/kernel_registry.h"
#include "src/executor/kernel_exec.h"
#include "src/litert/tensor_category.h"
#include "nnacl/nnacl_manager.h"
#include "nnacl/pow_parameter.h"

namespace mindspore {
class TestPowerFp32 : public mindspore::CommonTest {
 public:
  TestPowerFp32() {}
};

TEST_F(TestPowerFp32, Simple) {
  float in0_data[] = {1, 2, 3, 4};
  float in1_data[] = {5, 6, 7, 8};
  lite::Tensor input0(kNumberTypeFloat32, {2, 2});
  lite::Tensor input1(kNumberTypeFloat32, {2, 2});
  memcpy(input0.MutableData(), in0_data, input0.Size());
  memcpy(input1.MutableData(), in1_data, input1.Size());
  std::vector<lite::Tensor *> inputs = {&input0, &input1};

  lite::Tensor output(kNumberTypeFloat32, {2, 2});
  output.MallocData();
  std::vector<lite::Tensor *> outputs = {&output};

  auto param = new PowParameter();
  param->scale_ = 1;
  param->shift_ = 0;
  param->op_parameter_.type_ = schema::PrimitiveType_PowFusion;
  param->op_parameter_.thread_num_ = 1;

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, NHWC, param->op_parameter_.type_};
  auto kernel = nnacl::NNACLKernelRegistry(&param->op_parameter_, inputs, outputs, ctx, desc);
  ASSERT_NE(kernel, nullptr);

  EXPECT_EQ(0, kernel->Prepare());
  EXPECT_EQ(0, kernel->Run());

  float correct[] = {1, 64, 2187, 65536};
  float *output_data = reinterpret_cast<float *>(output.data());
  ASSERT_EQ(0, CompareOutputData(output_data, correct, output.ElementsNum(), 0.0001));

  delete kernel;
  delete ctx;
}

TEST_F(TestPowerFp32, Broadcast) {
  float in0_data[] = {1, 2, 3, 4};
  float in1_data[] = {2};
  lite::Tensor input0(kNumberTypeFloat32, {2, 2});
  lite::Tensor input1(kNumberTypeFloat32, {1});
  memcpy(input0.MutableData(), in0_data, input0.Size());
  memcpy(input1.MutableData(), in1_data, input1.Size());
  std::vector<lite::Tensor *> inputs = {&input0, &input1};

  lite::Tensor output(kNumberTypeFloat32, {2, 2});
  output.MallocData();
  std::vector<lite::Tensor *> outputs = {&output};

  auto param = new PowParameter();
  param->power_ = 2;
  param->scale_ = 1;
  param->shift_ = 0;
  param->op_parameter_.type_ = schema::PrimitiveType_PowFusion;
  param->op_parameter_.thread_num_ = 2;

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, NHWC, param->op_parameter_.type_};
  auto kernel = nnacl::NNACLKernelRegistry(&param->op_parameter_, inputs, outputs, ctx, desc);
  ASSERT_NE(kernel, nullptr);

  EXPECT_EQ(0, kernel->Prepare());
  EXPECT_EQ(0, kernel->Run());

  float correct[] = {1, 4, 9, 16};
  float *output_data = reinterpret_cast<float *>(output.data());
  ASSERT_EQ(0, CompareOutputData(output_data, correct, output.ElementsNum(), 0.0001));

  delete kernel;
  delete ctx;
}
}  // namespace mindspore

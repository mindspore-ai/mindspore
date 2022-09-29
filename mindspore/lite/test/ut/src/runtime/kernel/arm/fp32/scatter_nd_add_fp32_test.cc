/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "src/litert/kernel_registry.h"
#include "nnacl/scatter_nd_parameter.h"

namespace mindspore {
using mindspore::lite::Tensor;

class TestScatterNdAdd : public mindspore::CommonTest {
 public:
  TestScatterNdAdd() {}
};

TEST_F(TestScatterNdAdd, Fp32OneDims) {
  std::vector<lite::Tensor *> inputs;
  inputs.push_back(CreateTensor<float>(kNumberTypeFloat32, {11}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));  // input_x
  inputs.push_back(
    CreateTensor<int>(kNumberTypeInt32, {15, 1}, {0, 1, 5, 3, 2, 2, 3, 2, 8, 6, 9, 3, 4, 6, 7}));  // indices
  inputs.push_back(CreateTensor<float>(kNumberTypeFloat32, {15},
                                       {{66.66, 1, 1, 3, 2, 2, 3, 2, 1.2, 0.3, 88.8, -5.4, 5.5, 6, 8.9}}));  // updates

  std::vector<lite::Tensor *> outputs;
  outputs.push_back(CreateTensor<float>(kNumberTypeFloat32, {11}, {}));

  auto param = static_cast<ScatterNDParameter *>(malloc(sizeof(ScatterNDParameter)));
  memset(param, 0, sizeof(ScatterNDParameter));

  auto ctx = std::make_shared<lite::InnerContext>();
  ctx->thread_num_ = 2;
  ASSERT_EQ(ctx->Init(), RET_OK);
  param->op_parameter.thread_num_ = ctx->thread_num_;

  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, NHWC,
                            schema::PrimitiveType_TensorScatterAdd};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), ctx.get(), desc);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(kernel->Prepare(), RET_OK);
  ASSERT_EQ(kernel->Run(), RET_OK);

  std::vector<float> except_result = {66.66, 2, 8, 3.6, 9.5, 6, 12.3, 15.9, 9.2, 97.8, 10};
  ASSERT_EQ(0, CompareOutputData(static_cast<float *>(outputs[0]->data()), except_result.data(),
                                 outputs[0]->ElementsNum(), 0.000001));
  delete kernel;
  DestroyTensors(inputs);
  DestroyTensors(outputs);
}

TEST_F(TestScatterNdAdd, Int32OneDims) {
  std::vector<lite::Tensor *> inputs;
  inputs.push_back(CreateTensor<int>(kNumberTypeInt32, {11}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));  // input_x
  inputs.push_back(
    CreateTensor<int>(kNumberTypeInt32, {15, 1}, {0, 1, 5, 3, 2, 2, 3, 2, 8, 6, 9, 3, 4, 6, 7}));  // indices
  inputs.push_back(
    CreateTensor<int>(kNumberTypeInt32, {15}, {{1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8}}));  // updates

  std::vector<lite::Tensor *> outputs;
  outputs.push_back(CreateTensor<int>(kNumberTypeInt32, {11}, {}));

  auto param = static_cast<ScatterNDParameter *>(malloc(sizeof(ScatterNDParameter)));
  memset(param, 0, sizeof(ScatterNDParameter));

  auto ctx = std::make_shared<lite::InnerContext>();
  ctx->thread_num_ = 2;
  ASSERT_EQ(ctx->Init(), RET_OK);
  param->op_parameter.thread_num_ = ctx->thread_num_;

  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt32, NHWC, schema::PrimitiveType_TensorScatterAdd};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), ctx.get(), desc);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(kernel->Prepare(), RET_OK);
  ASSERT_EQ(kernel->Run(), RET_OK);

  std::vector<int> except_result = {1, 2, 12, 15, 11, 7, 18, 15, 13, 15, 10};
  ASSERT_EQ(0, CompareOutputData(static_cast<int *>(outputs[0]->data()), except_result.data(),
                                 outputs[0]->ElementsNum(), 0.000001));
  delete kernel;
  DestroyTensors(inputs);
  DestroyTensors(outputs);
}

TEST_F(TestScatterNdAdd, Fp32ThreeDims) {
  std::vector<lite::Tensor *> inputs;
  inputs.push_back(CreateTensor<float>(
    kNumberTypeFloat32, {4, 4, 4},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));  // input_x
  inputs.push_back(CreateTensor<int>(kNumberTypeInt32, {4, 2}, {0, 1, 1, 3, 2, 2, 3, 2}));               // indices
  inputs.push_back(CreateTensor<float>(kNumberTypeFloat32, {4, 4},
                                       {1, 2, 1, 2, 1.5, 1, 0.5, 1, 1.2, 1, 1.7, -0.3, 3.3, 1, 5.6, 1}));  // updates

  std::vector<lite::Tensor *> outputs;
  outputs.push_back(CreateTensor<float>(kNumberTypeFloat32, {4, 4, 4}, {}));

  auto param = static_cast<ScatterNDParameter *>(malloc(sizeof(ScatterNDParameter)));
  memset(param, 0, sizeof(ScatterNDParameter));

  auto ctx = std::make_shared<lite::InnerContext>();
  ctx->thread_num_ = 2;
  ASSERT_EQ(ctx->Init(), RET_OK);
  param->op_parameter.thread_num_ = ctx->thread_num_;

  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, NHWC,
                            schema::PrimitiveType_TensorScatterAdd};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), ctx.get(), desc);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(kernel->Prepare(), RET_OK);
  ASSERT_EQ(kernel->Run(), RET_OK);

  std::vector<float> except_result = {1, 1, 1, 1, 2, 3, 2,   3, 1,   1, 1, 1, 1,   1, 1,   1, 1, 1, 1,   1, 1,   1,
                                      1, 1, 1, 1, 1, 1, 2.5, 2, 1.5, 2, 1, 1, 1,   1, 1,   1, 1, 1, 2.2, 2, 2.7, 0.7,
                                      1, 1, 1, 1, 1, 1, 1,   1, 1,   1, 1, 1, 4.3, 2, 6.6, 2, 1, 1, 1,   1};
  ASSERT_EQ(0, CompareOutputData(static_cast<float *>(outputs[0]->data()), except_result.data(),
                                 outputs[0]->ElementsNum(), 0.000001));
  delete kernel;
  DestroyTensors(inputs);
  DestroyTensors(outputs);
}
}  // namespace mindspore

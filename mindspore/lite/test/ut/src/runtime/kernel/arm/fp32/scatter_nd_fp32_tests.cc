/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

class TestScatterNdFp32 : public mindspore::CommonTest {
 public:
  TestScatterNdFp32() {}
};

TEST_F(TestScatterNdFp32, ScatterNd) {
  std::vector<lite::Tensor *> inputs;
  inputs.push_back(CreateTensor<int>(kNumberTypeInt32, {4, 1}, {4, 3, 1, 7}));      // index
  inputs.push_back(CreateTensor<float>(kNumberTypeFloat32, {4}, {9, 10, 11, 12}));  // update
  inputs.push_back(CreateTensor<int>(kNumberTypeInt32, {1}, {8}));                  // shape

  std::vector<lite::Tensor *> outputs;
  outputs.push_back(CreateTensor<float>(kNumberTypeFloat32, {8}, {}));

  auto param = static_cast<ScatterNDParameter *>(malloc(sizeof(ScatterNDParameter)));
  memset(param, 0, sizeof(ScatterNDParameter));

  auto ctx = std::make_shared<lite::InnerContext>();
  ctx->thread_num_ = 2;
  ASSERT_EQ(ctx->Init(), RET_OK);
  param->op_parameter.thread_num_ = ctx->thread_num_;

  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt32, NHWC, schema::PrimitiveType_ScatterNd};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), ctx.get(), desc);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(kernel->Prepare(), RET_OK);
  ASSERT_EQ(kernel->Run(), RET_OK);

  std::vector<float> except_result = {0, 11, 0, 10, 9, 0, 0, 12};
  ASSERT_EQ(0, CompareOutputData(static_cast<float *>(outputs[0]->data()), except_result.data(),
                                 outputs[0]->ElementsNum(), 0.000001));
  delete kernel;
  DestroyTensors(inputs);
  DestroyTensors(outputs);
}

TEST_F(TestScatterNdFp32, ScatterNdUpdate) {
  std::vector<lite::Tensor *> inputs;
  inputs.push_back(CreateTensor<float>(kNumberTypeFloat32, {8}, {1, 2, 3, 4, 5, 6, 7, 8}));  // input
  inputs.push_back(CreateTensor<int>(kNumberTypeInt32, {4, 1}, {4, 3, 1, 7}));               // index
  inputs.push_back(CreateTensor<float>(kNumberTypeFloat32, {4}, {9, 10, 11, 12}));           // update

  std::vector<lite::Tensor *> outputs;
  outputs.push_back(CreateTensor<float>(kNumberTypeFloat32, {8}, {}));

  auto param = static_cast<ScatterNDParameter *>(malloc(sizeof(ScatterNDParameter)));
  memset(param, 0, sizeof(ScatterNDParameter));

  auto ctx = std::make_shared<lite::InnerContext>();
  ctx->thread_num_ = 2;
  ASSERT_EQ(ctx->Init(), RET_OK);
  param->op_parameter.thread_num_ = ctx->thread_num_;

  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, NHWC, schema::PrimitiveType_ScatterNdUpdate};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), ctx.get(), desc);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(kernel->Prepare(), RET_OK);
  ASSERT_EQ(kernel->Run(), RET_OK);

  std::vector<float> except_result = {1, 11, 3, 10, 9, 6, 7, 12};
  ASSERT_EQ(0, CompareOutputData(static_cast<float *>(outputs[0]->data()), except_result.data(),
                                 outputs[0]->ElementsNum(), 0.000001));
  delete kernel;
  DestroyTensors(inputs);
  DestroyTensors(outputs);
}
}  // namespace mindspore

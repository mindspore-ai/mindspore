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
#include "src/litert/cxx_api/kernel_executor/kernel_executor.h"
#include "ops/abs.h"

class KernelExecutorTest : public mindspore::CommonTest {
 public:
  KernelExecutorTest();
  ~KernelExecutorTest();

 protected:
  std::shared_ptr<mindspore::ops::Abs> op_ = std::make_shared<mindspore::ops::Abs>();
  std::shared_ptr<mindspore::Context> context_ = std::make_shared<mindspore::Context>();
  float *input_data_;
};

KernelExecutorTest::KernelExecutorTest() {
  auto cpu_context = std::make_shared<mindspore::CPUDeviceInfo>();
  context_->MutableDeviceInfo().push_back(cpu_context);
  context_->SetThreadNum(1);

  input_data_ = new float[12]{-1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12};
}

KernelExecutorTest::~KernelExecutorTest() { delete[] input_data_; }

TEST_F(KernelExecutorTest, TestBuild) {
  auto kernel_executor = std::make_shared<mindspore::KernelExecutor>();
  std::vector<mindspore::MSTensor> inputs_abs;
  mindspore::MSTensor tensor_abs("Abs", mindspore::DataType::kNumberTypeFloat32, {1, 3, 2, 3},
                                 reinterpret_cast<void *>(input_data_), 12 * sizeof(float));
  inputs_abs.emplace_back(tensor_abs);
  ASSERT_EQ(kernel_executor->Build(nullptr, {}, nullptr), mindspore::kLiteNullptr);
  ASSERT_EQ(kernel_executor->Build(op_, {}, nullptr), mindspore::kLiteError);
  ASSERT_EQ(kernel_executor->Build(op_, inputs_abs, nullptr), mindspore::kLiteNullptr);
  ASSERT_EQ(kernel_executor->Build(op_, inputs_abs, context_), mindspore::kSuccess);
}

TEST_F(KernelExecutorTest, TestResize) {
  auto kernel_executor = std::make_shared<mindspore::KernelExecutor>();
  std::vector<mindspore::MSTensor> inputs_abs;
  mindspore::MSTensor tensor_abs("Abs", mindspore::DataType::kNumberTypeFloat32, {1, 3, 2, 2},
                                 reinterpret_cast<void *>(input_data_), 12 * sizeof(float));
  inputs_abs.emplace_back(tensor_abs);

  std::vector<mindspore::MSTensor> inputs_abs_resize;
  mindspore::MSTensor tensor_abs_resize("Abs", mindspore::DataType::kNumberTypeFloat32, {1, 4, 3},
                                        reinterpret_cast<void *>(input_data_), 12 * sizeof(float));
  inputs_abs.emplace_back(tensor_abs_resize);

  ASSERT_EQ(kernel_executor->ReSize({}), mindspore::kLiteNullptr);
  kernel_executor->Build(op_, inputs_abs, context_);
  ASSERT_EQ(kernel_executor->ReSize({}), mindspore::kLiteError);
  ASSERT_EQ(kernel_executor->ReSize(inputs_abs_resize), mindspore::kSuccess);
}

TEST_F(KernelExecutorTest, TestExecute) {
  auto kernel_executor = std::make_shared<mindspore::KernelExecutor>();
  std::vector<mindspore::MSTensor> inputs_abs;
  std::vector<mindspore::MSTensor> outputs_abs;
  mindspore::MSTensor tensor_abs("Abs", mindspore::DataType::kNumberTypeFloat32, {1, 3, 2, 2},
                                 reinterpret_cast<void *>(input_data_), 12 * sizeof(float));
  inputs_abs.emplace_back(tensor_abs);

  ASSERT_EQ(kernel_executor->Execute(inputs_abs, &outputs_abs), mindspore::kLiteNullptr);
  kernel_executor->Build(nullptr, inputs_abs, nullptr);
  ASSERT_EQ(kernel_executor->Execute(inputs_abs, &outputs_abs), mindspore::kLiteNullptr);

  kernel_executor->Build(op_, inputs_abs, context_);
  ASSERT_EQ(kernel_executor->Execute(inputs_abs, &outputs_abs), mindspore::kSuccess);
  float correct[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs_abs[0].MutableData()), correct,
                                 outputs_abs[0].ElementNum(), 0.0001));
}

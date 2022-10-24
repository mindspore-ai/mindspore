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
#include <vector>
#include "common/common_test.h"
#include "include/registry/register_kernel_interface.h"
#include "include/registry/register_kernel.h"
#include "src/litert/cxx_api/kernel_executor/kernel_executor.h"
#include "ops/add.h"
#include "ops/transpose.h"
#include "ops/arg_max.h"
#include "ops/batch_norm.h"
#include "ops/conv2d.h"
#include "ops/mat_mul.h"
#include "ops/topk.h"
#include "ops/arg_min.h"
#include "ops/avg_pool.h"
#include "ops/ceil.h"
#include "ops/concat.h"
#include "ops/conv2d_transpose.h"
#include "ops/flatten.h"
#include "ops/gather.h"
#include "ops/gather_nd.h"
#include "ops/maximum.h"
#include "ops/max_pool.h"
#include "ops/minimum.h"
#include "ops/mul.h"
#include "ops/pad.h"
#include "ops/prelu.h"
#include "ops/reshape.h"
#include "ops/softmax.h"
#include "ops/strided_slice.h"
#include "ops/abs.h"
#include "ops/div.h"
#include "ops/equal.h"
#include "ops/relu.h"
#include "ops/base_operator.h"
#include "ops/sigmoid.h"
#include "ops/addn.h"

namespace mindspore {
class KernelExecutorTest : public mindspore::CommonTest {
 public:
  KernelExecutorTest();
  ~KernelExecutorTest() = default;

 protected:
  std::shared_ptr<mindspore::KernelExecutor> kernel_executor_;
  std::shared_ptr<mindspore::Context> context_;
  std::vector<float> input_data_;
};

KernelExecutorTest::KernelExecutorTest() {
  kernel_executor_ = std::make_shared<mindspore::KernelExecutor>();
  context_ = std::make_shared<mindspore::Context>();
  auto cpu_context = std::make_shared<mindspore::CPUDeviceInfo>();
  context_->MutableDeviceInfo().push_back(cpu_context);
  context_->SetThreadNum(1);

  input_data_ = {-1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12};
}

namespace {
const auto kFloat32 = DataType::kNumberTypeFloat32;
class CustomAddKernel : public kernel::Kernel {
 public:
  CustomAddKernel(const std::vector<MSTensor> &inputs, const std::vector<MSTensor> &outputs,
                  const schema::Primitive *primitive, const mindspore::Context *ctx)
      : Kernel(inputs, outputs, primitive, ctx) {}
  ~CustomAddKernel() = default;

  int Prepare() override { return static_cast<int>(kSuccess); }

  int Execute() override {
    const float *in0 = static_cast<const float *>(inputs_[0].Data().get());
    const float *in1 = static_cast<const float *>(inputs_[1].Data().get());
    float *out = static_cast<float *>(outputs_[0].MutableData());
    auto num = outputs_[0].ElementNum();
    for (int i = 0; i < num; ++i) {
      out[i] = in0[i] + in1[i];
    }
    return static_cast<int>(kSuccess);
  }
  int ReSize() override { return static_cast<int>(kSuccess); }
};

std::shared_ptr<kernel::Kernel> CustomAddCreator(const std::vector<MSTensor> &inputs,
                                                 const std::vector<MSTensor> &outputs,
                                                 const schema::Primitive *primitive, const mindspore::Context *ctx) {
  return std::make_shared<CustomAddKernel>(inputs, outputs, primitive, ctx);
}
REGISTER_CUSTOM_KERNEL(CPU, Tutorial, kFloat32, Custom_Add, CustomAddCreator)

class CustomAddInfer : public kernel::KernelInterface {
 public:
  CustomAddInfer() = default;
  ~CustomAddInfer() = default;
  Status Infer(std::vector<mindspore::MSTensor> *inputs, std::vector<mindspore::MSTensor> *outputs,
               const schema::Primitive *primitive) override {
    (*outputs)[0].SetFormat((*inputs)[0].format());
    (*outputs)[0].SetDataType((*inputs)[0].DataType());
    (*outputs)[0].SetShape((*inputs)[0].Shape());
    return kSuccess;
  }
};
std::shared_ptr<kernel::KernelInterface> CustomAddInferCreator() { return std::make_shared<CustomAddInfer>(); }
REGISTER_CUSTOM_KERNEL_INTERFACE(CustomOpTurial, Custom_Add, CustomAddInferCreator)
}  // namespace

TEST_F(KernelExecutorTest, TestBuild) {
  auto op = std::make_shared<ops::Abs>();
  std::vector<mindspore::MSTensor> inputs_abs;
  mindspore::MSTensor tensor_abs("Abs", mindspore::DataType::kNumberTypeFloat32, {1, 3, 2, 3},
                                 reinterpret_cast<void *>(input_data_.data()), 12 * sizeof(float));
  inputs_abs.emplace_back(tensor_abs);
  ASSERT_EQ(kernel_executor_->Build(nullptr, {}, nullptr), mindspore::kLiteNullptr);
  ASSERT_EQ(kernel_executor_->Build(op, {}, nullptr), mindspore::kLiteError);
  ASSERT_EQ(kernel_executor_->Build(op, inputs_abs, nullptr), mindspore::kLiteNullptr);
  ASSERT_EQ(kernel_executor_->Build(op, inputs_abs, context_), mindspore::kSuccess);

  auto addn = std::make_shared<ops::AddN>();
  ASSERT_EQ(kernel_executor_->Build(addn, inputs_abs, context_), mindspore::kLiteError);
  tensor_abs.SetDataType(mindspore::DataType::kNumberTypeInt8);
  ASSERT_EQ(kernel_executor_->Build(op, inputs_abs, context_), mindspore::kLiteError);
  tensor_abs.SetDataType(mindspore::DataType::kNumberTypeFloat16);
  ASSERT_EQ(kernel_executor_->Build(op, inputs_abs, context_), mindspore::kLiteError);
}

TEST_F(KernelExecutorTest, TestResize) {
  auto op = std::make_shared<ops::Abs>();
  std::vector<mindspore::MSTensor> inputs_abs;
  mindspore::MSTensor tensor_abs("Abs", mindspore::DataType::kNumberTypeFloat32, {1, 3, 2, 2},
                                 reinterpret_cast<void *>(input_data_.data()), 12 * sizeof(float));
  inputs_abs.emplace_back(tensor_abs);

  std::vector<mindspore::MSTensor> inputs_abs_resize;
  mindspore::MSTensor tensor_abs_resize("Abs", mindspore::DataType::kNumberTypeFloat32, {1, 4, 3},
                                        reinterpret_cast<void *>(input_data_.data()), 12 * sizeof(float));
  inputs_abs_resize.emplace_back(tensor_abs_resize);

  ASSERT_EQ(kernel_executor_->ReSize({}), mindspore::kLiteNullptr);
  kernel_executor_->Build(nullptr, {}, nullptr);
  ASSERT_EQ(kernel_executor_->ReSize({}), mindspore::kLiteNullptr);
  kernel_executor_->Build(op, inputs_abs, context_);
  ASSERT_EQ(kernel_executor_->ReSize({}), mindspore::kLiteError);
  ASSERT_EQ(kernel_executor_->ReSize(inputs_abs_resize), mindspore::kSuccess);
}

TEST_F(KernelExecutorTest, TestExecute) {
  auto op = std::make_shared<ops::Abs>();
  std::vector<mindspore::MSTensor> inputs_abs;
  std::vector<mindspore::MSTensor> outputs_abs;
  mindspore::MSTensor tensor_abs("Abs", mindspore::DataType::kNumberTypeFloat32, {1, 3, 2, 2},
                                 reinterpret_cast<void *>(input_data_.data()), 12 * sizeof(float));
  inputs_abs.emplace_back(tensor_abs);

  ASSERT_EQ(kernel_executor_->Execute(inputs_abs, &outputs_abs), mindspore::kLiteNullptr);
  kernel_executor_->Build(nullptr, inputs_abs, nullptr);
  ASSERT_EQ(kernel_executor_->Execute(inputs_abs, &outputs_abs), mindspore::kLiteNullptr);

  kernel_executor_->Build(op, inputs_abs, context_);
  ASSERT_EQ(kernel_executor_->Execute(inputs_abs, nullptr), mindspore::kLiteNullptr);
  ASSERT_EQ(kernel_executor_->Execute(inputs_abs, &outputs_abs), mindspore::kSuccess);
  float correct[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs_abs[0].MutableData()), correct,
                                 outputs_abs[0].ElementNum(), 0.0001));

  std::vector<mindspore::MSTensor> inputs_other;
  mindspore::MSTensor tensor_other("other", mindspore::DataType::kNumberTypeFloat32, {1, 3, 2, 2},
                                   reinterpret_cast<void *>(input_data_.data()), 12 * sizeof(float));
  inputs_other.emplace_back(tensor_other);
  tensor_other.SetShape({3, 1, 2, 2});
  ASSERT_EQ(kernel_executor_->Execute(inputs_other, &outputs_abs), mindspore::kLiteError);
  tensor_other.SetShape({1, 3, 4});
  ASSERT_EQ(kernel_executor_->Execute(inputs_other, &outputs_abs), mindspore::kLiteError);
  tensor_other.SetFormat(mindspore::NCHW);
  ASSERT_EQ(kernel_executor_->Execute(inputs_other, &outputs_abs), mindspore::kLiteError);
  tensor_other.SetDataType(mindspore::DataType::kNumberTypeFloat16);
  ASSERT_EQ(kernel_executor_->Execute(inputs_other, &outputs_abs), mindspore::kLiteError);
  inputs_other.emplace_back(tensor_abs);
  ASSERT_EQ(kernel_executor_->Execute(inputs_other, &outputs_abs), mindspore::kLiteError);
}

TEST_F(KernelExecutorTest, TestCustom) {
  auto op = std::make_shared<ops::Custom>();
  auto kernel_executor = std::make_shared<mindspore::KernelExecutor>();
  std::vector<mindspore::MSTensor> inputs;
  std::vector<mindspore::MSTensor> outputs;
  mindspore::MSTensor tensor("Custom", mindspore::DataType::kNumberTypeFloat32, {1, 3, 2, 2},
                             reinterpret_cast<void *>(input_data_.data()), 12 * sizeof(float));
  inputs.emplace_back(tensor);
  inputs.emplace_back(tensor);

  ASSERT_EQ(kernel_executor->Build(op, inputs, context_, 0), mindspore::kLiteError);
  ASSERT_EQ(kernel_executor->Build(op, inputs, context_), mindspore::kLiteError);
  ASSERT_EQ(kernel_executor->Build(op, inputs, context_, 1), mindspore::kLiteNotSupport);

  std::map<std::string, std::vector<uint8_t>> custom_attrs;
  std::string input_num = std::to_string(2);
  std::vector<uint8_t> input_num_attr(input_num.begin(), input_num.end());
  custom_attrs["input_num"] = input_num_attr;
  std::string op_kind = "custom op";
  std::vector<uint8_t> op_kind_attr(op_kind.begin(), op_kind.end());
  custom_attrs["op_kind"] = op_kind_attr;
  op->Init("Custom_Add", custom_attrs);
  ASSERT_EQ(kernel_executor->Build(op, inputs, context_, 1), mindspore::kSuccess);
  ASSERT_EQ(kernel_executor->Execute(inputs, &outputs), mindspore::kSuccess);
  float correct[] = {-2, 4, -6, 8, -10, 12, -14, 16, -18, 20, -22, 24};
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs[0].MutableData()), correct, outputs[0].ElementNum(),
                                 0.0001));
}

TEST_F(KernelExecutorTest, TestRelu) {
  auto op = std::make_shared<ops::ReLU>();
  std::vector<mindspore::MSTensor> inputs;
  std::vector<mindspore::MSTensor> outputs;
  mindspore::MSTensor tensor("Relu", mindspore::DataType::kNumberTypeFloat32, {1, 3, 2, 2},
                             reinterpret_cast<void *>(input_data_.data()), 12 * sizeof(float));
  inputs.emplace_back(tensor);

  ASSERT_EQ(kernel_executor_->Build(op, inputs, context_), mindspore::kSuccess);
  ASSERT_EQ(kernel_executor_->Execute(inputs, &outputs), mindspore::kSuccess);
  float correct[] = {0, 2, 0, 4, 0, 6, 0, 8, 0, 10, 0, 12};
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs[0].MutableData()), correct, outputs[0].ElementNum(),
                                 0.0001));
}

TEST_F(KernelExecutorTest, TestSigmoid) {
  auto op = std::make_shared<ops::Sigmoid>();
  std::vector<mindspore::MSTensor> inputs;
  std::vector<mindspore::MSTensor> outputs;
  std::vector<float> input_data{1, 2, 3, 4, 5};
  mindspore::MSTensor tensor("Sigmoid", mindspore::DataType::kNumberTypeFloat32, {5},
                             reinterpret_cast<void *>(input_data.data()), 5 * sizeof(float));
  inputs.emplace_back(tensor);

  ASSERT_EQ(kernel_executor_->Build(op, inputs, context_), mindspore::kSuccess);
  ASSERT_EQ(kernel_executor_->Execute(inputs, &outputs), mindspore::kSuccess);
  float correct[] = {0.731059, 0.88081, 0.952574, 0.982015, 0.993307};
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs[0].MutableData()), correct, outputs[0].ElementNum(),
                                 0.0001));
}

TEST_F(KernelExecutorTest, TestAdd) {
  auto op = std::make_shared<ops::Add>();
  std::vector<mindspore::MSTensor> inputs;
  std::vector<mindspore::MSTensor> outputs;
  mindspore::MSTensor tensor("Add", mindspore::DataType::kNumberTypeFloat32, {1, 3, 2, 2},
                             reinterpret_cast<void *>(input_data_.data()), 12 * sizeof(float));
  inputs.emplace_back(tensor);
  inputs.emplace_back(tensor);

  ASSERT_EQ(kernel_executor_->Build(op, inputs, context_), mindspore::kSuccess);
  ASSERT_EQ(kernel_executor_->Execute(inputs, &outputs), mindspore::kSuccess);
  float correct[] = {-2, 4, -6, 8, -10, 12, -14, 16, -18, 20, -22, 24};
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs[0].MutableData()), correct, outputs[0].ElementNum(),
                                 0.0001));
}

TEST_F(KernelExecutorTest, TestArgMax) {
  auto op = std::make_shared<ops::Argmax>();
  op->Init(-1);
  std::vector<float> argmax_data{1, 20, 5, 67, 8, 9, 130, 24, 15};
  std::vector<mindspore::MSTensor> inputs;
  std::vector<mindspore::MSTensor> outputs;
  mindspore::MSTensor tensor("Argmax", mindspore::DataType::kNumberTypeFloat32, {3, 3},
                             reinterpret_cast<void *>(argmax_data.data()), 9 * sizeof(float));
  inputs.emplace_back(tensor);

  ASSERT_EQ(kernel_executor_->Build(op, inputs, context_), mindspore::kSuccess);
  ASSERT_EQ(kernel_executor_->Execute(inputs, &outputs), mindspore::kSuccess);
  int32_t correct[] = {1, 0, 0};
  ASSERT_EQ(
    0, CompareOutputData(reinterpret_cast<int32_t *>(outputs[0].MutableData()), correct, outputs[0].ElementNum(), 0));
}

TEST_F(KernelExecutorTest, TestArgMin) {
  auto op = std::make_shared<ops::ArgMin>();
  op->Init();
  std::vector<float> input_data{2.0, 3.1, 1.2};
  std::vector<mindspore::MSTensor> inputs;
  std::vector<mindspore::MSTensor> outputs;
  mindspore::MSTensor input("input", mindspore::DataType::kNumberTypeFloat32, {3},
                            reinterpret_cast<void *>(input_data.data()), 3 * sizeof(float));
  inputs.emplace_back(input);

  ASSERT_EQ(kernel_executor_->Build(op, inputs, context_), mindspore::kSuccess);
  ASSERT_EQ(kernel_executor_->Execute(inputs, &outputs), mindspore::kSuccess);
  int32_t correct[] = {2};
  ASSERT_EQ(
    0, CompareOutputData(reinterpret_cast<int32_t *>(outputs[0].MutableData()), correct, outputs[0].ElementNum(), 0));
}

TEST_F(KernelExecutorTest, TestAvgPool) {
  auto op = std::make_shared<ops::AvgPool>();
  op->Init({2, 2}, {1, 1});
  std::vector<float> input_data{0, 12, 24, 1, 13, 25, 2, 14, 26, 3, 15, 27, 4,  16, 28, 5,  17, 29,
                                6, 18, 30, 7, 19, 31, 8, 20, 32, 9, 21, 33, 10, 22, 34, 11, 23, 35};
  std::vector<mindspore::MSTensor> inputs;
  std::vector<mindspore::MSTensor> outputs;
  mindspore::MSTensor input("input", mindspore::DataType::kNumberTypeFloat32, {1, 3, 4, 3},
                            reinterpret_cast<void *>(input_data.data()), 36 * sizeof(float));
  input.SetFormat(mindspore::Format::NHWC);
  inputs.emplace_back(input);

  ASSERT_EQ(kernel_executor_->Build(op, inputs, context_), mindspore::kSuccess);
  ASSERT_EQ(kernel_executor_->Execute(inputs, &outputs), mindspore::kSuccess);
  float correct[] = {2.5, 14.5, 26.5, 3.5, 15.5, 27.5, 4.5, 16.5, 28.5,
                     6.5, 18.5, 30.5, 7.5, 19.5, 31.5, 8.5, 20.5, 32.5};
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs[0].MutableData()), correct, outputs[0].ElementNum(),
                                 0.0001));
}

TEST_F(KernelExecutorTest, TestBatchNorm) {
  auto op = std::make_shared<ops::BatchNorm>();
  op->Init(true);
  std::vector<float> input_data{1, 1, 1, 1};
  std::vector<mindspore::MSTensor> inputs;
  std::vector<mindspore::MSTensor> outputs;
  mindspore::MSTensor input("input", mindspore::DataType::kNumberTypeFloat32, {1, 1, 2, 2},
                            reinterpret_cast<void *>(input_data.data()), 4 * sizeof(float));
  mindspore::MSTensor scale("input", mindspore::DataType::kNumberTypeFloat32, {2},
                            reinterpret_cast<void *>(input_data.data()), 2 * sizeof(float));
  mindspore::MSTensor bias("input", mindspore::DataType::kNumberTypeFloat32, {2},
                           reinterpret_cast<void *>(input_data.data()), 2 * sizeof(float));
  mindspore::MSTensor mean("input", mindspore::DataType::kNumberTypeFloat32, {2},
                           reinterpret_cast<void *>(input_data.data()), 2 * sizeof(float));
  mindspore::MSTensor variance("input", mindspore::DataType::kNumberTypeFloat32, {2},
                               reinterpret_cast<void *>(input_data.data()), 2 * sizeof(float));
  inputs.emplace_back(input);
  inputs.emplace_back(scale);
  inputs.emplace_back(bias);
  inputs.emplace_back(mean);
  inputs.emplace_back(variance);

  ASSERT_EQ(kernel_executor_->Build(op, inputs, context_), mindspore::kSuccess);
  ASSERT_EQ(kernel_executor_->Execute(inputs, &outputs), mindspore::kSuccess);
  float correct[] = {1, 1, 1, 1};
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs[0].MutableData()), correct, outputs[0].ElementNum(),
                                 0.0001));
}

TEST_F(KernelExecutorTest, TestCeil) {
  auto op = std::make_shared<ops::Ceil>();
  std::vector<float> input_data{1.1, 2.5, -1.5};
  std::vector<mindspore::MSTensor> inputs;
  std::vector<mindspore::MSTensor> outputs;
  mindspore::MSTensor input("input", mindspore::DataType::kNumberTypeFloat32, {3},
                            reinterpret_cast<void *>(input_data.data()), 3 * sizeof(float));
  inputs.emplace_back(input);

  ASSERT_EQ(kernel_executor_->Build(op, inputs, context_), mindspore::kSuccess);
  ASSERT_EQ(kernel_executor_->Execute(inputs, &outputs), mindspore::kSuccess);
  float correct[] = {2, 3, -1};
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs[0].MutableData()), correct, outputs[0].ElementNum(),
                                 0.0001));
}

TEST_F(KernelExecutorTest, TestConcat) {
  auto op = std::make_shared<ops::Concat>();
  op->Init(1);
  std::vector<float> input_data{0, 1, 2, 1};
  std::vector<mindspore::MSTensor> inputs;
  std::vector<mindspore::MSTensor> outputs;
  mindspore::MSTensor input("input", mindspore::DataType::kNumberTypeFloat32, {2, 2},
                            reinterpret_cast<void *>(input_data.data()), 4 * sizeof(float));
  inputs.emplace_back(input);
  inputs.emplace_back(input);

  ASSERT_EQ(kernel_executor_->Build(op, inputs, context_), mindspore::kSuccess);
  ASSERT_EQ(kernel_executor_->Execute(inputs, &outputs), mindspore::kSuccess);
  float correct[] = {0, 1, 0, 1, 2, 1, 2, 1};
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs[0].MutableData()), correct, outputs[0].ElementNum(),
                                 0.0001));
}

TEST_F(KernelExecutorTest, TestConv2D) {
  auto op = std::make_shared<ops::Conv2D>();
  op->Init(32, {3, 3});
  std::vector<float> input_data(10 * 32 * 32 * 32, 1);
  std::vector<mindspore::MSTensor> inputs;
  std::vector<mindspore::MSTensor> outputs;
  mindspore::MSTensor input("input", mindspore::DataType::kNumberTypeFloat32, {10, 32, 32, 32},
                            reinterpret_cast<void *>(input_data.data()), 10 * 32 * 32 * 32 * sizeof(float));
  mindspore::MSTensor weight("input", mindspore::DataType::kNumberTypeFloat32, {32, 3, 3, 32},
                             reinterpret_cast<void *>(input_data.data()), 32 * 3 * 3 * 32 * sizeof(float));
  input.SetFormat(mindspore::Format::NHWC);
  weight.SetFormat(mindspore::Format::NHWC);
  inputs.emplace_back(input);
  inputs.emplace_back(weight);

  ASSERT_EQ(kernel_executor_->Build(op, inputs, context_), mindspore::kSuccess);
  ASSERT_EQ(kernel_executor_->Execute(inputs, &outputs), mindspore::kSuccess);
  std::vector<int64_t> shape{10, 30, 30, 32};
  ASSERT_EQ(outputs[0].Shape(), shape);
}

TEST_F(KernelExecutorTest, TestConv2DTranspose) {
  auto op = std::make_shared<ops::Conv2DTranspose>();
  op->Init(32, 32, {3, 3});
  std::vector<float> input_data(10 * 32 * 32 * 32, 1);
  std::vector<mindspore::MSTensor> inputs;
  std::vector<mindspore::MSTensor> outputs;
  mindspore::MSTensor input("input", mindspore::DataType::kNumberTypeFloat32, {10, 30, 30, 32},
                            reinterpret_cast<void *>(input_data.data()), 10 * 30 * 30 * 32 * sizeof(float));
  mindspore::MSTensor weight("input", mindspore::DataType::kNumberTypeFloat32, {32, 3, 3, 32},
                             reinterpret_cast<void *>(input_data.data()), 32 * 3 * 3 * 32 * sizeof(float));
  input.SetFormat(mindspore::Format::NHWC);
  weight.SetFormat(mindspore::Format::NHWC);
  inputs.emplace_back(input);
  inputs.emplace_back(weight);

  ASSERT_EQ(kernel_executor_->Build(op, inputs, context_), mindspore::kSuccess);
  ASSERT_EQ(kernel_executor_->Execute(inputs, &outputs), mindspore::kSuccess);
  std::vector<int64_t> shape{10, 32, 32, 32};
  ASSERT_EQ(outputs[0].Shape(), shape);
}

TEST_F(KernelExecutorTest, TestDiv) {
  auto op = std::make_shared<ops::Div>();
  std::vector<float> input_data{-4, 5, 6};
  std::vector<float> input_data2{3, 2, 3};
  std::vector<mindspore::MSTensor> inputs;
  std::vector<mindspore::MSTensor> outputs;
  mindspore::MSTensor input("input", mindspore::DataType::kNumberTypeFloat32, {3},
                            reinterpret_cast<void *>(input_data.data()), 3 * sizeof(float));
  mindspore::MSTensor input2("input", mindspore::DataType::kNumberTypeFloat32, {3},
                             reinterpret_cast<void *>(input_data2.data()), 3 * sizeof(float));
  inputs.emplace_back(input);
  inputs.emplace_back(input2);

  ASSERT_EQ(kernel_executor_->Build(op, inputs, context_), mindspore::kSuccess);
  ASSERT_EQ(kernel_executor_->Execute(inputs, &outputs), mindspore::kSuccess);
  float correct[] = {-1.33333, 2.5, 2};
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs[0].MutableData()), correct, outputs[0].ElementNum(),
                                 0.0001));
}

TEST_F(KernelExecutorTest, TestEqual) {
  auto op = std::make_shared<ops::Equal>();
  std::vector<float> input_data{1, 2, 3};
  std::vector<float> input_data2{2};
  std::vector<mindspore::MSTensor> inputs;
  std::vector<mindspore::MSTensor> outputs;
  mindspore::MSTensor input("input", mindspore::DataType::kNumberTypeFloat32, {3},
                            reinterpret_cast<void *>(input_data.data()), 3 * sizeof(float));
  mindspore::MSTensor input2("input", mindspore::DataType::kNumberTypeFloat32, {1},
                             reinterpret_cast<void *>(input_data2.data()), 1 * sizeof(float));
  inputs.emplace_back(input);
  inputs.emplace_back(input2);

  ASSERT_EQ(kernel_executor_->Build(op, inputs, context_), mindspore::kSuccess);
  ASSERT_EQ(kernel_executor_->Execute(inputs, &outputs), mindspore::kSuccess);
}

TEST_F(KernelExecutorTest, TestFlatten) {
  auto op = std::make_shared<ops::Flatten>();
  std::vector<float> input_data(24, 1);
  std::vector<mindspore::MSTensor> inputs;
  std::vector<mindspore::MSTensor> outputs;
  mindspore::MSTensor input("input", mindspore::DataType::kNumberTypeFloat32, {1, 2, 3, 4},
                            reinterpret_cast<void *>(input_data.data()), 24 * sizeof(float));
  inputs.emplace_back(input);

  ASSERT_EQ(kernel_executor_->Build(op, inputs, context_), mindspore::kSuccess);
  ASSERT_EQ(kernel_executor_->Execute(inputs, &outputs), mindspore::kSuccess);
  std::vector<int64_t> shape{1, 24};
  ASSERT_EQ(outputs[0].Shape(), shape);
}

TEST_F(KernelExecutorTest, TestGather) {
  auto op = std::make_shared<ops::Gather>();
  std::vector<float> input_data{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<float> input_data2{0, 2, 4, 2, 6};
  std::vector<mindspore::MSTensor> inputs;
  std::vector<mindspore::MSTensor> outputs;
  mindspore::MSTensor input("input", mindspore::DataType::kNumberTypeFloat32, {3, 4},
                            reinterpret_cast<void *>(input_data.data()), 12 * sizeof(float));
  mindspore::MSTensor input2("input", mindspore::DataType::kNumberTypeFloat32, {2},
                             reinterpret_cast<void *>(input_data2.data()), 2 * sizeof(float));
  mindspore::MSTensor input3("input", mindspore::DataType::kNumberTypeFloat32, {1},
                             reinterpret_cast<void *>(input_data2.data()), 1 * sizeof(float));
  inputs.emplace_back(input);
  inputs.emplace_back(input2);
  inputs.emplace_back(input3);

  ASSERT_EQ(kernel_executor_->Build(op, inputs, context_), mindspore::kSuccess);
  ASSERT_EQ(kernel_executor_->Execute(inputs, &outputs), mindspore::kSuccess);
  float correct[] = {1, 2, 3, 4, 9, 10, 11, 12};
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs[0].MutableData()), correct, outputs[0].ElementNum(),
                                 0.0001));
}

TEST_F(KernelExecutorTest, TestGatherNd) {
  auto op = std::make_shared<ops::GatherNd>();
  std::vector<float> input_data{-0.1, 0.3, 3.6, 0.4, 0.5, -3.2};
  std::vector<int32_t> input_data2{0, 0, 1, 1};
  std::vector<mindspore::MSTensor> inputs;
  std::vector<mindspore::MSTensor> outputs;
  mindspore::MSTensor input("input", mindspore::DataType::kNumberTypeFloat32, {2, 3},
                            reinterpret_cast<void *>(input_data.data()), 6 * sizeof(float));
  mindspore::MSTensor input2("input", mindspore::DataType::kNumberTypeInt32, {2, 2},
                             reinterpret_cast<void *>(input_data2.data()), 4 * sizeof(int32_t));
  inputs.emplace_back(input);
  inputs.emplace_back(input2);

  ASSERT_EQ(kernel_executor_->Build(op, inputs, context_), mindspore::kSuccess);
  ASSERT_EQ(kernel_executor_->Execute(inputs, &outputs), mindspore::kSuccess);
  float correct[] = {-0.1, 0.5};
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs[0].MutableData()), correct, outputs[0].ElementNum(),
                                 0.0001));
}

TEST_F(KernelExecutorTest, TestMatMul) {
  auto op = std::make_shared<ops::MatMul>();
  std::vector<float> input_data(12, 1);
  std::vector<mindspore::MSTensor> inputs;
  std::vector<mindspore::MSTensor> outputs;
  mindspore::MSTensor input("input", mindspore::DataType::kNumberTypeFloat32, {1, 3},
                            reinterpret_cast<void *>(input_data.data()), 3 * sizeof(float));
  mindspore::MSTensor input2("input", mindspore::DataType::kNumberTypeFloat32, {3, 4},
                             reinterpret_cast<void *>(input_data.data()), 12 * sizeof(float));
  inputs.emplace_back(input);
  inputs.emplace_back(input2);

  ASSERT_EQ(kernel_executor_->Build(op, inputs, context_), mindspore::kSuccess);
  ASSERT_EQ(kernel_executor_->Execute(inputs, &outputs), mindspore::kSuccess);
  float correct[] = {3, 3, 3, 3};
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs[0].MutableData()), correct, outputs[0].ElementNum(),
                                 0.0001));
}

TEST_F(KernelExecutorTest, TestMaximum) {
  auto op = std::make_shared<ops::Maximum>();
  std::vector<float> input_data{1, 5, 3};
  std::vector<float> input_data2{4, 2, 6};
  std::vector<mindspore::MSTensor> inputs;
  std::vector<mindspore::MSTensor> outputs;
  mindspore::MSTensor input("input", mindspore::DataType::kNumberTypeFloat32, {3},
                            reinterpret_cast<void *>(input_data.data()), 3 * sizeof(float));
  mindspore::MSTensor input2("input", mindspore::DataType::kNumberTypeFloat32, {3},
                             reinterpret_cast<void *>(input_data2.data()), 3 * sizeof(float));
  inputs.emplace_back(input);
  inputs.emplace_back(input2);

  ASSERT_EQ(kernel_executor_->Build(op, inputs, context_), mindspore::kSuccess);
  ASSERT_EQ(kernel_executor_->Execute(inputs, &outputs), mindspore::kSuccess);
  float correct[] = {4, 5, 6};
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs[0].MutableData()), correct, outputs[0].ElementNum(),
                                 0.0001));
}

TEST_F(KernelExecutorTest, TestMaxPool) {
  auto op = std::make_shared<ops::MaxPool>();
  op->Init({2, 2}, {1, 1});
  std::vector<float> input_data{0, 12, 24, 1, 13, 25, 2, 14, 26, 3, 15, 27, 4,  16, 28, 5,  17, 29,
                                6, 18, 30, 7, 19, 31, 8, 20, 32, 9, 21, 33, 10, 22, 34, 11, 23, 35};
  std::vector<mindspore::MSTensor> inputs;
  std::vector<mindspore::MSTensor> outputs;
  mindspore::MSTensor input("input", mindspore::DataType::kNumberTypeFloat32, {1, 3, 4, 3},
                            reinterpret_cast<void *>(input_data.data()), 36 * sizeof(float));
  input.SetFormat(mindspore::Format::NHWC);
  inputs.emplace_back(input);

  ASSERT_EQ(kernel_executor_->Build(op, inputs, context_), mindspore::kSuccess);
  ASSERT_EQ(kernel_executor_->Execute(inputs, &outputs), mindspore::kSuccess);
  float correct[] = {5, 17, 29, 6, 18, 30, 7, 19, 31, 9, 21, 33, 10, 22, 34, 11, 23, 35};
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs[0].MutableData()), correct, outputs[0].ElementNum(),
                                 0.0001));
}

TEST_F(KernelExecutorTest, TestMinimum) {
  auto op = std::make_shared<ops::Minimum>();
  std::vector<float> input_data{1, 5, 3};
  std::vector<float> input_data2{4, 2, 6};
  std::vector<mindspore::MSTensor> inputs;
  std::vector<mindspore::MSTensor> outputs;
  mindspore::MSTensor input("input", mindspore::DataType::kNumberTypeFloat32, {3},
                            reinterpret_cast<void *>(input_data.data()), 3 * sizeof(float));
  mindspore::MSTensor input2("input", mindspore::DataType::kNumberTypeFloat32, {3},
                             reinterpret_cast<void *>(input_data2.data()), 3 * sizeof(float));
  inputs.emplace_back(input);
  inputs.emplace_back(input2);

  ASSERT_EQ(kernel_executor_->Build(op, inputs, context_), mindspore::kSuccess);
  ASSERT_EQ(kernel_executor_->Execute(inputs, &outputs), mindspore::kSuccess);
  float correct[] = {1, 2, 3};
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs[0].MutableData()), correct, outputs[0].ElementNum(),
                                 0.0001));
}

TEST_F(KernelExecutorTest, TestMul) {
  auto op = std::make_shared<ops::Mul>();
  std::vector<float> input_data{1, 2, 3};
  std::vector<float> input_data2{4, 5, 6};
  std::vector<mindspore::MSTensor> inputs;
  std::vector<mindspore::MSTensor> outputs;
  mindspore::MSTensor input("input", mindspore::DataType::kNumberTypeFloat32, {3},
                            reinterpret_cast<void *>(input_data.data()), 3 * sizeof(float));
  mindspore::MSTensor input2("input", mindspore::DataType::kNumberTypeFloat32, {3},
                             reinterpret_cast<void *>(input_data2.data()), 3 * sizeof(float));
  inputs.emplace_back(input);
  inputs.emplace_back(input2);

  ASSERT_EQ(kernel_executor_->Build(op, inputs, context_), mindspore::kSuccess);
  ASSERT_EQ(kernel_executor_->Execute(inputs, &outputs), mindspore::kSuccess);
  float correct[] = {4, 10, 18};
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs[0].MutableData()), correct, outputs[0].ElementNum(),
                                 0.0001));
}

TEST_F(KernelExecutorTest, TestPad) {
  auto op = std::make_shared<ops::Pad>();
  std::vector<float> input_data{-0.1, 0.3, 3.6, 0.4, 0.5, -3.2};
  std::vector<int32_t> input_data2{1, 2, 2, 1};
  std::vector<mindspore::MSTensor> inputs;
  std::vector<mindspore::MSTensor> outputs;
  mindspore::MSTensor input("input", mindspore::DataType::kNumberTypeFloat32, {2, 3},
                            reinterpret_cast<void *>(input_data.data()), 6 * sizeof(float));
  mindspore::MSTensor input2("input", mindspore::DataType::kNumberTypeInt32, {4},
                             reinterpret_cast<void *>(input_data2.data()), 4 * sizeof(int32_t));
  inputs.emplace_back(input);
  inputs.emplace_back(input2);

  ASSERT_EQ(kernel_executor_->Build(op, inputs, context_), mindspore::kSuccess);
  ASSERT_EQ(kernel_executor_->Execute(inputs, &outputs), mindspore::kSuccess);
  float correct[] = {0,   0,    0, 0, 0, 0, 0, 0, -0.1, 0.3, 3.6, 0, 0, 0, 0.4,
                     0.5, -3.2, 0, 0, 0, 0, 0, 0, 0,    0,   0,   0, 0, 0, 0};
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs[0].MutableData()), correct, outputs[0].ElementNum(),
                                 0.0001));
}

TEST_F(KernelExecutorTest, TestPReLU) {
  auto op = std::make_shared<ops::PReLU>();
  std::vector<float> input_data{-6, -4, -2, -5, -3, -1, 0, 2, 4, 1, 3, 5};
  std::vector<float> input_data2{0.1, 0.6, -0.3};
  std::vector<mindspore::MSTensor> inputs;
  std::vector<mindspore::MSTensor> outputs;
  mindspore::MSTensor input("input", mindspore::DataType::kNumberTypeFloat32, {2, 1, 2, 3},
                            reinterpret_cast<void *>(input_data.data()), 12 * sizeof(float));
  mindspore::MSTensor input2("input", mindspore::DataType::kNumberTypeFloat32, {3},
                             reinterpret_cast<void *>(input_data2.data()), 3 * sizeof(float));
  input.SetFormat(mindspore::Format::NHWC);
  inputs.emplace_back(input);
  inputs.emplace_back(input2);

  ASSERT_EQ(kernel_executor_->Build(op, inputs, context_), mindspore::kSuccess);
  ASSERT_EQ(kernel_executor_->Execute(inputs, &outputs), mindspore::kSuccess);
  float correct[] = {-0.6, -2.4, 0.6, -0.5, -1.8, 0.3, 0, 2, 4, 1, 3, 5};
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs[0].MutableData()), correct, outputs[0].ElementNum(),
                                 0.0001));
}

TEST_F(KernelExecutorTest, TestReshape) {
  auto op = std::make_shared<ops::Reshape>();
  std::vector<float> input_data{-0.1, 0.3, 3.6, 0.4, 0.5, -3.2};
  std::vector<int32_t> input_data2{3, 2};
  std::vector<mindspore::MSTensor> inputs;
  std::vector<mindspore::MSTensor> outputs;
  mindspore::MSTensor input("input", mindspore::DataType::kNumberTypeFloat32, {2, 3},
                            reinterpret_cast<void *>(input_data.data()), 6 * sizeof(float));
  mindspore::MSTensor input2("input", mindspore::DataType::kNumberTypeInt32, {2},
                             reinterpret_cast<void *>(input_data2.data()), 2 * sizeof(int32_t));
  inputs.emplace_back(input);
  inputs.emplace_back(input2);

  ASSERT_EQ(kernel_executor_->Build(op, inputs, context_), mindspore::kSuccess);
  ASSERT_EQ(kernel_executor_->Execute(inputs, &outputs), mindspore::kSuccess);
  float correct[] = {-0.1, 0.3, 3.6, 0.4, 0.5, -3.2};
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs[0].MutableData()), correct, outputs[0].ElementNum(),
                                 0.0001));
  std::vector<int64_t> shape{3, 2};
  ASSERT_EQ(outputs[0].Shape(), shape);
}

TEST_F(KernelExecutorTest, TestSoftmax) {
  auto op = std::make_shared<ops::Softmax>();
  op->Init();
  std::vector<float> input_data{1, 2, 3, 4, 5};
  std::vector<mindspore::MSTensor> inputs;
  std::vector<mindspore::MSTensor> outputs;
  mindspore::MSTensor input("input", mindspore::DataType::kNumberTypeFloat32, {1, 5},
                            reinterpret_cast<void *>(input_data.data()), 5 * sizeof(float));
  inputs.emplace_back(input);

  ASSERT_EQ(kernel_executor_->Build(op, inputs, context_), mindspore::kSuccess);
  ASSERT_EQ(kernel_executor_->Execute(inputs, &outputs), mindspore::kSuccess);
  float correct[] = {0.0116558, 0.0316853, 0.0861187, 0.234124, 0.636416};
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs[0].MutableData()), correct, outputs[0].ElementNum(),
                                 0.0001));
}

TEST_F(KernelExecutorTest, TestStridedSlice) {
  auto op = std::make_shared<ops::StridedSlice>();
  std::vector<float> input_data{1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6};
  std::vector<int32_t> input_data2{1, 0, 0};
  std::vector<int32_t> input_data3{2, 1, 3};
  std::vector<int32_t> input_data4{1, 1, 1};
  std::vector<mindspore::MSTensor> inputs;
  std::vector<mindspore::MSTensor> outputs;
  mindspore::MSTensor input("input", mindspore::DataType::kNumberTypeFloat32, {3, 2, 3},
                            reinterpret_cast<void *>(input_data.data()), 18 * sizeof(float));
  mindspore::MSTensor input2("input", mindspore::DataType::kNumberTypeInt32, {3},
                             reinterpret_cast<void *>(input_data2.data()), 3 * sizeof(int32_t));
  mindspore::MSTensor input3("input", mindspore::DataType::kNumberTypeInt32, {3},
                             reinterpret_cast<void *>(input_data3.data()), 3 * sizeof(int32_t));
  mindspore::MSTensor input4("input", mindspore::DataType::kNumberTypeInt32, {3},
                             reinterpret_cast<void *>(input_data4.data()), 3 * sizeof(int32_t));
  inputs.emplace_back(input);
  inputs.emplace_back(input2);
  inputs.emplace_back(input3);
  inputs.emplace_back(input4);

  ASSERT_EQ(kernel_executor_->Build(op, inputs, context_), mindspore::kSuccess);
  ASSERT_EQ(kernel_executor_->Execute(inputs, &outputs), mindspore::kSuccess);
  float correct[] = {3, 3, 3};
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs[0].MutableData()), correct, outputs[0].ElementNum(),
                                 0.0001));
}

TEST_F(KernelExecutorTest, TestTopK) {
  auto op = std::make_shared<ops::TopK>();
  op->Init(true);
  std::vector<float> input_data{1, 2, 3, 4, 5};
  std::vector<int32_t> input_data2{3};
  std::vector<mindspore::MSTensor> inputs;
  std::vector<mindspore::MSTensor> outputs;
  mindspore::MSTensor input("input", mindspore::DataType::kNumberTypeFloat32, {5},
                            reinterpret_cast<void *>(input_data.data()), 5 * sizeof(float));
  mindspore::MSTensor input2("input", mindspore::DataType::kNumberTypeInt32, {1},
                             reinterpret_cast<void *>(input_data2.data()), 1 * sizeof(int32_t));
  inputs.emplace_back(input);
  inputs.emplace_back(input2);

  ASSERT_EQ(kernel_executor_->Build(op, inputs, context_), mindspore::kSuccess);
  ASSERT_EQ(kernel_executor_->Execute(inputs, &outputs), mindspore::kSuccess);
  float correct[] = {5, 4, 3};
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs[0].MutableData()), correct, outputs[0].ElementNum(),
                                 0.0001));
}

TEST_F(KernelExecutorTest, TestTranspose) {
  auto op = std::make_shared<ops::Transpose>();
  std::vector<float> input_data{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<int32_t> input_data2{0, 2, 1};
  std::vector<mindspore::MSTensor> inputs;
  std::vector<mindspore::MSTensor> outputs;
  mindspore::MSTensor input("input", mindspore::DataType::kNumberTypeFloat32, {2, 2, 3},
                            reinterpret_cast<void *>(input_data.data()), 12 * sizeof(float));
  mindspore::MSTensor input2("input", mindspore::DataType::kNumberTypeInt32, {3},
                             reinterpret_cast<void *>(input_data2.data()), 3 * sizeof(int32_t));
  inputs.emplace_back(input);
  inputs.emplace_back(input2);

  ASSERT_EQ(kernel_executor_->Build(op, inputs, context_), mindspore::kSuccess);
  ASSERT_EQ(kernel_executor_->Execute(inputs, &outputs), mindspore::kSuccess);
  float correct[] = {1, 4, 2, 5, 3, 6, 7, 10, 8, 11, 9, 12};
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs[0].MutableData()), correct, outputs[0].ElementNum(),
                                 0.0001));
}
}  // namespace mindspore

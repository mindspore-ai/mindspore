/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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
#include <vector>
#include "common/common_test.h"
#include "ir/primitive.h"
#include "ir/value.h"
#define private public
#define protected public
#include "adam_cpu_kernel.h"
#undef private
#undef protected

namespace mindspore {
namespace kernel {
class AdamCpuKernelTest : public UT::Common {
 public:
  AdamCpuKernelTest() : adam_(std::make_shared<adam_cpu::AdamCpuKernelMod>()) {}

  void SetUp() override {
    kernel_tensor_inputs_.clear();
    kernel_tensor_outputs_.clear();
    primitive_ = std::make_shared<Primitive>("Adam");
  }

  KernelTensor *CreateKernelTensor(const std::vector<int64_t> &shape, const TypePtr &dtype) {
    auto shape_ab = std::make_shared<abstract::Shape>(shape);
    auto new_abstract = std::make_shared<abstract::AbstractTensor>(dtype, shape_ab);
    auto *res_tensor = new KernelTensor(new_abstract->GetShape(), new_abstract->GetType(), kValueAny);
    res_tensor->set_format(mindspore::Format::NCHW);
    return res_tensor;
  }

  void CreateInputKernelTensor(const std::vector<int64_t> &var_shape, const std::vector<int64_t> &m_shape,
                               const std::vector<int64_t> &v_shape, const std::vector<int64_t> &beta1_power_shape,
                               const std::vector<int64_t> &beta2_power_shape, const std::vector<int64_t> &lr_shape,
                               const std::vector<int64_t> &beta1_shape, const std::vector<int64_t> &beta2_shape,
                               const std::vector<int64_t> &epsilon_shape, const std::vector<int64_t> &grad_shape) {
    kernel_tensor_inputs_.clear();
    kernel_tensor_inputs_.push_back(CreateKernelTensor(var_shape, kFloat32));
    kernel_tensor_inputs_.push_back(CreateKernelTensor(m_shape, kFloat32));
    kernel_tensor_inputs_.push_back(CreateKernelTensor(v_shape, kFloat32));
    kernel_tensor_inputs_.push_back(CreateKernelTensor(beta1_power_shape, kFloat32));
    kernel_tensor_inputs_.push_back(CreateKernelTensor(beta2_power_shape, kFloat32));
    kernel_tensor_inputs_.push_back(CreateKernelTensor(lr_shape, kFloat32));
    kernel_tensor_inputs_.push_back(CreateKernelTensor(beta1_shape, kFloat32));
    kernel_tensor_inputs_.push_back(CreateKernelTensor(beta2_shape, kFloat32));
    kernel_tensor_inputs_.push_back(CreateKernelTensor(epsilon_shape, kFloat32));
    kernel_tensor_inputs_.push_back(CreateKernelTensor(grad_shape, kFloat32));
  }

  void CreateOutputKernelTensor(const std::vector<int64_t> &var_shape) {
    kernel_tensor_outputs_.clear();
    kernel_tensor_outputs_.push_back(CreateKernelTensor(var_shape, kFloat32));
    kernel_tensor_outputs_.push_back(CreateKernelTensor(var_shape, kFloat32));
    kernel_tensor_outputs_.push_back(CreateKernelTensor(var_shape, kFloat32));
  }

  std::vector<KernelTensor *> kernel_tensor_inputs_;
  std::vector<KernelTensor *> kernel_tensor_outputs_;
  PrimitivePtr primitive_;
  std::shared_ptr<adam_cpu::AdamCpuKernelMod> adam_;
};

/// Feature: Adam CPU kernel.
/// Description: Resize rejects moment or gradient tensors whose shapes differ from var.
/// Expectation: Resize fails before Launch can access mismatched buffers.
TEST_F(AdamCpuKernelTest, resize_failed_when_state_shape_differs_from_var) {
  const std::vector<int64_t> var_shape = {2, 2};
  CreateInputKernelTensor(var_shape, {1}, var_shape, {1}, {1}, {1}, {1}, {1}, {1}, var_shape);
  CreateOutputKernelTensor(var_shape);
  adam_->primitive_ = primitive_;
  ASSERT_TRUE(adam_->Init(kernel_tensor_inputs_, kernel_tensor_outputs_));
  EXPECT_EQ(adam_->Resize(kernel_tensor_inputs_, kernel_tensor_outputs_), KRET_RESIZE_FAILED);
}

/// Feature: Adam CPU kernel.
/// Description: Resize rejects scalar hyper-parameter tensors with no element.
/// Expectation: Resize fails before Launch can read beta1[0], beta2[0] or epsilon[0].
TEST_F(AdamCpuKernelTest, resize_failed_when_beta_or_epsilon_is_empty) {
  const std::vector<int64_t> var_shape = {2, 2};
  CreateInputKernelTensor(var_shape, var_shape, var_shape, {1}, {1}, {1}, {0}, {1}, {1}, var_shape);
  CreateOutputKernelTensor(var_shape);
  adam_->primitive_ = primitive_;
  ASSERT_TRUE(adam_->Init(kernel_tensor_inputs_, kernel_tensor_outputs_));
  EXPECT_EQ(adam_->Resize(kernel_tensor_inputs_, kernel_tensor_outputs_), KRET_RESIZE_FAILED);
}

/// Feature: Adam CPU kernel.
/// Description: Resize rejects inconsistent batch scalar shapes.
/// Expectation: Resize fails before Launch can read beta power tensors past their buffer end.
TEST_F(AdamCpuKernelTest, resize_failed_when_batch_scalar_shapes_differ) {
  const std::vector<int64_t> var_shape = {2, 2};
  primitive_->AddAttr("batch_rank", MakeValue<int64_t>(1));
  CreateInputKernelTensor(var_shape, var_shape, var_shape, {1}, {1}, {2}, {1}, {1}, {1}, var_shape);
  CreateOutputKernelTensor(var_shape);
  adam_->primitive_ = primitive_;
  ASSERT_TRUE(adam_->Init(kernel_tensor_inputs_, kernel_tensor_outputs_));
  EXPECT_EQ(adam_->Resize(kernel_tensor_inputs_, kernel_tensor_outputs_), KRET_RESIZE_FAILED);
}

/// Feature: Adam CPU kernel.
/// Description: Resize rejects a batched lr shape that does not match the batch prefix of var.
/// Expectation: Resize fails before Launch derives an invalid per-batch element count.
TEST_F(AdamCpuKernelTest, resize_failed_when_var_batch_shape_differs_from_lr) {
  const std::vector<int64_t> var_shape = {2, 2};
  const std::vector<int64_t> lr_shape = {3};
  primitive_->AddAttr("batch_rank", MakeValue<int64_t>(1));
  CreateInputKernelTensor(var_shape, var_shape, var_shape, lr_shape, lr_shape, lr_shape, {1}, {1}, {1}, var_shape);
  CreateOutputKernelTensor(var_shape);
  adam_->primitive_ = primitive_;
  ASSERT_TRUE(adam_->Init(kernel_tensor_inputs_, kernel_tensor_outputs_));
  EXPECT_EQ(adam_->Resize(kernel_tensor_inputs_, kernel_tensor_outputs_), KRET_RESIZE_FAILED);
}

/// Feature: Adam CPU kernel.
/// Description: Resize accepts batched scalar hyper-parameters whose shapes match lr.
/// Expectation: Resize succeeds and keeps per-batch element count consistent with var.
TEST_F(AdamCpuKernelTest, resize_success_when_batch_hyper_parameter_shapes_match_lr) {
  const std::vector<int64_t> var_shape = {2, 2};
  const std::vector<int64_t> lr_shape = {2};
  primitive_->AddAttr("batch_rank", MakeValue<int64_t>(1));
  CreateInputKernelTensor(var_shape, var_shape, var_shape, lr_shape, lr_shape, lr_shape, lr_shape, lr_shape, lr_shape,
                          var_shape);
  CreateOutputKernelTensor(var_shape);
  adam_->primitive_ = primitive_;
  ASSERT_TRUE(adam_->Init(kernel_tensor_inputs_, kernel_tensor_outputs_));
  EXPECT_EQ(adam_->Resize(kernel_tensor_inputs_, kernel_tensor_outputs_), KRET_OK);
  EXPECT_EQ(adam_->input_elements_, 2U);
}
}  // namespace kernel
}  // namespace mindspore

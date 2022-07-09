/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "ops/fused_sparse_ftrl.h"
#define private public
#define protected public
#include "plugin/device/cpu/kernel/sparse_apply_ftrl_cpu_kernel.h"
#undef private
#undef protected

namespace mindspore {
namespace kernel {
class FusedSparseFtrlCpuKernelTest : public UT::Common {
 public:
  FusedSparseFtrlCpuKernelTest() : sparse_ftrl_(std::make_shared<FusedSparseFtrlCpuKernelMod>()) {}

  void SetUp() override {
    sparse_ftrl_->lr_ = 0.001;
    sparse_ftrl_->l1_ = 0.0;
    sparse_ftrl_->l2_ = 0.0;
    sparse_ftrl_->lr_power_ = -0.5;
    var_.clear();
    accum_.clear();
    linear_.clear();
    grad_.clear();
    inputs_.clear();
    workspace_.clear();
    outputs_.clear();
  }

  AddressPtr CreateKernelAddress(void *addr) {
    auto kernel_addr = std::make_shared<Address>();
    kernel_addr->addr = addr;
    return kernel_addr;
  }

  void CreateInputAddress(std::vector<int64_t> &indices) {
    inputs_.push_back(CreateKernelAddress(var_.data()));
    inputs_.push_back(CreateKernelAddress(accum_.data()));
    inputs_.push_back(CreateKernelAddress(linear_.data()));
    inputs_.push_back(CreateKernelAddress(grad_.data()));
    inputs_.push_back(CreateKernelAddress(indices.data()));
  }

  void CreateWorkspaceAddress(std::vector<float> &new_grad, std::vector<int64_t> &new_indices,
                              std::vector<float> &tmp_grad, std::vector<int64_t> &tmp_indices) {
    workspace_.push_back(CreateKernelAddress(new_grad.data()));
    workspace_.push_back(CreateKernelAddress(new_indices.data()));
    workspace_.push_back(CreateKernelAddress(tmp_grad.data()));
    workspace_.push_back(CreateKernelAddress(tmp_indices.data()));
  }

  KernelTensorPtr CreateKernelTensor(const std::vector<int64_t> &shape, const TypePtr &dtype) {
    auto shape_ab = std::make_shared<abstract::Shape>(shape);
    auto new_abstract = std::make_shared<abstract::AbstractTensor>(dtype, shape_ab);
    TensorInfo tensor_info{mindspore::Format::NCHW, new_abstract, shape};
    KernelTensorPtr res_tensor = std::make_shared<KernelTensor>();
    res_tensor->SetTensorInfo(tensor_info);
    return res_tensor;
  }

  void CreateInputKernelTensor(const std::vector<int64_t> &var_shape, const std::vector<int64_t> &indices_shape) {
    std::vector<int64_t> grad_shape = var_shape;
    grad_shape[0] = indices_shape[0];
    kernel_tensor_inputs_.clear();
    kernel_tensor_inputs_.push_back(CreateKernelTensor(var_shape, kFloat32));
    kernel_tensor_inputs_.push_back(CreateKernelTensor(var_shape, kFloat32));
    kernel_tensor_inputs_.push_back(CreateKernelTensor(var_shape, kFloat32));
    kernel_tensor_inputs_.push_back(CreateKernelTensor(grad_shape, kFloat32));
    kernel_tensor_inputs_.push_back(CreateKernelTensor(indices_shape, kInt64));
  }

  void CreateOutputKernelTensor() {
    std::vector<int64_t> var_shape = {3, 3, 3};
    std::vector<int64_t> indices_shape = {3};
    std::vector<int64_t> grad_shape = {3, 3, 3};
    kernel_tensor_outputs_.clear();
    kernel_tensor_outputs_.push_back(CreateKernelTensor({1}, kFloat32));
    kernel_tensor_outputs_.push_back(CreateKernelTensor({1}, kFloat32));
    kernel_tensor_outputs_.push_back(CreateKernelTensor({1}, kFloat32));
  }

  std::vector<float> var_;
  std::vector<float> accum_;
  std::vector<float> linear_;
  std::vector<float> grad_;
  std::vector<AddressPtr> inputs_;
  std::vector<AddressPtr> workspace_;
  std::vector<AddressPtr> outputs_;
  std::vector<KernelTensorPtr> kernel_tensor_inputs_;
  std::vector<KernelTensorPtr> kernel_tensor_outputs_;
  std::shared_ptr<FusedSparseFtrlCpuKernelMod> sparse_ftrl_;
};

/// Feature: FusedSparseFtrl
/// Description: Run FusedSparseFtrl
/// Expectation: pass
TEST_F(FusedSparseFtrlCpuKernelTest, dense_test) {
  for (size_t i = 0; i < 3 * 3 * 3; ++i) {
    var_.push_back(1.0);
    accum_.push_back(1.0);
    linear_.push_back(1.0);
    grad_.push_back(1.0);
  }
  auto ops = std::make_shared<ops::FusedSparseFtrl>();
  ops->Init(0.001, 0, 0, -0.5);
  std::vector<int64_t> var_shape = {3, 3, 3};
  std::vector<int64_t> indices_shape = {3};
  CreateInputKernelTensor(var_shape, indices_shape);
  CreateOutputKernelTensor();
  sparse_ftrl_->Init(ops, kernel_tensor_inputs_, kernel_tensor_outputs_);
  sparse_ftrl_->Resize(ops, kernel_tensor_inputs_, kernel_tensor_outputs_, {});

  std::vector<int64_t> indices{0, 1, 2};
  CreateInputAddress(indices);
  std::vector<float> new_grad(3 * 3 * 3);
  std::vector<int64_t> new_indices(3);
  std::vector<float> tmp_grad(3 * 3 * 3);
  std::vector<int64_t> tmp_indices(3);
  CreateWorkspaceAddress(new_grad, new_indices, tmp_grad, tmp_indices);
  sparse_ftrl_->Launch(inputs_, workspace_, outputs_);
  for (size_t i = 0; i < 3 * 3 * 3; ++i) {
    EXPECT_TRUE(std::fabs(var_[i] - 0.291479) < 1e-6);
  }
}

/// Feature: FusedSparseFtrl
/// Description: Run FusedSparseFtrl
/// Expectation: pass
TEST_F(FusedSparseFtrlCpuKernelTest, sparse_test1) {
  for (size_t i = 0; i < 3 * 3 * 3; ++i) {
    var_.push_back(1.0);
    accum_.push_back(1.0);
    linear_.push_back(1.0);
  }
  for (size_t i = 0; i < 2 * 3 * 3; ++i) {
    grad_.push_back(1.0);
  }
  auto ops = std::make_shared<ops::FusedSparseFtrl>();
  ops->Init(0.001, 0, 0, -0.5);
  std::vector<int64_t> var_shape = {3, 3, 3};
  std::vector<int64_t> indices_shape = {2};
  CreateInputKernelTensor(var_shape, indices_shape);
  CreateOutputKernelTensor();
  sparse_ftrl_->Init(ops, kernel_tensor_inputs_, kernel_tensor_outputs_);
  sparse_ftrl_->Resize(ops, kernel_tensor_inputs_, kernel_tensor_outputs_, {});

  std::vector<int64_t> indices{0, 2};
  CreateInputAddress(indices);
  std::vector<float> new_grad(3 * 3 * 3);
  std::vector<int64_t> new_indices(3);
  std::vector<float> tmp_grad(3 * 3 * 3);
  std::vector<int64_t> tmp_indices(3);
  CreateWorkspaceAddress(new_grad, new_indices, tmp_grad, tmp_indices);
  sparse_ftrl_->Launch(inputs_, workspace_, outputs_);
  for (size_t i = 0; i < 3 * 3; ++i) {
    EXPECT_TRUE(std::fabs(var_[i] - 0.291479) < 1e-6);
  }
  for (size_t i = 3 * 3; i < 2 * 3 * 3; ++i) {
    EXPECT_EQ(var_[i], 1.0);
  }
  for (size_t i = 2 * 3 * 3; i < 3 * 3 * 3; ++i) {
    EXPECT_TRUE(std::fabs(var_[i] - 0.291479) < 1e-6);
  }
}

/// Feature: FusedSparseFtrl
/// Description: Run FusedSparseFtrl
/// Expectation: pass
TEST_F(FusedSparseFtrlCpuKernelTest, sparse_test2) {
  for (size_t i = 0; i < 3 * 3 * 3; ++i) {
    var_.push_back(1.0);
    accum_.push_back(1.0);
    linear_.push_back(1.0);
    grad_.push_back(1.0);
  }
  auto ops = std::make_shared<ops::FusedSparseFtrl>();
  ops->Init(0.001, 0, 0, -0.5);
  std::vector<int64_t> var_shape = {3, 3, 3};
  std::vector<int64_t> indices_shape = {3};
  CreateInputKernelTensor(var_shape, indices_shape);
  CreateOutputKernelTensor();
  sparse_ftrl_->Init(ops, kernel_tensor_inputs_, kernel_tensor_outputs_);
  sparse_ftrl_->Resize(ops, kernel_tensor_inputs_, kernel_tensor_outputs_, {});

  std::vector<int64_t> indices{2, 2, 1};
  CreateInputAddress(indices);
  std::vector<float> new_grad(3 * 3 * 3);
  std::vector<int64_t> new_indices(3);
  std::vector<float> tmp_grad(3 * 3 * 3);
  std::vector<int64_t> tmp_indices(3);
  CreateWorkspaceAddress(new_grad, new_indices, tmp_grad, tmp_indices);
  sparse_ftrl_->Launch(inputs_, workspace_, outputs_);
  for (size_t i = 0; i < 3 * 3; ++i) {
    EXPECT_EQ(var_[i], 1.0);
  }
  for (size_t i = 3 * 3; i < 2 * 3 * 3; ++i) {
    EXPECT_TRUE(std::fabs(var_[i] - 0.291479) < 1e-6);
  }
  for (size_t i = 2 * 3 * 3; i < 3 * 3 * 3; ++i) {
    EXPECT_TRUE(std::fabs(var_[i] - 0.551445) < 1e-6);
  }
}
}  // namespace kernel
}  // namespace mindspore

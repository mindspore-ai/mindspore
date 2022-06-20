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
#include "ops/fused_sparse_lazy_adam.h"
#define private public
#define protected public
#include "plugin/device/cpu/kernel/sparse_apply_lazy_adam_cpu_kernel.h"
#undef private
#undef protected

namespace mindspore {
namespace kernel {
class SparseApplyLazyAdamCpuKernelTest : public UT::Common {
 public:
  SparseApplyLazyAdamCpuKernelTest() : sparse_lazy_adam_(std::make_shared<SparseApplyLazyAdamCpuKernelMod>()) {}

  void SetUp() override {
    var_.clear();
    m_.clear();
    v_.clear();
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
    inputs_.push_back(CreateKernelAddress(m_.data()));
    inputs_.push_back(CreateKernelAddress(v_.data()));
    inputs_.push_back(CreateKernelAddress(&beta1_power_));
    inputs_.push_back(CreateKernelAddress(&beta2_power_));
    inputs_.push_back(CreateKernelAddress(&lr_));
    inputs_.push_back(CreateKernelAddress(&beta1_));
    inputs_.push_back(CreateKernelAddress(&beta2_));
    inputs_.push_back(CreateKernelAddress(&epsilon_));
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
    kernel_tensor_inputs_.push_back(CreateKernelTensor({1}, kFloat32));
    kernel_tensor_inputs_.push_back(CreateKernelTensor({1}, kFloat32));
    kernel_tensor_inputs_.push_back(CreateKernelTensor({1}, kFloat32));
    kernel_tensor_inputs_.push_back(CreateKernelTensor({1}, kFloat32));
    kernel_tensor_inputs_.push_back(CreateKernelTensor({1}, kFloat32));
    kernel_tensor_inputs_.push_back(CreateKernelTensor({1}, kFloat32));
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
  std::vector<float> m_;
  std::vector<float> v_;
  std::vector<float> grad_;
  std::vector<AddressPtr> inputs_;
  std::vector<AddressPtr> workspace_;
  std::vector<AddressPtr> outputs_;
  std::vector<KernelTensorPtr> kernel_tensor_inputs_;
  std::vector<KernelTensorPtr> kernel_tensor_outputs_;
  std::shared_ptr<SparseApplyLazyAdamCpuKernelMod> sparse_lazy_adam_;
  float beta1_power_ = 0.9;
  float beta2_power_ = 0.999;
  float lr_ = 0.001;
  float beta1_ = 0.9;
  float beta2_ = 0.999;
  float epsilon_ = 1e-8;
};

TEST_F(SparseApplyLazyAdamCpuKernelTest, dense_test) {
  for (size_t i = 0; i < 3 * 3 * 3; ++i) {
    var_.push_back(1.0);
    m_.push_back(1.0);
    v_.push_back(1.0);
    grad_.push_back(1.0);
  }
  auto ops = std::make_shared<ops::FusedSparseLazyAdam>();
  ops->Init();
  std::vector<int64_t> var_shape = {3, 3, 3};
  std::vector<int64_t> indices_shape = {3};
  CreateInputKernelTensor(var_shape, indices_shape);
  CreateOutputKernelTensor();
  sparse_lazy_adam_->Init(ops, kernel_tensor_inputs_, kernel_tensor_outputs_);
  sparse_lazy_adam_->Resize(ops, kernel_tensor_inputs_, kernel_tensor_outputs_, {});

  std::vector<int64_t> indices{0, 1, 2};
  CreateInputAddress(indices);
  std::vector<float> new_grad(3 * 3 * 3);
  std::vector<int64_t> new_indices(3);
  std::vector<float> tmp_grad(3 * 3 * 3);
  std::vector<int64_t> tmp_indices(3);
  CreateWorkspaceAddress(new_grad, new_indices, tmp_grad, tmp_indices);
  sparse_lazy_adam_->Launch(inputs_, workspace_, outputs_);
  for (size_t i = 0; i < 3 * 3 * 3; ++i) {
    EXPECT_TRUE(std::fabs(var_[i] - 0.999684) < 1e-6);
  }
}

TEST_F(SparseApplyLazyAdamCpuKernelTest, sparse_test1) {
  for (size_t i = 0; i < 3 * 3 * 3; ++i) {
    var_.push_back(1.0);
    m_.push_back(1.0);
    v_.push_back(1.0);
  }
  for (size_t i = 0; i < 2 * 3 * 3; ++i) {
    grad_.push_back(1.0);
  }
  auto ops = std::make_shared<ops::FusedSparseLazyAdam>();
  ops->Init();
  std::vector<int64_t> var_shape = {3, 3, 3};
  std::vector<int64_t> indices_shape = {2};
  CreateInputKernelTensor(var_shape, indices_shape);
  CreateOutputKernelTensor();
  sparse_lazy_adam_->Init(ops, kernel_tensor_inputs_, kernel_tensor_outputs_);
  sparse_lazy_adam_->Resize(ops, kernel_tensor_inputs_, kernel_tensor_outputs_, {});

  std::vector<int64_t> indices{0, 2};
  CreateInputAddress(indices);
  std::vector<float> new_grad(3 * 3 * 3);
  std::vector<int64_t> new_indices(3);
  std::vector<float> tmp_grad(3 * 3 * 3);
  std::vector<int64_t> tmp_indices(3);
  CreateWorkspaceAddress(new_grad, new_indices, tmp_grad, tmp_indices);
  sparse_lazy_adam_->Launch(inputs_, workspace_, outputs_);
  for (size_t i = 0; i < 3 * 3; ++i) {
    EXPECT_TRUE(std::fabs(var_[i] - 0.999684) < 1e-6);
  }
  for (size_t i = 3 * 3; i < 2 * 3 * 3; ++i) {
    EXPECT_EQ(var_[i], 1.0);
  }
  for (size_t i = 2 * 3 * 3; i < 3 * 3 * 3; ++i) {
    EXPECT_TRUE(std::fabs(var_[i] - 0.999684) < 1e-6);
  }
}

TEST_F(SparseApplyLazyAdamCpuKernelTest, sparse_test2) {
  for (size_t i = 0; i < 3 * 3 * 3; ++i) {
    var_.push_back(1.0);
    m_.push_back(1.0);
    v_.push_back(1.0);
    grad_.push_back(1.0);
  }
  auto ops = std::make_shared<ops::FusedSparseLazyAdam>();
  ops->Init();
  std::vector<int64_t> var_shape = {3, 3, 3};
  std::vector<int64_t> indices_shape = {3};
  CreateInputKernelTensor(var_shape, indices_shape);
  CreateOutputKernelTensor();
  sparse_lazy_adam_->Init(ops, kernel_tensor_inputs_, kernel_tensor_outputs_);
  sparse_lazy_adam_->Resize(ops, kernel_tensor_inputs_, kernel_tensor_outputs_, {});

  std::vector<int64_t> indices{2, 2, 1};
  CreateInputAddress(indices);
  std::vector<float> new_grad(3 * 3 * 3);
  std::vector<int64_t> new_indices(3);
  std::vector<float> tmp_grad(3 * 3 * 3);
  std::vector<int64_t> tmp_indices(3);
  CreateWorkspaceAddress(new_grad, new_indices, tmp_grad, tmp_indices);
  sparse_lazy_adam_->Launch(inputs_, workspace_, outputs_);
  for (size_t i = 0; i < 3 * 3; ++i) {
    EXPECT_EQ(var_[i], 1.0);
  }
  for (size_t i = 3 * 3; i < 2 * 3 * 3; ++i) {
    EXPECT_TRUE(std::fabs(var_[i] - 0.999684) < 1e-6);
  }
  for (size_t i = 2 * 3 * 3; i < 3 * 3 * 3; ++i) {
    EXPECT_TRUE(std::fabs(var_[i] - 0.999653) < 1e-6);
  }
}
}  // namespace kernel
}  // namespace mindspore

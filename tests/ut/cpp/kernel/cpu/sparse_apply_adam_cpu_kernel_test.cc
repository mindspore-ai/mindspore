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
#include "common/common_test.h"
#define private public
#define protected public
#include "kernel/cpu/sparse_apply_adam_cpu_kernel.h"
#undef private
#undef protected

namespace mindspore {
namespace kernel {
class SparseApplyAdamCpuKernelTest : public UT::Common {
 public:
  SparseApplyAdamCpuKernelTest() : sparse_adam_(std::make_shared<SparseApplyAdamCPUKernel>()) {}

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

  void CreateInputAddress(std::vector<int> &indices) {
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

  void CreateWorkspaceAddress(std::vector<float> &new_grad, std::vector<int> &new_indices, std::vector<float> &m_t) {
    workspace_.push_back(CreateKernelAddress(new_grad.data()));
    workspace_.push_back(CreateKernelAddress(new_indices.data()));
    workspace_.push_back(CreateKernelAddress(m_t.data()));
  }

  std::vector<float> var_;
  std::vector<float> m_;
  std::vector<float> v_;
  std::vector<float> grad_;
  std::vector<AddressPtr> inputs_;
  std::vector<AddressPtr> workspace_;
  std::vector<AddressPtr> outputs_;
  std::shared_ptr<SparseApplyAdamCPUKernel> sparse_adam_;
  float beta1_power_ = 0.9;
  float beta2_power_ = 0.999;
  float lr_ = 0.001;
  float beta1_ = 0.9;
  float beta2_ = 0.999;
  float epsilon_ = 1e-8;
};

TEST_F(SparseApplyAdamCpuKernelTest, dense_test) {
  for (size_t i = 0; i < 3 * 3 * 3; ++i) {
    var_.push_back(1.0);
    m_.push_back(1.0);
    v_.push_back(1.0);
    grad_.push_back(1.0);
  }
  sparse_adam_->indices_size_ = 3;
  sparse_adam_->var_first_dim_size_ = 3;
  sparse_adam_->var_outer_dim_size_ = 9;

  std::vector<int> indices{0, 1, 2};
  CreateInputAddress(indices);
  std::vector<float> new_grad(3 * 3 * 3);
  std::vector<int> new_indices(3);
  std::vector<float> m_t(3 * 3 * 3);
  CreateWorkspaceAddress(new_grad, new_indices, m_t);
  sparse_adam_->Launch(inputs_, workspace_, outputs_);
  for (size_t i = 0; i < 3 * 3 * 3; ++i) {
    EXPECT_TRUE(std::fabs(var_[i] - 0.999684) < 1e-6);
  }
}

TEST_F(SparseApplyAdamCpuKernelTest, sparse_test1) {
  for (size_t i = 0; i < 3 * 3 * 3; ++i) {
    var_.push_back(1.0);
    m_.push_back(1.0);
    v_.push_back(1.0);
  }
  for (size_t i = 0; i < 2 * 3 * 3; ++i) {
    grad_.push_back(1.0);
  }
  sparse_adam_->indices_size_ = 2;
  sparse_adam_->var_first_dim_size_ = 3;
  sparse_adam_->var_outer_dim_size_ = 9;

  std::vector<int> indices{0, 2};
  CreateInputAddress(indices);
  std::vector<float> new_grad(3 * 3 * 3);
  std::vector<int> new_indices(3);
  std::vector<float> m_t(3 * 3 * 3);
  CreateWorkspaceAddress(new_grad, new_indices, m_t);
  sparse_adam_->Launch(inputs_, workspace_, outputs_);
  for (size_t i = 0; i < 3 * 3; ++i) {
    EXPECT_TRUE(std::fabs(var_[i] - 0.999684) < 1e-6);
  }
  for (size_t i = 3 * 3; i < 2 * 3 * 3; ++i) {
    EXPECT_TRUE(std::fabs(var_[i] - 0.999715) < 1e-6);
  }
  for (size_t i = 2 * 3 * 3; i < 3 * 3 * 3; ++i) {
    EXPECT_TRUE(std::fabs(var_[i] - 0.999684) < 1e-6);
  }
}

TEST_F(SparseApplyAdamCpuKernelTest, sparse_test2) {
  for (size_t i = 0; i < 3 * 3 * 3; ++i) {
    var_.push_back(1.0);
    m_.push_back(1.0);
    v_.push_back(1.0);
    grad_.push_back(1.0);
  }
  sparse_adam_->indices_size_ = 3;
  sparse_adam_->var_first_dim_size_ = 3;
  sparse_adam_->var_outer_dim_size_ = 9;

  std::vector<int> indices{2, 2, 1};
  CreateInputAddress(indices);
  std::vector<float> new_grad(3 * 3 * 3);
  std::vector<int> new_indices(3);
  std::vector<float> m_t(3 * 3 * 3);
  CreateWorkspaceAddress(new_grad, new_indices, m_t);
  sparse_adam_->Launch(inputs_, workspace_, outputs_);
  for (size_t i = 0; i < 3 * 3; ++i) {
    EXPECT_TRUE(std::fabs(var_[i] - 0.999715) < 1e-6);
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

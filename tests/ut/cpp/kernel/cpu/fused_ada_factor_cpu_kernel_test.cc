/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "plugin/device/cpu/kernel/fused_ada_factor_cpu_kernel.h"
#undef private
#undef protected

namespace mindspore {
namespace kernel {
static constexpr size_t kSizeFloat32 = sizeof(float);
class FusedAdaFactorCpuKernelTest : public UT::Common {
 public:
  FusedAdaFactorCpuKernelTest() : ada_factor_(std::make_shared<FusedAdaFactorCpuKernelMod>()) {}

  void SetUp() override {
    ada_factor_->elem_num_ = elem_num_;
    ada_factor_->kernel_name_ = "AdaFactorTest";
    ada_factor_->last_row_dim_size_ = last_row_dim_size_;
    ada_factor_->last_col_dim_size_ = last_col_dim_size_;
  }

  void InitDataFp32() {
    exp_avg_.resize(elem_num_, 0.0f);
    exp_avg_sq_.resize(elem_num_, 0.0f);
    update_.resize(elem_num_, 0.0f);
    param_.resize(elem_num_, 1.0f);
    grad_.resize(elem_num_, 1.0f);

    auto r_factor_num = elem_num_ / last_row_dim_size_;
    exp_avg_sq_row_.resize(r_factor_num, 0.0f);
    r_factor_.resize(r_factor_num, 0.0f);

    auto c_factor_num = elem_num_ / last_col_dim_size_;
    exp_avg_sq_col_.resize(c_factor_num, 0.0f);
    c_factor_.resize(c_factor_num, 0.0f);
  }

  void InitDataFp16() {
    param_.resize(elem_num_);
    grad_.resize(elem_num_);
    exp_avg_.resize(elem_num_);
    exp_avg_sq_.resize(elem_num_);
    update_.resize(elem_num_, 0.0f);
    for (size_t i = 0; i < elem_num_; ++i) {
      auto ptr = (float16 *)param_.data();
      ptr[i] = static_cast<float16>(1.0f);
      ptr = (float16 *)grad_.data();
      ptr[i] = static_cast<float16>(1.0f);
      ptr = (float16 *)exp_avg_.data();
      ptr[i] = static_cast<float16>(0.0f);
      ptr = (float16 *)exp_avg_sq_.data();
      ptr[i] = static_cast<float16>(0.0f);
    }

    auto r_factor_num = elem_num_ / last_row_dim_size_;
    exp_avg_sq_row_.resize(r_factor_num, 0.0f);
    r_factor_.resize(r_factor_num, 0.0f);
    for (size_t i = 0; i < r_factor_num; ++i) {
      auto ptr = (float16 *)exp_avg_sq_row_.data();
      ptr[i] = static_cast<float16>(0.0f);
    }

    auto c_factor_num = elem_num_ / last_col_dim_size_;
    exp_avg_sq_col_.resize(c_factor_num, 0.0f);
    c_factor_.resize(c_factor_num, 0.0f);
    for (size_t i = 0; i < c_factor_num; ++i) {
      auto ptr = (float16 *)exp_avg_sq_col_.data();
      ptr[i] = static_cast<float16>(0.0f);
    }
  }

  AddressPtr CreateKernelAddress(void *addr, size_t elem_num, size_t type_size) {
    auto kernel_addr = std::make_shared<Address>();
    kernel_addr->addr = addr;
    kernel_addr->size = elem_num * type_size;
    return kernel_addr;
  }

  void CreateAddress(bool enable_global_norm) {
    constexpr size_t eps_num = 2;
    inputs_.push_back(CreateKernelAddress(epsilon_.data(), eps_num, kSizeFloat32));
    inputs_.push_back(CreateKernelAddress(&clip_threshold_, 1, kSizeFloat32));
    inputs_.push_back(CreateKernelAddress(&beta1_, 1, kSizeFloat32));
    inputs_.push_back(CreateKernelAddress(&beta2t_, 1, kSizeFloat32));
    inputs_.push_back(CreateKernelAddress(&weight_decay_, 1, kSizeFloat32));
    inputs_.push_back(CreateKernelAddress(&lr_, 1, kSizeFloat32));
    inputs_.push_back(CreateKernelAddress(grad_.data(), elem_num_, type_size_));
    inputs_.push_back(CreateKernelAddress(param_.data(), elem_num_, type_size_));
    inputs_.push_back(CreateKernelAddress(exp_avg_.data(), elem_num_, type_size_));
    inputs_.push_back(CreateKernelAddress(exp_avg_sq_row_.data(), elem_num_ / last_row_dim_size_, type_size_));
    inputs_.push_back(CreateKernelAddress(exp_avg_sq_col_.data(), elem_num_ / last_col_dim_size_, type_size_));
    inputs_.push_back(CreateKernelAddress(exp_avg_sq_.data(), elem_num_, type_size_));
    workspace_.push_back(CreateKernelAddress(update_.data(), elem_num_, kSizeFloat32));
    workspace_.push_back(CreateKernelAddress(r_factor_.data(), elem_num_ / last_row_dim_size_, kSizeFloat32));
    workspace_.push_back(CreateKernelAddress(c_factor_.data(), elem_num_ / last_col_dim_size_, kSizeFloat32));
    if (enable_global_norm) {
      inputs_.push_back(CreateKernelAddress(&global_norm_, 1, kSizeFloat32));
    }
  }

  void ComputeFp32(bool enable_global_norm) {
    ada_factor_->param_dtype_ = kNumberTypeFloat32;
    type_size_ = sizeof(float);
    InitDataFp32();

    CreateAddress(enable_global_norm);
    ada_factor_->Launch(inputs_, workspace_, outputs_);

    for (size_t i = 0; i < elem_num_; ++i) {
      EXPECT_TRUE(std::fabs(param_[i] - result_) < 1e-6);
    }
  }

  void ComputeFp16(bool enable_global_norm) {
    ada_factor_->param_dtype_ = kNumberTypeFloat16;
    type_size_ = sizeof(float16);
    InitDataFp16();

    CreateAddress(enable_global_norm);
    ada_factor_->Launch(inputs_, workspace_, outputs_);
    auto ptr = (float16 *)param_.data();
    for (size_t i = 0; i < elem_num_; ++i) {
      EXPECT_TRUE(std::fabs(static_cast<float>(ptr[i]) - result_) < 1e-3);
    }
  }

  std::vector<float> epsilon_{1e-30, 1e-3};
  float clip_threshold_ = 1.0;
  float lr_ = 0.03;
  float beta1_ = 0.9;
  float beta2t_ = 0.8;
  float weight_decay_ = 1e-2;
  float global_norm_ = 10.0f;
  float result_ = 0.97;
  std::vector<float> param_;
  std::vector<float> grad_;
  std::vector<float> exp_avg_;
  std::vector<float> exp_avg_sq_row_;
  std::vector<float> exp_avg_sq_col_;
  std::vector<float> exp_avg_sq_;

  std::vector<float> update_;
  std::vector<float> r_factor_;
  std::vector<float> c_factor_;

  std::vector<AddressPtr> inputs_;
  std::vector<AddressPtr> workspace_;
  std::vector<AddressPtr> outputs_;
  std::shared_ptr<FusedAdaFactorCpuKernelMod> ada_factor_;

  size_t last_row_dim_size_ = 4;
  size_t last_col_dim_size_ = 6;
  size_t elem_num_ = 2 * 6 * 4;
  size_t type_size_ = 4;
};

/// Feature: FusedAdaFactor
/// Description: Run FusedAdaFactor that needs factor state with fp32 data inputs
/// Expectation: pass
TEST_F(FusedAdaFactorCpuKernelTest, compute_fp32_factor) {
  ada_factor_->need_factor_ = true;
  ComputeFp32(false);
}

/// Feature: FusedAdaFactor
/// Description: Run FusedAdaFactor that doesn't need factor state with fp32 data inputs
/// Expectation: pass
TEST_F(FusedAdaFactorCpuKernelTest, compute_fp32_no_factor) {
  ada_factor_->need_factor_ = false;
  ComputeFp32(false);
}

/// Feature: FusedAdaFactor
/// Description: Run FusedAdaFactor that needs factor state with fp32 data inputs and global norm
/// Expectation: pass
TEST_F(FusedAdaFactorCpuKernelTest, compute_fp32_factor_global_norm) {
  ada_factor_->need_factor_ = true;
  ComputeFp32(true);
}

/// Feature: FusedAdaFactor
/// Description: Run FusedAdaFactor that doesn't need factor state with fp32 data inputs and global norm
/// Expectation: pass
TEST_F(FusedAdaFactorCpuKernelTest, compute_fp32_no_factor_global_norm) {
  ada_factor_->need_factor_ = false;
  ComputeFp32(true);
}

/// Feature: FusedAdaFactor
/// Description: Run FusedAdaFactor that needs factor state with fp16 data inputs
/// Expectation: pass
TEST_F(FusedAdaFactorCpuKernelTest, compute_fp16_factor) {
  ada_factor_->need_factor_ = true;
  ComputeFp16(false);
}

/// Feature: FusedAdaFactor
/// Description: Run FusedAdaFactor that doesn't need factor state with fp16 data inputs
/// Expectation: pass
TEST_F(FusedAdaFactorCpuKernelTest, compute_fp16_no_factor) {
  ada_factor_->need_factor_ = false;
  ComputeFp16(false);
}

/// Feature: FusedAdaFactor
/// Description: Run FusedAdaFactor that needs factor state with fp16 data inputs and global norm
/// Expectation: pass
TEST_F(FusedAdaFactorCpuKernelTest, compute_fp16_factor_global_norm) {
  ada_factor_->need_factor_ = true;
  ComputeFp16(true);
}

/// Feature: FusedAdaFactor
/// Description: Run FusedAdaFactor that doesn't need factor state with fp16 data inputs and global norm
/// Expectation: pass
TEST_F(FusedAdaFactorCpuKernelTest, compute_fp16_no_factor_global_norm) {
  ada_factor_->need_factor_ = false;
  ComputeFp16(true);
}
}  // namespace kernel
}  // namespace mindspore

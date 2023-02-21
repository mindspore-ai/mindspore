/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "plugin/device/cpu/kernel/adam_delta_cpu_kernel.h"
#undef private
#undef protected

namespace mindspore {
namespace kernel {
class AdamDeltaCpuKernelTest : public UT::Common {
 public:
  AdamDeltaCpuKernelTest() : adam_delta_(std::make_shared<AdamDeltaCpuKernelMod>()) {}

  void SetUp() override {
    delta_.clear();
    m_.clear();
    v_.clear();
    grad_.clear();
    inputs_.clear();
    workspace_.clear();
    outputs_.clear();
  }

  AddressPtr CreateKernelAddress(void *addr, size_t elem_num) {
    auto kernel_addr = std::make_shared<Address>();
    kernel_addr->addr = addr;
    kernel_addr->size = elem_num * sizeof(float);
    return kernel_addr;
  }

  void CreateAddress() {
    inputs_.push_back(CreateKernelAddress(m_.data(), elem_num_));
    inputs_.push_back(CreateKernelAddress(v_.data(), elem_num_));
    inputs_.push_back(CreateKernelAddress(&beta1_power_, 1));
    inputs_.push_back(CreateKernelAddress(&beta2_power_, 1));
    inputs_.push_back(CreateKernelAddress(&lr_, 1));
    inputs_.push_back(CreateKernelAddress(&beta1_, 1));
    inputs_.push_back(CreateKernelAddress(&beta2_, 1));
    inputs_.push_back(CreateKernelAddress(&epsilon_, 1));
    inputs_.push_back(CreateKernelAddress(grad_.data(), elem_num_));
    outputs_.push_back(CreateKernelAddress(delta_.data(), elem_num_));
  }

  std::vector<float> delta_;
  std::vector<float> m_;
  std::vector<float> v_;
  std::vector<float> grad_;
  std::vector<AddressPtr> inputs_;
  std::vector<AddressPtr> workspace_;
  std::vector<AddressPtr> outputs_;
  std::shared_ptr<AdamDeltaCpuKernelMod> adam_delta_;
  float beta1_power_ = 0.9;
  float beta2_power_ = 0.999;
  float lr_ = 0.001;
  float beta1_ = 0.9;
  float beta2_ = 0.999;
  float epsilon_ = 1e-8;
  size_t elem_num_ = 27;
};

/// Feature: Develop AdamDelta op on CPU.
/// Description: Test AdamDeltaCpuKernel.
/// Expectation: The AdamDeltaCpuKernel is successfully executed and a correct result is returned.
TEST_F(AdamDeltaCpuKernelTest, compute_test) {
  for (size_t i = 0; i < elem_num_; ++i) {
    delta_.push_back(1.0);
    m_.push_back(1.0);
    v_.push_back(1.0);
    grad_.push_back(1.0);
  }
  adam_delta_->elem_num_ = elem_num_;
  CreateAddress();
  adam_delta_->Launch(inputs_, workspace_, outputs_);
  for (size_t i = 0; i < elem_num_; ++i) {
    EXPECT_TRUE(std::fabs(delta_[i] + 0.000316) < 1e-6);
  }
}
}  // namespace kernel
}  // namespace mindspore

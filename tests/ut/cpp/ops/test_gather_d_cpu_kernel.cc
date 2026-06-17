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

#include <unistd.h>
#include <cstring>
#include <memory>
#include <vector>
#include "common/common_test.h"
#define private public
#define protected public
#include "../../../../mindspore/ops/kernel/cpu/native/gather_d_cpu_kernel.cc"
#undef private
#undef protected

namespace mindspore {
namespace kernel {
class GatherDCpuKernelTest : public UT::Common {
 public:
  GatherDCpuKernelTest() : gather_d_(std::make_shared<gather_d_cpu::GatherDCpuKernelMod>()) {}

  void SetUp() override {
    x_.clear();
    index_.clear();
    output_.clear();
    inputs_.clear();
    workspace_.clear();
    outputs_.clear();
    kernel_tensor_inputs_.clear();
    kernel_tensor_outputs_.clear();
    dim_ = 0;
  }

  KernelTensor *CreateKernelAddress(void *addr, size_t bytes) {
    auto kernel_addr = new KernelTensor();
    kernel_addr->set_device_ptr(addr);
    kernel_addr->set_size(bytes);
    return kernel_addr;
  }

  KernelTensor *CreateKernelTensor(const std::vector<int64_t> &shape, const TypePtr &dtype) {
    auto shape_ab = std::make_shared<abstract::Shape>(shape);
    auto new_abstract = std::make_shared<abstract::AbstractTensor>(dtype, shape_ab);
    auto *tensor = new KernelTensor(new_abstract->GetShape(), new_abstract->GetType(), kValueAny);
    tensor->set_format(mindspore::Format::DEFAULT_FORMAT);
    return tensor;
  }

  void CreateKernelTensors() {
    kernel_tensor_inputs_.push_back(CreateKernelTensor({2, 3}, kFloat32));
    kernel_tensor_inputs_.push_back(CreateKernelTensor({}, kInt64));
    kernel_tensor_inputs_.push_back(CreateKernelTensor({2, 3}, kInt32));
    kernel_tensor_outputs_.push_back(CreateKernelTensor({2, 3}, kFloat32));
  }

  void CreateAddresses() {
    inputs_.push_back(CreateKernelAddress(x_.data(), x_.size() * sizeof(float)));
    inputs_.push_back(CreateKernelAddress(&dim_, sizeof(dim_)));
    inputs_.push_back(CreateKernelAddress(index_.data(), index_.size() * sizeof(int32_t)));
    outputs_.push_back(CreateKernelAddress(output_.data(), output_.size() * sizeof(float)));
  }

  std::vector<float> x_;
  std::vector<int32_t> index_;
  std::vector<float> output_;
  int64_t dim_{0};
  std::vector<KernelTensor *> inputs_;
  std::vector<KernelTensor *> workspace_;
  std::vector<KernelTensor *> outputs_;
  std::vector<KernelTensor *> kernel_tensor_inputs_;
  std::vector<KernelTensor *> kernel_tensor_outputs_;
  std::shared_ptr<gather_d_cpu::GatherDCpuKernelMod> gather_d_;
};

/// Feature: GatherD CPU kernel invalid dim validation.
/// Description: Launch GatherD on CPU with dim beyond the valid rank range.
/// Expectation: Throw a clear exception instead of crashing the process.
TEST_F(GatherDCpuKernelTest, invalid_dim_should_throw_instead_of_crash) {
  x_ = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f};
  index_ = {0, 1, 0, 1, 0, 1};
  output_.resize(index_.size(), 0.f);
  dim_ = 5;
  CreateKernelTensors();
  ASSERT_TRUE(gather_d_->Init(kernel_tensor_inputs_, kernel_tensor_outputs_));
  ASSERT_EQ(gather_d_->Resize(kernel_tensor_inputs_, kernel_tensor_outputs_), KRET_OK);

  EXPECT_EXIT(
    {
      CreateAddresses();
      try {
        (void)gather_d_->Launch(inputs_, workspace_, outputs_, nullptr);
      } catch (const std::exception &e) {
        if (std::strstr(e.what(), "value of 'dim'") != nullptr) {
          _exit(0);
        }
        _exit(2);
      }
      _exit(1);
    },
    ::testing::ExitedWithCode(0), "");
}
}  // namespace kernel
}  // namespace mindspore

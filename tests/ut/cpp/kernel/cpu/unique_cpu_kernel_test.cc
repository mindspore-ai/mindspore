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
#include "backend/kernel_compiler/cpu/unique_cpu_kernel.h"
#undef private
#undef protected

namespace mindspore {
namespace kernel {
class UniqueCpuKernelTest : public UT::Common {
 public:
  UniqueCpuKernelTest() : unique_(std::make_shared<UniqueCPUKernel>()) {}

  void SetUp() override {
    unique_->input_size_ = 9;
    unique_->dtype_ = kNumberTypeFloat32;
    inputs_.clear();
    workspace_.clear();
    outputs_.clear();
  }

  AddressPtr CreateKernelAddress(void *addr) {
    auto kernel_addr = std::make_shared<Address>();
    kernel_addr->addr = addr;
    return kernel_addr;
  }

  void CreateAddress() {
    inputs_.push_back(CreateKernelAddress(x_.data()));
    outputs_.push_back(CreateKernelAddress(y_.data()));
    outputs_.push_back(CreateKernelAddress(idx_.data()));
    workspace_.push_back(CreateKernelAddress(workspace_idx_.data()));
    workspace_.push_back(CreateKernelAddress(workspace_idx_.data()));
    workspace_.push_back(CreateKernelAddress(workspace_idx_.data()));
  }

  std::vector<float> x_;
  std::vector<float> y_;
  std::vector<int> idx_;
  std::vector<int64_t> workspace_idx_;
  std::vector<AddressPtr> inputs_;
  std::vector<AddressPtr> workspace_;
  std::vector<AddressPtr> outputs_;
  std::shared_ptr<UniqueCPUKernel> unique_;
};

TEST_F(UniqueCpuKernelTest, compute_test) {
  x_ = {1, 1, 2, 4, 4, 4, 7, 8, 8};
  y_ = {1, 1, 1, 1, 1};
  idx_ = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  workspace_idx_ = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  CreateAddress();
  unique_->Launch(inputs_, workspace_, outputs_);

  // check compute result
  std::vector<float> expect_y{1, 2, 4, 7, 8};
  std::vector<int> expect_idx{0, 0, 1, 2, 2, 2, 3, 4, 4};
  EXPECT_TRUE(y_ == expect_y);
  EXPECT_TRUE(idx_ == expect_idx);
}
}  // namespace kernel
}  // namespace mindspore

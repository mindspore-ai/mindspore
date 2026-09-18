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

#include "gtest/gtest.h"
#include "ir/tensor.h"
#include "ir/tensor_new.h"

namespace mindspore {
namespace tensor {
/// Feature: CPU host buffer allocation in from_spec_fast.
/// Description: The requested host buffer is far larger than the machine can back, so std::malloc fails.
/// Expectation: The factory reports the allocation failure instead of returning a Tensor whose data_c() is nullptr.
TEST(TensorNewTest, FromSpecFastReportsCpuAllocationFailure) {
  // 2^58 float32 elements need 1 EiB, which is beyond the address space of any host, so std::malloc returns nullptr
  // even when the kernel overcommits memory.
  constexpr int64_t kHugeDim = 1LL << 58;
  ASSERT_THROW(from_spec_fast(kNumberTypeFloat32, {kHugeDim}, device::DeviceType::kCPU), std::exception);
}
}  // namespace tensor
}  // namespace mindspore

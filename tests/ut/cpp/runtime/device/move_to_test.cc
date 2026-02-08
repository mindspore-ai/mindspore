/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include <string>
#include <memory>
#include <algorithm>
#include "common/common_test.h"
#include "common/py_func_graph_fetcher.h"
#include "include/utils/utils.h"
#include "include/backend/ms_backend/runtime/graph_scheduler/base/move_to.h"
#include "device_address/device_type.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "include/securec.h"
#include "ir/tensor.h"
#include "ir/tensor_new.h"

namespace mindspore {
namespace device {
using device::DeviceAddressPtr;
class MoveToTest : public UT::Common {
 public:
  MoveToTest() = default;
  void SetUp() override {}
  void TearDown() override {}
};

/// Feature: Tensor.move_to
/// Description: Test tensor move to cpu
/// Expectation: src tensor and dst tensor has same value
TEST_F(MoveToTest, TestMoveToCaseToCPUNoBlocking) {
  std::vector<int64_t> input = {2, 2, 2, 2};
  TypePtr type = kInt64;
  auto src_tensor = tensor::from_vector(input, type);
  auto dst_tensor = tensor::from_spec(src_tensor->data_type(), src_tensor->shape(), device::DeviceType::kCPU);
  dst_tensor->set_device_address(nullptr);
  const std::string to = "CPU";
  bool return_self = false;
  device::MoveTo(src_tensor, dst_tensor, to, false, &return_self);
  std::vector<int64_t> tmp;
  for (size_t i = 0; i < input.size(); i++) {
    auto dst_value = (reinterpret_cast<int64_t *>(dst_tensor->data_c()))[i];
    tmp.push_back(dst_value);
  }
  EXPECT_EQ(tmp, input);
}

/// Feature: Tensor.move_to
/// Description: Test tensor move to cpu
/// Expectation: src tensor and dst tensor has same value
TEST_F(MoveToTest, TestMoveToCaseToCPUBlocking) {
  std::vector<int64_t> input = {2, 2, 2, 2};
  TypePtr type = kInt64;
  auto src_tensor = tensor::from_vector(input, type);
  auto dst_tensor = tensor::from_spec(src_tensor->data_type(), src_tensor->shape(), device::DeviceType::kCPU);
  dst_tensor->set_device_address(nullptr);
  const std::string to = "CPU";
  bool return_self = false;
  device::MoveTo(src_tensor, dst_tensor, to, true, &return_self);
  std::vector<int64_t> tmp;
  for (size_t i = 0; i < input.size(); i++) {
    auto dst_value = (reinterpret_cast<int64_t *>(dst_tensor->data_c()))[i];
    tmp.push_back(dst_value);
  }
  EXPECT_EQ(tmp, input);
}

/// Feature: Tensor.move_to
/// Description: Test tensor move to cpu
/// Expectation: dst tensor value is 0
TEST_F(MoveToTest, TestNoNeedMove) {
  std::vector<int64_t> input = {2, 2, 2, 2};
  TypePtr type = kInt64;
  auto src_tensor = tensor::from_vector(input, type);
  auto ptr = std::make_shared<DeviceAddress>(input.data(), input.size() * sizeof(int64_t), device::DeviceType::kCPU);
  src_tensor->set_device_address(ptr);
  auto dst_tensor = tensor::from_spec(src_tensor->data_type(), src_tensor->shape(), device::DeviceType::kCPU);
  dst_tensor->set_device_address(nullptr);
  const std::string to = "CPU";
  bool return_self = false;
  device::MoveTo(src_tensor, dst_tensor, to, false, &return_self);
  std::vector<int64_t> tmp;
  std::vector<int64_t> all_zero = {0, 0, 0, 0};
  for (size_t i = 0; i < input.size(); i++) {
    auto dst_value = (reinterpret_cast<int64_t *>(dst_tensor->data_c()))[i];
    tmp.push_back(dst_value);
  }
  EXPECT_EQ(tmp, all_zero);
}
}  // namespace device
}  // namespace mindspore

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
#include "include/common/utils/utils.h"
#include "runtime/device/move_to.h"
#include "include/backend/device_type.h"
#include "runtime/hardware/device_context_manager.h"
#include "securec/include/securec.h"
#include "ir/tensor.h"
#include "pybind_api/ir/tensor_py.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "plugin/device/ascend/hal/device/ascend_device_address.h"


namespace mindspore {
namespace device {
using mindspore::tensor::TensorPy;
using device::DeviceAddressPtr;
using device::cpu::CPUDeviceAddress;
using device::ascend::AscendDeviceAddress;

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
  auto src_tensor = std::make_shared<tensor::Tensor>(input, type);
  auto dst_tensor = std::make_shared<tensor::Tensor>(src_tensor->data_type(), src_tensor->shape());
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
  auto src_tensor = std::make_shared<tensor::Tensor>(input, type);
  auto dst_tensor = std::make_shared<tensor::Tensor>(src_tensor->data_type(), src_tensor->shape());
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
  auto src_tensor = std::make_shared<tensor::Tensor>(input, type);
  auto ptr = std::make_shared<CPUDeviceAddress>(input.data(), input.size() * sizeof(int64_t ));
  src_tensor->set_device_address(ptr);
  auto dst_tensor = std::make_shared<tensor::Tensor>(src_tensor->data_type(), src_tensor->shape());
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
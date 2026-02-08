/**
* Copyright 2025 Huawei Technologies Co., Ltd
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
#include "mockcpp/mockcpp.hpp"
#include "include/pynative/utils/pyboost/functions/dispatch.h"
#include "ir/tensor_new.h"

class DispatchTest : public testing::Test {
 protected:
  virtual void SetUp() {}

  virtual void TearDown() { GlobalMockObject::verify(); }

  static void SetUpTestCase() {}

  static void TearDownTestCase() {}
};


namespace mindspore {
mindspore::tensor::TensorPtr MakeTensor(device::DeviceType device_type) {
  auto t = tensor::from_spec(TypeId::kNumberTypeInt32, {1}, device::DeviceType::kNone);
  auto address = std::make_shared<DeviceAddress>();
  address->SetDeviceType(device_type);
  t->set_device_address(address);
  return t;
}

mindspore::ValuePtr MakeTensorValue(device::DeviceType device_type) {
  return MakeTensor(device_type);
}

mindspore::ValueTuplePtr MakeValueTuple(const std::vector<mindspore::ValuePtr> &values) {
  return std::make_shared<mindspore::ValueTuple>(values);
}

/// Feature: Device dispatch helper.
/// Description: Test c++ template compile-time boolean.
/// Expectation: assert true.
TEST_F(DispatchTest, TestTraits) {
  static_assert(is_tensor_ptr<mindspore::tensor::TensorPtr>::value, "TensorPtr should be tensor");
  static_assert(!is_tensor_ptr<int>::value, "int should not be tensor");

  static_assert(is_value_ptr<mindspore::ValuePtr>::value, "ValuePtr should be value");
  static_assert(!is_value_ptr<mindspore::tensor::TensorPtr>::value, "TensorPtr should not be value");

  static_assert(is_value_tuple<mindspore::ValueTuplePtr>::value, "ValueTuplePtr should be value tuple");
  static_assert(!is_value_tuple<mindspore::ValuePtr>::value, "ValuePtr should not be value tuple");

  static_assert(is_vector_value<std::vector<mindspore::ValuePtr>>::value, "vector<ValuePtr> should be vector value");
  static_assert(!is_vector_value<std::vector<int>>::value, "vector<int> should not be vector value");
}

/// Feature: Device dispatch helper.
/// Description: Get device from Tensor input.
/// Expectation: Device type equals tensor device.
TEST_F(DispatchTest, TestGetDeviceSingleFromTensor) {
  auto tensor = MakeTensor(device::DeviceType::kCPU);
  auto device_type = get_device_single(tensor);
  ASSERT_EQ(device_type, device::DeviceType::kCPU);
}

/// Feature: Device dispatch helper.
/// Description: Get device from Value containing Tensor.
/// Expectation: Device type equals inner tensor device.
TEST_F(DispatchTest, TestGetDeviceSingleFromValueTensor) {
  auto value = MakeTensorValue(device::DeviceType::kCPU);
  auto device_type = get_device_single(value);
  ASSERT_EQ(device_type, device::DeviceType::kCPU);
}

/// Feature: Device dispatch helper.
/// Description: Get device from ValueTuple with only CPU tensors.
/// Expectation: Device type is CPU.
TEST_F(DispatchTest, TestGetDeviceSingleFromValueTupleAllCpu) {
  auto v1 = MakeTensorValue(device::DeviceType::kCPU);
  auto v2 = MakeTensorValue(device::DeviceType::kCPU);
  auto tuple = MakeValueTuple({v1, v2});

  auto device_type = get_device_single(tuple);
  ASSERT_EQ(device_type, device::DeviceType::kCPU);
}

/// Feature: Device dispatch helper.
/// Description: Get device from ValueTuple with mixed devices.
/// Expectation: Device type is the highest priority (largest enum value).
TEST_F(DispatchTest, TestGetDeviceSingleFromValueTupleMixed) {
  auto v1 = MakeTensorValue(device::DeviceType::kCPU);
  auto v2 = MakeTensorValue(device::DeviceType::kAscend);
  auto tuple = MakeValueTuple({v1, v2});

  auto device_type = get_device_single(tuple);
  ASSERT_EQ(device_type, device::DeviceType::kAscend);
}

/// Feature: Device dispatch helper.
/// Description: Get device from vector<ValuePtr>.
/// Expectation: Device type is the highest priority in the vector.
TEST_F(DispatchTest, TestGetDeviceSingleFromVectorValue) {
  auto v1 = MakeTensorValue(device::DeviceType::kCPU);
  auto v2 = MakeTensorValue(device::DeviceType::kAscend);
  std::vector<mindspore::ValuePtr> values{v1, v2};

  auto device_type = get_device_single(values);
  ASSERT_EQ(device_type, device::DeviceType::kAscend);
}

/// Feature: Device dispatch helper.
/// Description: Get device from unsupported input type.
/// Expectation: Returns kUnknown.
TEST_F(DispatchTest, TestGetDeviceSingleUnknownType) {
  int x = 0;
  auto device_type = get_device_single(x);
  ASSERT_EQ(device_type, device::DeviceType::kUnknown);
}

/// Feature: Device dispatch helper.
/// Description: Get device from single Tensor argument.
/// Expectation: Device equals tensor device.
TEST_F(DispatchTest, TestGetDeviceSingleArg) {
  auto tensor = MakeTensor(device::DeviceType::kGPU);
  auto device_type = get_device(tensor);
  ASSERT_EQ(device_type, device::DeviceType::kGPU);
}

/// Feature: Device dispatch helper.
/// Description: Get device from multiple Tensor arguments with different devices.
/// Expectation: Device is the highest priority device.
TEST_F(DispatchTest, TestGetDeviceMultipleArgsMixedPriority) {
  auto cpu_tensor = MakeTensor(device::DeviceType::kCPU);
  auto ascend_tensor = MakeTensor(device::DeviceType::kAscend);
  auto gpu_tensor = MakeTensor(device::DeviceType::kGPU);

  auto device_type = get_device(cpu_tensor, ascend_tensor, gpu_tensor);
  ASSERT_EQ(device_type, device::DeviceType::kGPU);
}

/// Feature: Device dispatch helper.
/// Description: Ignore kUnknown and kNone when collecting device.
/// Expectation: Device is the highest valid device.
TEST_F(DispatchTest, TestGetDeviceIgnoreUnknownAndNone) {
  auto none_tensor = MakeTensor(device::DeviceType::kNone);
  auto unknown_tensor = MakeTensor(device::DeviceType::kUnknown);
  auto ascend_tensor = MakeTensor(device::DeviceType::kAscend);

  auto device_type = get_device(none_tensor, unknown_tensor, ascend_tensor);
  ASSERT_EQ(device_type, device::DeviceType::kAscend);
}

/// Feature: Device dispatch helper.
/// Description: Get device from mixed Tensor and Value inputs.
/// Expectation: Device is the highest priority among them.
TEST_F(DispatchTest, TestGetDeviceFromMixedValueAndTensor) {
  auto cpu_tensor = MakeTensor(device::DeviceType::kCPU);
  auto gpu_value = MakeTensorValue(device::DeviceType::kGPU);

  auto device_type = get_device(cpu_tensor, gpu_value);
  ASSERT_EQ(device_type, device::DeviceType::kGPU);
}
} // namespace mindspore

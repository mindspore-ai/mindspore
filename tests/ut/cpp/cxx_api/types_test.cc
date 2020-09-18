/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include <memory>
#include "common/common_test.h"
#include "include/api/types.h"

namespace mindspore {
class TestCxxApiTypes : public UT::Common {
 public:
  TestCxxApiTypes() = default;
};

TEST_F(TestCxxApiTypes, test_tensor_set_name_SUCCESS) {
  std::string tensor_name_before = "TEST1";
  std::string tensor_name_after = "TEST2";
  api::Tensor tensor1(tensor_name_before, api::DataType::kMsFloat32, {}, nullptr, 0);
  api::Tensor tensor2 = tensor1;
  api::Tensor tensor3 = tensor1.Clone();

  // name
  ASSERT_EQ(tensor1.Name(), tensor_name_before);
  ASSERT_EQ(tensor2.Name(), tensor_name_before);
  ASSERT_EQ(tensor3.Name(), tensor_name_before);

  tensor1.SetName(tensor_name_after);
  ASSERT_EQ(tensor1.Name(), tensor_name_after);
  ASSERT_EQ(tensor2.Name(), tensor_name_after);
  ASSERT_EQ(tensor3.Name(), tensor_name_before);
}

TEST_F(TestCxxApiTypes, test_tensor_set_dtype_SUCCESS) {
  api::Tensor tensor1("", api::DataType::kMsFloat32, {}, nullptr, 0);
  api::Tensor tensor2 = tensor1;
  api::Tensor tensor3 = tensor1.Clone();

  // dtype
  ASSERT_EQ(tensor1.DataType(), api::DataType::kMsFloat32);
  ASSERT_EQ(tensor2.DataType(), api::DataType::kMsFloat32);
  ASSERT_EQ(tensor3.DataType(), api::DataType::kMsFloat32);

  tensor1.SetDataType(api::DataType::kMsUint32);
  ASSERT_EQ(tensor1.DataType(), api::DataType::kMsUint32);
  ASSERT_EQ(tensor2.DataType(), api::DataType::kMsUint32);
  ASSERT_EQ(tensor3.DataType(), api::DataType::kMsFloat32);
}

TEST_F(TestCxxApiTypes, test_tensor_set_shape_SUCCESS) {
  std::vector<int64_t> shape = {3, 4, 5, 6};
  api::Tensor tensor1("", api::DataType::kMsFloat32, {}, nullptr, 0);
  api::Tensor tensor2 = tensor1;
  api::Tensor tensor3 = tensor1.Clone();

  // shape
  ASSERT_EQ(tensor1.Shape(), std::vector<int64_t>());
  ASSERT_EQ(tensor2.Shape(), std::vector<int64_t>());
  ASSERT_EQ(tensor3.Shape(), std::vector<int64_t>());

  tensor1.SetShape(shape);
  ASSERT_EQ(tensor1.Shape(), shape);
  ASSERT_EQ(tensor2.Shape(), shape);
  ASSERT_EQ(tensor3.Shape(), std::vector<int64_t>());
}


TEST_F(TestCxxApiTypes, test_tensor_util_SUCCESS) {
  std::vector<int64_t> shape = {3, 4, 5, 6};
  std::vector<uint32_t> data(3 * 4 * 5 * 6, 123);
  api::Tensor tensor1("", api::DataType::kMsFloat32, shape, data.data(), data.size() * sizeof(uint32_t));

  // data
  ASSERT_EQ(api::Tensor::GetTypeSize(api::DataType::kMsUint32), sizeof(uint32_t));
  ASSERT_EQ(tensor1.ElementNum(), 3 * 4 * 5 * 6);
}

TEST_F(TestCxxApiTypes, test_tensor_data_ref_and_copy_SUCCESS) {
  std::vector<int64_t> shape = {3, 4, 5, 6};
  std::vector<uint32_t> data(3 * 4 * 5 * 6, 123);
  api::Tensor tensor1("", api::DataType::kMsFloat32, shape, data.data(), data.size() * sizeof(uint32_t));
  api::Tensor tensor2 = tensor1;
  api::Tensor tensor3 = tensor1.Clone();

  // data
  ASSERT_EQ(tensor1.DataSize(), tensor2.DataSize());
  ASSERT_EQ(tensor1.DataSize(), tensor3.DataSize());
  ASSERT_EQ(tensor1.Data(), tensor2.MutableData());
  ASSERT_NE(tensor1.Data(), tensor3.Data());
}

TEST_F(TestCxxApiTypes, test_tensor_resize_data_SUCCESS) {
  std::vector<int64_t> shape = {3, 4, 5, 6};
  std::vector<uint32_t> data(3 * 4 * 5 * 6, 123);
  api::Tensor tensor1("", api::DataType::kMsFloat32, shape, data.data(), data.size() * sizeof(uint32_t));

  // data
  ASSERT_EQ(tensor1.ResizeData(0), true);
}

TEST_F(TestCxxApiTypes, test_tensor_set_data_wrong_data_size_FAILED) {
  std::vector<int64_t> shape = {3, 4, 5, 6};
  std::vector<uint32_t> data(3 * 4 * 5 * 6, 123);
  api::Tensor tensor1("", api::DataType::kMsFloat32, shape, data.data(), data.size() * sizeof(uint32_t));

  // data
  ASSERT_EQ(tensor1.SetData(nullptr, 1), false);
  ASSERT_EQ(tensor1.SetData(data.data(), 0), false);
}

TEST_F(TestCxxApiTypes, test_tensor_set_data_SUCCESS) {
  std::vector<int64_t> shape = {3, 4, 5, 6};
  std::vector<uint32_t> data(3 * 4 * 5 * 6, 123);
  api::Tensor tensor1("", api::DataType::kMsFloat32, shape, data.data(), data.size() * sizeof(uint32_t));

  // data
  ASSERT_EQ(tensor1.SetData(nullptr, 0), true);
  ASSERT_EQ(tensor1.SetData(data.data(), data.size() * sizeof(uint32_t)), true);
}

TEST_F(TestCxxApiTypes, test_buffer_data_ref_and_copy_SUCCESS) {
  std::vector<uint32_t> data(3 * 4 * 5 * 6, 123);
  api::Buffer buffer1(data.data(), data.size() * sizeof(uint32_t));
  api::Buffer buffer2 = buffer1;
  api::Buffer buffer3 = buffer1.Clone();

  // data
  ASSERT_EQ(buffer1.DataSize(), buffer2.DataSize());
  ASSERT_EQ(buffer1.DataSize(), buffer3.DataSize());
  ASSERT_EQ(buffer1.Data(), buffer2.MutableData());
  ASSERT_NE(buffer1.Data(), buffer3.Data());
}

TEST_F(TestCxxApiTypes, test_buffer_resize_data_SUCCESS) {
  std::vector<uint32_t> data(3 * 4 * 5 * 6, 123);
  api::Buffer buffer1(data.data(), data.size() * sizeof(uint32_t));

  // data
  ASSERT_EQ(buffer1.ResizeData(0), true);
}

TEST_F(TestCxxApiTypes, test_buffer_set_data_wrong_data_size_FAILED) {
  std::vector<uint32_t> data(3 * 4 * 5 * 6, 123);
  api::Buffer buffer1(data.data(), data.size() * sizeof(uint32_t));

  // data
  ASSERT_EQ(buffer1.SetData(nullptr, 1), false);
  ASSERT_EQ(buffer1.SetData(data.data(), 0), false);
}

TEST_F(TestCxxApiTypes, test_buffer_set_data_SUCCESS) {
  std::vector<uint32_t> data(3 * 4 * 5 * 6, 123);
  api::Buffer buffer1(data.data(), data.size() * sizeof(uint32_t));

  // data
  ASSERT_EQ(buffer1.SetData(nullptr, 0), true);
  ASSERT_EQ(buffer1.SetData(data.data(), data.size() * sizeof(uint32_t)), true);
}
}  // namespace mindspore

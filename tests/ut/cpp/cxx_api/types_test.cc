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
#define private public
#include "include/api/types.h"
#undef private

namespace mindspore {
class TestCxxApiTypes : public UT::Common {
 public:
  TestCxxApiTypes() = default;
};

TEST_F(TestCxxApiTypes, test_tensor_default_attr_SUCCESS) {
  MSTensor tensor;
  ASSERT_EQ(tensor.Name(), "");
  ASSERT_EQ(tensor.DataType(), DataType::kTypeUnknown);
  ASSERT_EQ(tensor.Shape().size(), 0);
  ASSERT_EQ(tensor.MutableData(), nullptr);
  ASSERT_EQ(tensor.DataSize(), 0);
  ASSERT_EQ(tensor.IsDevice(), false);
}

TEST_F(TestCxxApiTypes, test_tensor_attr_SUCCESS) {
  std::string tensor_name = "Name1";
  auto data_type = DataType::kNumberTypeFloat16;
  MSTensor tensor(tensor_name, data_type, {}, nullptr, 0);
  ASSERT_EQ(tensor.Name(), tensor_name);
  ASSERT_EQ(tensor.DataType(), data_type);
  ASSERT_EQ(tensor.Shape().size(), 0);
  ASSERT_EQ(tensor.MutableData(), nullptr);
  ASSERT_EQ(tensor.DataSize(), 0);
  ASSERT_EQ(tensor.IsDevice(), false);
}

TEST_F(TestCxxApiTypes, test_tensor_create_FAILED) {
  MSTensor tensor(nullptr);
  ASSERT_EQ(tensor, nullptr);
}

TEST_F(TestCxxApiTypes, test_tensor_data_SUCCESS) {
  std::vector<int32_t> data = {1, 2, 3, 4};
  MSTensor tensor("", DataType::kNumberTypeInt32, {4}, data.data(), data.size() * sizeof(int32_t));
  auto value = tensor.Data();
  int32_t *p = (int32_t *)value.get();
  for (size_t i = 0; i < data.size(); ++i) {
    ASSERT_EQ(p[i], data[i]);
  }
}

TEST_F(TestCxxApiTypes, test_tensor_ref_SUCCESS) {
  std::vector<int32_t> data = {1, 2, 3, 4};
  MSTensor tensor("", DataType::kNumberTypeInt32, {4}, data.data(), data.size() * sizeof(int32_t));
  MSTensor tensor2 = tensor;
  auto value = tensor2.Data();
  int32_t *p = (int32_t *)value.get();
  for (size_t i = 0; i < data.size(); ++i) {
    ASSERT_EQ(p[i], data[i]);
  }
}

TEST_F(TestCxxApiTypes, test_tensor_clone_SUCCESS) {
  std::vector<int32_t> data = {1, 2, 3, 4};
  MSTensor tensor("", DataType::kNumberTypeInt32, {4}, data.data(), data.size() * sizeof(int32_t));
  MSTensor *tensor2 = tensor.Clone();
  auto value = tensor2->Data();
  int32_t *p = (int32_t *)value.get();
  for (size_t i = 0; i < data.size(); ++i) {
    ASSERT_EQ(p[i], data[i]);
  }
  MSTensor::DestroyTensorPtr(tensor2);
}

TEST_F(TestCxxApiTypes, test_tensor_ref_modified_SUCCESS) {
  std::vector<int32_t> data = {1, 2, 3, 4};
  std::vector<int32_t> data_modified = {2, 3, 4, 5};
  MSTensor tensor("", DataType::kNumberTypeInt32, {4}, data.data(), data.size() * sizeof(int32_t));
  MSTensor tensor2 = tensor;
  (void)memcpy(tensor.MutableData(), data_modified.data(), data_modified.size() * sizeof(int32_t));
  auto value = tensor2.Data();
  int32_t *p = (int32_t *)value.get();
  for (size_t i = 0; i < data.size(); ++i) {
    ASSERT_EQ(p[i], data_modified[i]);
  }
}

TEST_F(TestCxxApiTypes, test_tensor_clone_modified_SUCCESS) {
  std::vector<int32_t> data = {1, 2, 3, 4};
  std::vector<int32_t> data_modified = {2, 3, 4, 5};
  MSTensor tensor("", DataType::kNumberTypeInt32, {4}, data.data(), data.size() * sizeof(int32_t));
  MSTensor *tensor2 = tensor.Clone();
  ASSERT_TRUE(tensor2 != nullptr);
  (void)memcpy(tensor.MutableData(), data_modified.data(), data_modified.size() * sizeof(int32_t));
  auto value = tensor2->Data();
  int32_t *p = (int32_t *)value.get();
  for (size_t i = 0; i < data.size(); ++i) {
    ASSERT_EQ(p[i], data[i]);
  }
  MSTensor::DestroyTensorPtr(tensor2);
}

TEST_F(TestCxxApiTypes, test_tensor_ref_creator_function_SUCCESS) {
  std::vector<int32_t> data = {1, 2, 3, 4};
  MSTensor *tensor =
    MSTensor::CreateRefTensor("", DataType::kNumberTypeInt32, {4}, data.data(), data.size() * sizeof(int32_t));
  ASSERT_TRUE(tensor != nullptr);
  data = {3, 4, 5, 6};
  auto value = tensor->Data();
  int32_t *p = (int32_t *)value.get();
  for (size_t i = 0; i < data.size(); ++i) {
    ASSERT_EQ(p[i], data[i]);
  }
  MSTensor::DestroyTensorPtr(tensor);
}

TEST_F(TestCxxApiTypes, test_tensor_creator_function_SUCCESS) {
  std::vector<int32_t> data = {1, 2, 3, 4};
  MSTensor *tensor =
    MSTensor::CreateTensor("", DataType::kNumberTypeInt32, {4}, data.data(), data.size() * sizeof(int32_t));
  ASSERT_TRUE(tensor != nullptr);
  data = {3, 4, 5, 6};
  auto value = tensor->Data();
  int32_t *p = (int32_t *)value.get();
  for (size_t i = 0; i < data.size(); ++i) {
    ASSERT_NE(p[i], data[i]);
  }
  MSTensor::DestroyTensorPtr(tensor);
}

TEST_F(TestCxxApiTypes, test_tensor_string_tensor_SUCCESS) {
  std::string tensor_name = "tensor_name";
  std::vector<std::string> origin_strs;
  origin_strs.emplace_back("qwe");
  origin_strs.emplace_back("asd");
  origin_strs.emplace_back("");
  origin_strs.emplace_back("zxc");
  auto tensor = MSTensor::StringsToTensor(tensor_name, origin_strs);
  ASSERT_TRUE(tensor != nullptr);
  ASSERT_EQ(tensor->Name(), tensor_name);
  auto new_strs = MSTensor::TensorToStrings(*tensor);
  ASSERT_EQ(new_strs.size(), origin_strs.size());
  for (size_t i = 0; i < new_strs.size(); ++i) {
    ASSERT_EQ(new_strs[i], origin_strs[i]);
  }
}

TEST_F(TestCxxApiTypes, test_tensor_empty_string_tensor_SUCCESS) {
  std::string tensor_name = "tensor_name";
  std::vector<std::string> origin_strs;
  auto tensor = MSTensor::StringsToTensor(tensor_name, origin_strs);
  ASSERT_TRUE(tensor != nullptr);
  ASSERT_EQ(tensor->Name(), tensor_name);
  auto new_strs = MSTensor::TensorToStrings(*tensor);
  ASSERT_EQ(new_strs.size(), origin_strs.size());
}

TEST_F(TestCxxApiTypes, test_tensor_string_tensor_invalid_type_FAILED) {
  MSTensor tensor("", DataType::kNumberTypeInt32, {1}, nullptr, sizeof(int32_t));
  auto new_strs = MSTensor::TensorToStrings(tensor);
  ASSERT_TRUE(new_strs.empty());
}

TEST_F(TestCxxApiTypes, test_buffer_data_ref_and_copy_SUCCESS) {
  std::vector<uint32_t> data(3 * 4 * 5 * 6, 123);
  Buffer buffer1(data.data(), data.size() * sizeof(uint32_t));
  Buffer buffer2 = buffer1;
  Buffer buffer3 = buffer1.Clone();

  // data
  ASSERT_EQ(buffer1.DataSize(), buffer2.DataSize());
  ASSERT_EQ(buffer1.DataSize(), buffer3.DataSize());
  ASSERT_EQ(buffer1.Data(), buffer2.MutableData());
  ASSERT_NE(buffer1.Data(), buffer3.Data());
}

TEST_F(TestCxxApiTypes, test_buffer_resize_data_SUCCESS) {
  std::vector<uint32_t> data(3 * 4 * 5 * 6, 123);
  Buffer buffer1(data.data(), data.size() * sizeof(uint32_t));

  // data
  ASSERT_EQ(buffer1.ResizeData(0), true);
}

TEST_F(TestCxxApiTypes, test_buffer_set_data_wrong_data_size_FAILED) {
  std::vector<uint32_t> data(3 * 4 * 5 * 6, 123);
  Buffer buffer1(data.data(), data.size() * sizeof(uint32_t));

  // data
  ASSERT_EQ(buffer1.SetData(nullptr, 1), false);
  ASSERT_EQ(buffer1.SetData(data.data(), 0), false);
}

TEST_F(TestCxxApiTypes, test_buffer_set_data_SUCCESS) {
  std::vector<uint32_t> data(3 * 4 * 5 * 6, 123);
  Buffer buffer1(data.data(), data.size() * sizeof(uint32_t));

  // data
  ASSERT_EQ(buffer1.SetData(nullptr, 0), true);
  ASSERT_EQ(buffer1.SetData(data.data(), data.size() * sizeof(uint32_t)), true);
}
}  // namespace mindspore

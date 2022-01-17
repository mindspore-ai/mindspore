/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "include/c_api/tensor_c.h"
#include "common/common_test.h"

namespace mindspore {
class TensorCTest : public mindspore::CommonTest {
 public:
  TensorCTest() {}
};

TEST_F(TensorCTest, common_test) {
  constexpr size_t shape_num = 2;
  int64_t shape[shape_num] = {2, 3};
  MSTensorHandle tensor = MSTensorCreate("name001", kMSDataTypeNumberTypeInt32, shape, shape_num, nullptr, 0);
  ASSERT_TRUE(tensor != nullptr);
  ASSERT_STREQ(MSTensorGetName(tensor), "name001");
  ASSERT_EQ(MSTensorGetDataType(tensor), kMSDataTypeNumberTypeInt32);
  size_t ret_shape_num;
  const int64_t *ret_shape = MSTensorGetShape(tensor, &ret_shape_num);
  ASSERT_EQ(ret_shape_num, shape_num);
  for (size_t i = 0; i < ret_shape_num; i++) {
    ASSERT_EQ(ret_shape[i], shape[i]);
  }
  ASSERT_EQ(MSTensorGetElementNum(tensor), 6);
  ASSERT_EQ(MSTensorGetDataSize(tensor), 6 * sizeof(int32_t));
  ASSERT_EQ(MSTensorGetData(tensor), nullptr);
  ASSERT_TRUE(MSTensorGetMutableData(tensor) != nullptr);

  MSTensorSetName(tensor, "name002");
  ASSERT_STREQ(MSTensorGetName(tensor), "name002");

  MSTensorSetDataType(tensor, kMSDataTypeNumberTypeFloat32);
  ASSERT_EQ(MSTensorGetDataType(tensor), kMSDataTypeNumberTypeFloat32);
  constexpr size_t new_shape_num = 4;
  int64_t new_shape[new_shape_num] = {1, 2, 3, 1};
  MSTensorSetShape(tensor, new_shape, new_shape_num);
  size_t new_ret_shape_num;
  const int64_t *new_ret_shape = MSTensorGetShape(tensor, &new_ret_shape_num);
  ASSERT_EQ(new_ret_shape_num, new_shape_num);
  for (size_t i = 0; i < new_ret_shape_num; i++) {
    ASSERT_EQ(new_ret_shape[i], new_shape[i]);
  }

  MSTensorSetFormat(tensor, kMSFormatNCHW);
  ASSERT_EQ(MSTensorGetFormat(tensor), kMSFormatNCHW);

  constexpr size_t data_len = 6;
  ASSERT_EQ(MSTensorGetElementNum(tensor), data_len);
  ASSERT_EQ(MSTensorGetDataSize(tensor), data_len * sizeof(float));

  float data[data_len] = {1, 2, 3, 4, 5, 6};
  MSTensorSetData(tensor, data);
  const float *ret_data = static_cast<const float *>(MSTensorGetData(tensor));
  for (size_t i = 0; i < data_len; i++) {
    ASSERT_EQ(ret_data[i], data[i]);
  }

  MSTensorHandle clone = MSTensorClone(tensor);
  ASSERT_TRUE(clone != nullptr);
  ASSERT_STREQ(MSTensorGetName(clone), "");
  ASSERT_EQ(MSTensorGetDataType(clone), kMSDataTypeNumberTypeFloat32);
  size_t clone_shape_num;
  const int64_t *clone_shape = MSTensorGetShape(clone, &clone_shape_num);
  ASSERT_EQ(clone_shape_num, new_ret_shape_num);
  for (size_t i = 0; i < clone_shape_num; i++) {
    ASSERT_EQ(clone_shape[i], new_ret_shape[i]);
  }
  ASSERT_EQ(MSTensorGetElementNum(clone), MSTensorGetElementNum(tensor));
  ASSERT_EQ(MSTensorGetDataSize(clone), MSTensorGetDataSize(tensor));
  ASSERT_TRUE(MSTensorGetData(clone) != MSTensorGetData(tensor));

  MSTensorDestroy(&tensor);
  MSTensorDestroy(&clone);
}
}  // namespace mindspore

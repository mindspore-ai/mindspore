/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include <cmath>
#include <memory>
#include <sstream>
#include <unordered_map>
#include "common/common_test.h"
#include "c_api/include/tensor.h"
#include "c_api/include/context.h"
#include "c_api/base/status.h"
#include "c_api/base/handle_types.h"

namespace mindspore {
class TestCApiTensor : public UT::Common {
 public:
  TestCApiTensor() = default;
};

/// Feature: C_API
/// Description: test tensor create, set and get method.
/// Expectation: create/set/get works correctly.
TEST_F(TestCApiTensor, test_new_tensor) {
  STATUS ret;
  ResMgrHandle res_mgr = MSResourceManagerCreate();
  ASSERT_TRUE(res_mgr != nullptr);
  float data_value[] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  int64_t data_shape[] = {1, 1, 3, 3};
  TensorHandle tensor_false = MSNewTensor(res_mgr, NULL, MS_FLOAT32, data_shape, 4, 9 * sizeof(float));
  ASSERT_TRUE(tensor_false == nullptr);
  TensorHandle tensor = MSNewTensor(res_mgr, data_value, MS_FLOAT32, data_shape, 4, 9 * sizeof(float));
  ASSERT_TRUE(tensor != nullptr);
  size_t ele_num = MSTensorGetElementNum(res_mgr, tensor, &ret);
  ASSERT_EQ(ret, RET_OK);
  ASSERT_EQ(ele_num, 9);
  size_t data_size = MSTensorGetDataSize(res_mgr, tensor, &ret);
  ASSERT_EQ(ret, RET_OK);
  ASSERT_EQ(data_size, 9 * sizeof(float));
  void *result = MSTensorGetData(res_mgr, tensor);
  ASSERT_TRUE(result != nullptr);
  auto res = static_cast<float *>(result);
  ASSERT_EQ(res[0], 0);
  ASSERT_EQ(res[4], 4);
  ASSERT_EQ(res[8], 8);
  ret = MSTensorSetDataType(res_mgr, tensor, MS_INT32);
  ASSERT_EQ(ret, RET_OK);
  DataTypeC type = MSTensorGetDataType(res_mgr, tensor, &ret);
  ASSERT_EQ(ret, RET_OK);
  ASSERT_EQ(type, MS_INT32);
  int64_t new_shape[] = {2, 3, 4, 5};
  ret = MSTensorSetShape(res_mgr, tensor, new_shape, 4);
  ASSERT_EQ(ret, RET_OK);
  int64_t new_shape_ret[4];
  ret = MSTensorGetShape(res_mgr, tensor, new_shape_ret, 4);
  ASSERT_EQ(ret, RET_OK);
  ASSERT_EQ(new_shape_ret[0], 2);
  ASSERT_EQ(new_shape_ret[1], 3);
  ASSERT_EQ(new_shape_ret[2], 4);
  ASSERT_EQ(new_shape_ret[3], 5);
  MSResourceManagerDestroy(res_mgr);
}

/// Feature: C_API
/// Description: test tensor create and get method.
/// Expectation: create/and get work correctly.
TEST_F(TestCApiTensor, test_new_tensor_with_src_type) {
  STATUS ret;
  ResMgrHandle res_mgr = MSResourceManagerCreate();
  ASSERT_TRUE(res_mgr != nullptr);
  float data_value[] = {0, 1, 2, 3};
  int64_t data_shape[] = {1, 1, 2, 2};
  TensorHandle tensor_false = MSNewTensorWithSrcType(res_mgr, data_value, NULL, 4, MS_INT32, MS_FLOAT32);
  ASSERT_TRUE(tensor_false == nullptr);
  TensorHandle tensor = MSNewTensorWithSrcType(res_mgr, data_value, data_shape, 4, MS_INT32, MS_FLOAT32);
  ASSERT_TRUE(tensor != nullptr);
  DataTypeC type = MSTensorGetDataType(res_mgr, tensor, &ret);
  ASSERT_EQ(ret, RET_OK);
  ASSERT_EQ(type, MS_INT32);
  MSResourceManagerDestroy(res_mgr);
}
}

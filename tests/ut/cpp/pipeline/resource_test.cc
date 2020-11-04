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
#include <iostream>
#include <memory>

#include "common/common_test.h"
#include "utils/log_adapter.h"
#include "pipeline/jit/resource.h"
#include "ir/primitive.h"
#include "frontend/operator/ops.h"

namespace mindspore {
namespace pipeline {

using MethodMap = std::unordered_map<int64_t, std::unordered_map<std::string, Any>>;

extern MethodMap& GetMethodMap();

class TestResource : public UT::Common {
 public:
  TestResource() {}
  void SetUp() {}
  void TearDown() {}
};

TEST_F(TestResource, test_built_in_type_map) {
  ASSERT_TRUE(true == Resource::IsTypeInBuiltInMap(kNumberTypeInt));
  ASSERT_TRUE(true == Resource::IsTypeInBuiltInMap(kNumberTypeInt8));
  ASSERT_TRUE(true == Resource::IsTypeInBuiltInMap(kNumberTypeInt16));
  ASSERT_TRUE(true == Resource::IsTypeInBuiltInMap(kNumberTypeInt32));
  ASSERT_TRUE(true == Resource::IsTypeInBuiltInMap(kNumberTypeInt64));

  ASSERT_TRUE(true == Resource::IsTypeInBuiltInMap(kNumberTypeFloat));
  ASSERT_TRUE(true == Resource::IsTypeInBuiltInMap(kNumberTypeFloat16));
  ASSERT_TRUE(true == Resource::IsTypeInBuiltInMap(kNumberTypeFloat32));
  ASSERT_TRUE(true == Resource::IsTypeInBuiltInMap(kNumberTypeFloat64));

  ASSERT_TRUE(true == Resource::IsTypeInBuiltInMap(kNumberTypeBool));
  ASSERT_TRUE(true == Resource::IsTypeInBuiltInMap(kNumberTypeUInt));
  ASSERT_TRUE(true == Resource::IsTypeInBuiltInMap(kObjectTypeTuple));
  ASSERT_TRUE(true == Resource::IsTypeInBuiltInMap(kObjectTypeList));
  ASSERT_TRUE(true == Resource::IsTypeInBuiltInMap(kObjectTypeTensorType));

  MethodMap& map = GetMethodMap();
  for (auto& iter : map) {
    for (auto& iter_map : iter.second) {
      Any value = iter_map.second;
      ASSERT_TRUE(value.is<std::string>() || value.is<PrimitivePtr>());
    }
  }

  Any value = Resource::GetMethodPtr(kNumberTypeInt, "__add__");
  ASSERT_TRUE(value == Any(prim::kPrimScalarAdd));
  value = Resource::GetMethodPtr(kNumberTypeInt64, "__add__");
  ASSERT_TRUE(value == Any(prim::kPrimScalarAdd));
  value = Resource::GetMethodPtr(kNumberTypeFloat, "__add__");
  ASSERT_TRUE(value == Any(prim::kPrimScalarAdd));
  value = Resource::GetMethodPtr(kNumberTypeFloat64, "__add__");
  ASSERT_TRUE(value == Any(prim::kPrimScalarAdd));
}

}  // namespace pipeline
}  // namespace mindspore

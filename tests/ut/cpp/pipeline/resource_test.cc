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
#include "pipeline/resource.h"
#include "ir/primitive.h"
#include "operator/ops.h"

namespace mindspore {
namespace pipeline {

using MethodMap = std::unordered_map<int, std::unordered_map<std::string, Any>>;

extern MethodMap& GetMethodMap();

class TestResource : public UT::Common {
 public:
  TestResource() {}
  void SetUp() {}
  void TearDown() {}
};

TEST_F(TestResource, test_standard_method_map) {
  ASSERT_TRUE(true == Resource::IsTypeInMethodMap(kNumberTypeInt));
  ASSERT_TRUE(true == Resource::IsTypeInMethodMap(kNumberTypeInt8));
  ASSERT_TRUE(true == Resource::IsTypeInMethodMap(kNumberTypeInt16));
  ASSERT_TRUE(true == Resource::IsTypeInMethodMap(kNumberTypeInt32));
  ASSERT_TRUE(true == Resource::IsTypeInMethodMap(kNumberTypeInt64));

  ASSERT_TRUE(true == Resource::IsTypeInMethodMap(kNumberTypeFloat));
  ASSERT_TRUE(true == Resource::IsTypeInMethodMap(kNumberTypeFloat16));
  ASSERT_TRUE(true == Resource::IsTypeInMethodMap(kNumberTypeFloat32));
  ASSERT_TRUE(true == Resource::IsTypeInMethodMap(kNumberTypeFloat64));

  ASSERT_TRUE(true == Resource::IsTypeInMethodMap(kNumberTypeBool));
  ASSERT_TRUE(true == Resource::IsTypeInMethodMap(kNumberTypeUInt));
  ASSERT_TRUE(true == Resource::IsTypeInMethodMap(kObjectTypeTuple));
  ASSERT_TRUE(true == Resource::IsTypeInMethodMap(kObjectTypeList));
  ASSERT_TRUE(true == Resource::IsTypeInMethodMap(kObjectTypeTensorType));

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

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
#include "c_api/include/attribute.h"
#include "c_api/include/context.h"
#include "c_api/include/graph.h"
#include "c_api/include/node.h"
#include "c_api/base/status.h"
#include "c_api/base/handle_types.h"

namespace mindspore {
class TestCApiAttr : public UT::Common {
 public:
  TestCApiAttr() = default;
};

/// Feature: C_API_NEW_ATTR
/// Description: Making new attributes.
/// Expectation: New Attributes work correctly.
TEST_F(TestCApiAttr, test_attr) {
  STATUS ret;
  ResMgrHandle res_mgr = MSResourceManagerCreate();
  ASSERT_TRUE(res_mgr != nullptr);
  GraphHandle fg = MSFuncGraphCreate(res_mgr);
  ASSERT_TRUE(fg != nullptr);

  AttrHandle attr1 = MSNewAttrInt64(res_mgr, 1);
  ASSERT_TRUE(attr1 != nullptr);
  int64_t attr2_raw[] = {2, 2};
  AttrHandle attr2 = MSOpNewAttrs(res_mgr, attr2_raw, 2, kNumberTypeInt64);
  ASSERT_TRUE(attr2 != nullptr);
  char name1[] = "attr1";
  char name2[] = "attr2";
  char *attr_names[] = {name1, name2};
  AttrHandle attrs[] = {attr1, attr2};
  size_t attr_num = 2;

  NodeHandle x = MSNewPlaceholder(res_mgr, fg, kNumberTypeInt32, NULL, 0);
  ASSERT_TRUE(x != nullptr);
  NodeHandle y = MSNewScalarConstantInt32(res_mgr, 2);
  ASSERT_TRUE(y != nullptr);
  NodeHandle input_nodes[] = {x, y};
  size_t input_num = 2;
  NodeHandle op_add = MSNewOp(res_mgr, fg, "Add", input_nodes, input_num, attr_names, attrs, attr_num);
  ASSERT_TRUE(op_add != nullptr);
  int64_t attr1_retrived = MSOpGetScalarAttrInt64(res_mgr, op_add, "attr1", &ret);
  ASSERT_EQ(ret, RET_OK);
  ASSERT_EQ(attr1_retrived, 1);
  int64_t values[2];
  ret = MSOpGetAttrArrayInt64(res_mgr, op_add, "attr2", values, 2);
  ASSERT_EQ(ret, RET_OK);
  ASSERT_EQ(values[0], 2);
  ASSERT_EQ(values[1], 2);

  ret = MSOpSetScalarAttrInt64(res_mgr, op_add, "attr1", 2);
  ASSERT_EQ(ret, RET_OK);
  attr1_retrived = MSOpGetScalarAttrInt64(res_mgr, op_add, "attr1", &ret);
  ASSERT_EQ(ret, RET_OK);
  ASSERT_EQ(attr1_retrived, 2);
  values[0] = 1;
  values[1] = 1;
  ret = MSOpSetAttrArray(res_mgr, op_add, "attr2", values, 2, kNumberTypeInt64);
  ASSERT_EQ(ret, RET_OK);
  ret = MSOpGetAttrArrayInt64(res_mgr, op_add, "attr2", values, 2);
  ASSERT_EQ(ret, RET_OK);
  ASSERT_EQ(values[0], 1);
  ASSERT_EQ(values[1], 1);
  MSResourceManagerDestroy(res_mgr);
}
}  // namespace mindspore

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
#include <sstream>
#include <memory>
#include <algorithm>
#include <unordered_map>

#include "common/common_test.h"
#include "frontend/operator/ops.h"
#include "utils/any.h"
#include "utils/misc.h"

using std::cout;
using std::endl;
using std::string;

namespace mindspore {

class TestAny : public UT::Common {
 public:
  TestAny() {}
};

using Primitive = Primitive;
using PrimitivePtr = PrimitivePtr;

Any f(const Any &a) {
  Any value = a;
  return value;
}

TEST_F(TestAny, test_common) {
  Any value(std::make_shared<Primitive>("add"));
  if (value.type() == typeid(PrimitivePtr)) {
    PrimitivePtr a = value.cast<PrimitivePtr>();
  }
  if (value.is<PrimitivePtr>()) {
    PrimitivePtr a = value.cast<PrimitivePtr>();
  }

  Any value1 = f(std::make_shared<Primitive>("add"));

  Any a = 1;
  Any b = string("hello, world");
  Any c;

  ASSERT_FALSE(a.empty());
  ASSERT_FALSE(b.empty());
  ASSERT_TRUE(c.empty());

  ASSERT_TRUE(a.is<int>());
  ASSERT_FALSE(a.is<std::string>());
  ASSERT_EQ(1, a.cast<int>());
  c = a;

  ASSERT_TRUE(c.is<int>());
  ASSERT_FALSE(c.is<std::string>());
  ASSERT_EQ(1, c.cast<int>());
}

TEST_F(TestAny, test_unordered_map) {
  std::unordered_map<Any, int, AnyHash> ids;
  Any a = 1;
  ids[a] = 2;
  ASSERT_EQ(2, ids[a]);
}

TEST_F(TestAny, test_unordered_map1) {
  std::unordered_map<int, Any> ids;
  ids[1] = 1;
  ASSERT_EQ(1, ids[1].cast<int>());
}

}  // namespace mindspore

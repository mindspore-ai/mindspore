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
#include "common/common_test.h"
#include "pipeline/jit/static_analysis/static_analysis.h"
#include "utils/symbolic.h"

using std::cout;
using std::endl;
using std::string;

namespace mindspore {
class TestSymbolic : public UT::Common {
 public:
  TestSymbolic() {}
};

TEST_F(TestSymbolic, test_env) {
  auto sk1 = std::make_shared<SymbolicKeyInstance>(NewValueNode(static_cast<int64_t>(1)), abstract::FromValue(1234));
  auto sk1b = std::make_shared<SymbolicKeyInstance>(NewValueNode(static_cast<int64_t>(1)), abstract::FromValue(1234));

  ASSERT_EQ(*sk1, *sk1b);

  auto sk2 = std::make_shared<SymbolicKeyInstance>(NewValueNode(static_cast<int64_t>(2)), abstract::FromValue(1234));

  EnvInstance e = newenv->Set(sk1, 100);
  ASSERT_FALSE(e == *newenv);

  ASSERT_EQ(newenv->Len(), 0);
  ASSERT_EQ(e.Len(), 1);
  ASSERT_EQ(e.Get(sk1, 0), 100);
  ASSERT_EQ(e.Get(sk2, 0), 0);

  EnvInstance e2 = e.Set(sk1b, 200);
  ASSERT_EQ(e2.Len(), 1);
  ASSERT_EQ(e2.Get(sk1, 0), 200);
  ASSERT_EQ(e2.Get(sk2, 0), 0);

  EnvInstance e3 = e2.Set(sk2, 300);
  ASSERT_EQ(e3.Len(), 2);
  ASSERT_EQ(e3.Get(sk1, 0), 200);
  ASSERT_EQ(e3.Get(sk2, 0), 300);
}

}  // namespace mindspore

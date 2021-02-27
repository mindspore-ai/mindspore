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
#include "abstract/utils.h"

#include "common/common_test.h"
#include "pipeline/jit/static_analysis/static_analysis.h"

namespace mindspore {
namespace abstract {
class TestUtils : public UT::Common {
 public:
  TestUtils() {}
  virtual void SetUp() {}
  virtual void TearDown() {}
};

TEST_F(TestUtils, test_join) {
  // AbstractScalar
  AbstractBasePtr abs_s1 = FromValue(static_cast<int64_t>(1), false);
  AbstractBasePtr abs_s2 = FromValue(static_cast<int64_t>(2), false);
  AbstractBasePtr abs_s_anything = FromValue(static_cast<int64_t>(2), true);
  abs_s_anything->set_value(kAnyValue);

  AbstractBasePtr res_s1 = abs_s1->Join(abs_s2);
  ASSERT_EQ(*res_s1, *abs_s_anything);

  abs_s1 = FromValue(static_cast<int64_t>(1), false);

  AbstractBasePtr t1 = std::make_shared<AbstractTuple>(AbstractBasePtrList({abs_s1, abs_s_anything}));
  AbstractBasePtr t2 = std::make_shared<AbstractTuple>(AbstractBasePtrList({abs_s1, abs_s_anything}));
  AbstractBasePtr t3 = std::make_shared<AbstractTuple>(AbstractBasePtrList({abs_s_anything, abs_s_anything}));

  AbstractBasePtr res_t1 = t1->Join(t2);
  ASSERT_EQ(res_t1, t1);

  res_t1 = t1->Join(t3);
  ASSERT_EQ(*res_t1, *t3);

  res_t1 = t3->Join(t1);
  ASSERT_EQ(res_t1, t3);
}

}  // namespace abstract
}  // namespace mindspore

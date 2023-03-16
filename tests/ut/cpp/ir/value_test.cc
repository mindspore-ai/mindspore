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
#include "ir/value.h"
#include "abstract/abstract_value.h"
#include "utils/log_adapter.h"

namespace mindspore {
using AbstractScalar = abstract::AbstractScalar;
using AbstractTuple = abstract::AbstractTuple;
using AbstractBasePtrList = abstract::AbstractBasePtrList;

class TestValue : public UT::Common {
 public:
  TestValue() {}
};

TEST_F(TestValue, test_int64) {
  auto i64a = std::make_shared<Int64Imm>(2);
  ASSERT_TRUE(i64a != nullptr);
}

TEST_F(TestValue, testToAbstract) {
  ValuePtr boolv = std::make_shared<BoolImm>(true);
  AbstractBasePtr boola = std::make_shared<AbstractScalar>(true);
  AbstractBasePtr ret = boolv->ToAbstract();
  ASSERT_TRUE(ret);
  ASSERT_EQ(*(ret), *(boola));
  ASSERT_FALSE(*(ret) == *(std::make_shared<AbstractScalar>(false)));
  ASSERT_FALSE(*(ret) == *(std::make_shared<AbstractScalar>(static_cast<int64_t>(2))));

  ValuePtr i64v = std::make_shared<Int64Imm>(2);
  AbstractBasePtr i64a = std::make_shared<AbstractScalar>(static_cast<int64_t>(2));
  ret = i64v->ToAbstract();
  ASSERT_TRUE(ret);
  ASSERT_EQ(*(ret), *(i64a));

  ValuePtr f32v = std::make_shared<FP32Imm>(1.0);
  AbstractBasePtr f32a = std::make_shared<AbstractScalar>(1.0f);
  ret = f32v->ToAbstract();
  ASSERT_TRUE(ret);
  ASSERT_EQ(*(ret), *(f32a));

  ValuePtr sv = std::make_shared<StringImm>("_");
  AbstractBasePtr sa = std::make_shared<AbstractScalar>(std::string("_"));
  ret = sv->ToAbstract();
  ASSERT_TRUE(ret);
  ASSERT_EQ(*(ret), *(sa));

  ValuePtr vv = std::make_shared<ValueAny>();
  AbstractBasePtr va = std::make_shared<AbstractScalar>();
  ret = vv->ToAbstract();
  ASSERT_TRUE(ret);
  ASSERT_EQ(*(ret), *(va));

  ValuePtr tv = std::make_shared<ValueTuple>(std::vector<ValuePtr>({boolv, i64v, f32v, sv, vv}));
  AbstractBasePtr ta = std::make_shared<AbstractTuple>(AbstractBasePtrList({boola, i64a, f32a, sa, va}));
  ret = tv->ToAbstract();
  ASSERT_TRUE(ret);
  ASSERT_EQ(*(ret), *(ta));
}

TEST_F(TestValue, GetValue) {
  ValuePtr fv = MakeValue("test");
  const char* fv_c = GetValue<const char*>(fv);
  MS_LOG(INFO) << "" << fv_c;
  MS_LOG(INFO) << "" << GetValue<const char*>(fv);
  ASSERT_TRUE(fv_c != nullptr);
}
}  // namespace mindspore

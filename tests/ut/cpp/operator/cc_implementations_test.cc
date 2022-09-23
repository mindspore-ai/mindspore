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
#include <vector>

#include "common/common_test.h"
#include "frontend/operator/cc_implementations.h"

namespace mindspore {
namespace prim {

class TestImplementations : public UT::Common {
 public:
  TestImplementations() {}
  virtual void SetUp() {}
};

TEST_F(TestImplementations, ScalarAddTest) {
  ValuePtrList list;
  list.push_back(MakeValue(static_cast<int64_t>(1)));
  list.push_back(MakeValue(static_cast<int64_t>(2)));
  ASSERT_EQ(ScalarAdd(list)->cast<Int64ImmPtr>()->value(), 3);
  list.clear();

  list.push_back(MakeValue(1.0f));
  list.push_back(MakeValue(1.5f));
  ASSERT_EQ(ScalarAdd(list)->cast<FP32ImmPtr>()->value(), 2.5f);
  list.clear();

  list.push_back(MakeValue(INT64_MAX));
  list.push_back(MakeValue(static_cast<int64_t>(2)));
  try {
    ScalarAdd(list);
    FAIL();
  } catch (std::runtime_error const &err) {
    ASSERT_TRUE(std::string(err.what()).find("Overflow of the sum of two signed number") != std::string::npos);
  }
  list.clear();

  list.push_back(MakeValue(INT64_MIN));
  list.push_back(MakeValue(static_cast<int64_t>(-1)));
  try {
    ScalarAdd(list);
    FAIL();
  } catch (std::runtime_error const &err) {
    ASSERT_TRUE(std::string(err.what()).find("Overflow of the sum of two signed number") != std::string::npos);
  }
  list.clear();
}

TEST_F(TestImplementations, ScalarSubTest) {
  ValuePtrList list;
  list.push_back(MakeValue(static_cast<int64_t>(1)));
  list.push_back(MakeValue(static_cast<int64_t>(3)));
  ASSERT_EQ(ScalarSub(list)->cast<Int64ImmPtr>()->value(), -2);
  list.clear();

  list.push_back(MakeValue(1.0f));
  list.push_back(MakeValue(1.5f));
  ASSERT_EQ(ScalarSub(list)->cast<FP32ImmPtr>()->value(), -0.5f);
  list.clear();

  list.push_back(MakeValue(INT64_MAX));
  list.push_back(MakeValue(static_cast<int64_t>(-1)));
  try {
    ScalarSub(list);
    FAIL();
  } catch (std::runtime_error const &err) {
    ASSERT_TRUE(std::string(err.what()).find("Overflow of the sub of two signed number") != std::string::npos);
  }
  list.clear();

  list.push_back(MakeValue(INT64_MIN));
  list.push_back(MakeValue(static_cast<int64_t>(1)));
  try {
    ScalarSub(list);
    FAIL();
  } catch (std::runtime_error const &err) {
    ASSERT_TRUE(std::string(err.what()).find("Overflow of the sub of two signed number") != std::string::npos);
  }
  list.clear();
}

TEST_F(TestImplementations, ScalarMulTest) {
  ValuePtrList list;
  list.push_back(MakeValue(static_cast<int64_t>(2)));
  list.push_back(MakeValue(static_cast<int64_t>(3)));
  ASSERT_EQ(ScalarMul(list)->cast<Int64ImmPtr>()->value(), 6);
  list.clear();

  list.push_back(MakeValue(2.0f));
  list.push_back(MakeValue(1.5f));
  ASSERT_EQ(ScalarMul(list)->cast<FP32ImmPtr>()->value(), 3.0f);
  list.clear();

  list.push_back(MakeValue(static_cast<int64_t>(10)));
  list.push_back(MakeValue(INT64_MAX));
  try {
    ScalarMul(list);
    FAIL();
  } catch (std::runtime_error const &err) {
    ASSERT_TRUE(std::string(err.what()).find("Overflow of the mul of two signed number") != std::string::npos);
  }
  list.clear();

  list.push_back(MakeValue(INT64_MIN));
  list.push_back(MakeValue(static_cast<int64_t>(-1)));
  try {
    ScalarMul(list);
    FAIL();
  } catch (std::runtime_error const &err) {
    ASSERT_TRUE(std::string(err.what()).find("Overflow of the mul of two signed number") != std::string::npos);
  }
  list.clear();

  list.push_back(MakeValue(static_cast<int64_t>(-2)));
  list.push_back(MakeValue(INT64_MAX));
  try {
    ScalarMul(list);
    FAIL();
  } catch (std::runtime_error const &err) {
    ASSERT_TRUE(std::string(err.what()).find("Overflow of the mul of two signed number") != std::string::npos);
  }
  list.clear();

  list.push_back(MakeValue(static_cast<int64_t>(2)));
  list.push_back(MakeValue(INT64_MIN));
  try {
    ScalarMul(list);
    FAIL();
  } catch (std::runtime_error const &err) {
    ASSERT_TRUE(std::string(err.what()).find("Overflow of the mul of two signed number") != std::string::npos);
  }
  list.clear();

  list.push_back(MakeValue(static_cast<int64_t>(0)));
  list.push_back(MakeValue(INT64_MIN));
  ASSERT_EQ(ScalarDiv(list)->cast<Int64ImmPtr>()->value(), 0);
  list.clear();
}

TEST_F(TestImplementations, ScalarDivTest) {
  ValuePtrList list;
  list.push_back(MakeValue(static_cast<int64_t>(6)));
  list.push_back(MakeValue(static_cast<int64_t>(3)));
  ASSERT_EQ(ScalarDiv(list)->cast<Int64ImmPtr>()->value(), 2);
  list.clear();

  list.push_back(MakeValue(3.0f));
  list.push_back(MakeValue(1.5f));
  ASSERT_EQ(ScalarDiv(list)->cast<FP32ImmPtr>()->value(), 2.0f);
  list.clear();

  list.push_back(MakeValue(INT64_MAX));
  list.push_back(MakeValue(static_cast<int64_t>(0)));
  try {
    ScalarDiv(list);
    FAIL();
  } catch (std::runtime_error const &err) {
    ASSERT_TRUE(std::string(err.what()).find("The divisor could not be zero.") != std::string::npos);
  }
  list.clear();

  list.push_back(MakeValue(INT64_MIN));
  list.push_back(MakeValue(static_cast<int64_t>(-1)));
  try {
    ScalarDiv(list);
    FAIL();
  } catch (std::runtime_error const &err) {
    ASSERT_TRUE(std::string(err.what()).find("Overflow of the div of two signed number") != std::string::npos);
  }
  list.clear();

  list.push_back(MakeValue(static_cast<int64_t>(-1)));
  list.push_back(MakeValue(INT64_MIN));
  ASSERT_EQ(ScalarDiv(list)->cast<Int64ImmPtr>()->value(), 0);
  list.clear();
}

TEST_F(TestImplementations, ScalarModTest) {
  ValuePtrList list;
  list.push_back(MakeValue(static_cast<int64_t>(7)));
  list.push_back(MakeValue(static_cast<int64_t>(3)));
  ASSERT_EQ(ScalarMod(list)->cast<Int64ImmPtr>()->value(), 1);
  list.clear();

  list.push_back(MakeValue(static_cast<int64_t>(-8)));
  list.push_back(MakeValue(static_cast<int64_t>(3)));
  ASSERT_EQ(ScalarMod(list)->cast<Int64ImmPtr>()->value(), -2);
  list.clear();

  list.push_back(MakeValue(static_cast<int64_t>(-9)));
  list.push_back(MakeValue(static_cast<int64_t>(2)));
  ASSERT_EQ(ScalarMod(list)->cast<Int64ImmPtr>()->value(), -1);
  list.clear();

  list.push_back(MakeValue(INT64_MIN));
  list.push_back(MakeValue(static_cast<int64_t>(0)));
  try {
    ScalarMod(list);
    FAIL();
  } catch (std::runtime_error const &err) {
    ASSERT_TRUE(std::string(err.what()).find("Cannot perform modulo operation on zero.") != std::string::npos);
  }
  list.clear();

  list.push_back(MakeValue(INT64_MIN));
  list.push_back(MakeValue(static_cast<int64_t>(-1)));
  try {
    ScalarMod(list);
    FAIL();
  } catch (std::runtime_error const &err) {
    ASSERT_TRUE(std::string(err.what()).find("Overflow of the mod of two signed number") != std::string::npos);
  }
  list.clear();
}

TEST_F(TestImplementations, ScalarUAddTest) {
  ValuePtrList list;
  list.push_back(MakeValue((uint64_t)1));
  ASSERT_EQ(ScalarUAdd(list)->cast<UInt64ImmPtr>()->value(), 1);
  list.clear();
}

TEST_F(TestImplementations, ScalarLogTest) {
  ValuePtrList list;
  list.push_back(MakeValue(static_cast<float>(7.3890560989306495)));
  ASSERT_EQ(ScalarLog(list)->cast<FP32ImmPtr>()->value(), 2.0);
  list.clear();
}

TEST_F(TestImplementations, ScalarUSubTest) {
  ValuePtrList list;
  list.push_back(MakeValue(static_cast<int64_t>(1)));
  ASSERT_EQ(ScalarUSub(list)->cast<Int64ImmPtr>()->value(), -1);
  list.clear();
}

TEST_F(TestImplementations, ScalarEqTest) {
  ValuePtrList list;
  list.push_back(MakeValue(1.0f));
  list.push_back(MakeValue(1.0f));
  ASSERT_EQ(ScalarEq(list)->cast<BoolImmPtr>()->value(), true);
  list.clear();

  list.push_back(MakeValue(1.0f));
  list.push_back(MakeValue(-1.0f));
  ASSERT_EQ(ScalarEq(list)->cast<BoolImmPtr>()->value(), false);
  list.clear();
}

TEST_F(TestImplementations, ScalarLtTest) {
  ValuePtrList list;
  list.push_back(MakeValue(1.0f));
  list.push_back(MakeValue(1.0f));
  ASSERT_EQ(ScalarLt(list)->cast<BoolImmPtr>()->value(), false);
  list.clear();

  list.push_back(MakeValue(1.0f));
  list.push_back(MakeValue(-1.0f));
  ASSERT_EQ(ScalarLt(list)->cast<BoolImmPtr>()->value(), false);
  list.clear();
}

TEST_F(TestImplementations, ScalarGtTest) {
  ValuePtrList list;
  list.push_back(MakeValue(1.0f));
  list.push_back(MakeValue(2.0f));
  ASSERT_EQ(ScalarGt(list)->cast<BoolImmPtr>()->value(), false);
  list.clear();

  list.push_back(MakeValue(2.0f));
  list.push_back(MakeValue(-1.0f));
  ASSERT_EQ(ScalarGt(list)->cast<BoolImmPtr>()->value(), true);
  list.clear();
}

TEST_F(TestImplementations, ScalarNeTest) {
  ValuePtrList list;
  list.push_back(MakeValue(1.0f));
  list.push_back(MakeValue(1.0f));
  ASSERT_EQ(ScalarNe(list)->cast<BoolImmPtr>()->value(), false);
  list.clear();

  list.push_back(MakeValue(1.0f));
  list.push_back(MakeValue(-1.0f));
  ASSERT_EQ(ScalarNe(list)->cast<BoolImmPtr>()->value(), true);
  list.clear();
}

TEST_F(TestImplementations, ScalarLeTest) {
  ValuePtrList list;
  list.push_back(MakeValue(1.0f));
  list.push_back(MakeValue(1.0f));
  ASSERT_EQ(ScalarLe(list)->cast<BoolImmPtr>()->value(), true);
  list.clear();

  list.push_back(MakeValue(1.0f));
  list.push_back(MakeValue(-1.0f));
  ASSERT_EQ(ScalarLe(list)->cast<BoolImmPtr>()->value(), false);
  list.clear();
}

TEST_F(TestImplementations, ScalarGeTest) {
  ValuePtrList list;
  list.push_back(MakeValue(1.0f));
  list.push_back(MakeValue(1.0f));
  ASSERT_EQ(ScalarGe(list)->cast<BoolImmPtr>()->value(), true);
  list.clear();

  list.push_back(MakeValue(1.0f));
  list.push_back(MakeValue(-1.0f));
  ASSERT_EQ(ScalarGe(list)->cast<BoolImmPtr>()->value(), true);
  list.clear();
}

TEST_F(TestImplementations, BoolNotTest) {
  ValuePtrList list;
  list.push_back(MakeValue(true));
  ASSERT_EQ(BoolNot(list)->cast<BoolImmPtr>()->value(), false);
  list.clear();

  list.push_back(MakeValue(false));
  ASSERT_EQ(BoolNot(list)->cast<BoolImmPtr>()->value(), true);
  list.clear();
}

TEST_F(TestImplementations, BoolAndTest) {
  ValuePtrList list;
  list.push_back(MakeValue(true));
  list.push_back(MakeValue(false));
  ASSERT_EQ(BoolAnd(list)->cast<BoolImmPtr>()->value(), false);
  list.clear();

  list.push_back(MakeValue(true));
  list.push_back(MakeValue(true));
  ASSERT_EQ(BoolAnd(list)->cast<BoolImmPtr>()->value(), true);
  list.clear();

  list.push_back(MakeValue(false));
  list.push_back(MakeValue(false));
  ASSERT_EQ(BoolAnd(list)->cast<BoolImmPtr>()->value(), false);
  list.clear();
}

TEST_F(TestImplementations, BoolOrTest) {
  ValuePtrList list;
  list.push_back(MakeValue(true));
  list.push_back(MakeValue(false));
  ASSERT_EQ(BoolOr(list)->cast<BoolImmPtr>()->value(), true);
  list.clear();

  list.push_back(MakeValue(true));
  list.push_back(MakeValue(true));
  ASSERT_EQ(BoolOr(list)->cast<BoolImmPtr>()->value(), true);
  list.clear();

  list.push_back(MakeValue(false));
  list.push_back(MakeValue(false));
  ASSERT_EQ(BoolOr(list)->cast<BoolImmPtr>()->value(), false);
  list.clear();
}

TEST_F(TestImplementations, BoolEqTest) {
  ValuePtrList list;
  list.push_back(MakeValue(true));
  list.push_back(MakeValue(false));
  ASSERT_EQ(BoolEq(list)->cast<BoolImmPtr>()->value(), false);
  list.clear();

  list.push_back(MakeValue(true));
  list.push_back(MakeValue(true));
  ASSERT_EQ(BoolEq(list)->cast<BoolImmPtr>()->value(), true);
  list.clear();

  list.push_back(MakeValue(false));
  list.push_back(MakeValue(false));
  ASSERT_EQ(BoolEq(list)->cast<BoolImmPtr>()->value(), true);
  list.clear();
}

}  // namespace prim
}  // namespace mindspore

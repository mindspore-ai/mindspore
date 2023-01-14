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
#include "common/common_test.h"
#include "ir/dtype.h"
#include "ir/dtype/ref.h"
#include "ir/dtype/number.h"
#include "ir/dtype/container.h"
#include "ir/dtype/empty.h"
#include "ir/scalar.h"

namespace mindspore {
class TestDType : public UT::Common {
 public:
  TestDType() : b(), i16(16), i32(32), i64(64), i64_2(64), u16(16), u32(32), u64(64), f32(32), f64(64), f64_2(64) {}

  void SetUp();
  void TearDown();

  Bool b;
  Int i16;
  Int i32;
  Int i64;
  Int i64_2;
  UInt u16;
  UInt u32;
  UInt u64;
  Float f32;
  Float f64;
  Float f64_2;
};

void TestDType::SetUp() {
  // init resource
}

void TestDType::TearDown() {
  // destroy resource
}

// test_instantiate
// test_cache
TEST_F(TestDType, TestNumber) {
  try {
    Float f = Float(32);
  } catch (std::range_error &e) {
    MS_LOG(ERROR) << "build float 32 failed!!! error:" << e.what();
    ASSERT_TRUE(0);
    return;
  }
  ASSERT_TRUE(1);
}

TEST_F(TestDType, TestList) {
  List l = {std::make_shared<Float>(64), std::make_shared<Int>(8), std::make_shared<Bool>()};
  ASSERT_TRUE(*l.elements()[1] == Int(8));
  List t1 = {std::make_shared<Bool>(), std::make_shared<Int>(32)};
  std::vector<std::shared_ptr<Type>> v2 = {std::make_shared<Bool>(), std::make_shared<Int>(32)};
  List t2(v2);

  ASSERT_EQ(t1, t2);
}

TEST_F(TestDType, TestTuple) {
  Tuple t = {std::make_shared<Float>(64), std::make_shared<Int>(8), std::make_shared<Bool>()};
  ASSERT_TRUE(*t.elements()[1] == Int(8));
  Tuple t1 = {std::make_shared<Bool>(), std::make_shared<Int>(32)};
  std::vector<std::shared_ptr<Type>> v2 = {std::make_shared<Bool>(), std::make_shared<Int>(32)};
  Tuple t2(v2);

  ASSERT_EQ(t1, t2);
}

TEST_F(TestDType, TestDictionary) {
  std::vector<std::pair<ValuePtr, TypePtr>> kv = {
    std::make_pair(std::make_shared<Int64Imm>(8), std::make_shared<Int>(8)),
    std::make_pair(std::make_shared<Int64Imm>(8), std::make_shared<Bool>())};
  Dictionary d1 = Dictionary(kv);
  Dictionary d2 = Dictionary(kv);
  ASSERT_EQ(d1, d2);
}

TEST_F(TestDType, TestFunction) {
  std::vector<std::shared_ptr<Type>> args1{std::make_shared<Int>(32), std::make_shared<Float>(32)};
  std::vector<std::shared_ptr<Type>> args2{std::make_shared<Int>(64), std::make_shared<Float>(64)};
  std::vector<std::shared_ptr<Type>> args3{};
  std::vector<std::shared_ptr<Type>> args4{};

  std::shared_ptr<Type> retval1 = std::make_shared<Float>(64);
  std::shared_ptr<Type> retval2 = std::make_shared<Float>(64);
  std::shared_ptr<Type> retval3 = std::make_shared<Bool>();

  Function fn1 = Function(args1, retval1);
  Function fn2 = Function(args1, retval2);
  Function fn3 = Function(args2, retval1);
  Function fn4 = Function(args2, retval3);

  Function fn5 = Function(args3, retval1);
  Function fn6 = Function(args4, retval1);

  ASSERT_EQ(fn1, fn2);
  ASSERT_FALSE(fn1 != fn2);
  ASSERT_FALSE(fn1 == fn3);
  ASSERT_FALSE(fn1 == fn4);
  ASSERT_FALSE(fn3 == fn4);
  ASSERT_EQ(fn5, fn6);
}

TEST_F(TestDType, TestTypeCloner) {
  Tuple t1 = {std::make_shared<Float>(64), std::make_shared<Int>(64), std::make_shared<Bool>()};
  Tuple t2 = {std::make_shared<Float>(64), std::make_shared<Int>(32), std::make_shared<Bool>()};
  std::shared_ptr<Type> tc = Clone(t1);

  ASSERT_EQ(*tc, t1);
}

TEST_F(TestDType, EqTest) {
  ASSERT_EQ(i64, i64_2);
  ASSERT_FALSE(i64 != i64_2);
  ASSERT_EQ(f64, f64_2);
  ASSERT_FALSE(f64 != f64_2);
  ASSERT_FALSE(f32 == f64);
  ASSERT_FALSE(i32 == i64);
}

// test_get_generic
// test_repr
TEST_F(TestDType, TestRepr) {
  UInt u = UInt(16);
  ASSERT_EQ(u.ToString(), "UInt16");
}

TEST_F(TestDType, TestStringToType) {
  ASSERT_EQ(TypeNone(), *StringToType("None"));
  ASSERT_EQ(TypeType(), *StringToType("TypeType"));
  ASSERT_EQ(Number(), *StringToType("Number"));

  ASSERT_EQ(Bool(), *StringToType("Bool"));

  ASSERT_EQ(UInt(8), *StringToType("UInt8"));
  ASSERT_EQ(UInt(16), *StringToType("UInt16"));
  ASSERT_EQ(UInt(32), *StringToType("UInt32"));
  ASSERT_EQ(UInt(64), *StringToType("UInt64"));

  ASSERT_EQ(Int(8), *StringToType("Int8"));
  ASSERT_EQ(Int(16), *StringToType("Int16"));
  ASSERT_EQ(Int(32), *StringToType("Int32"));
  ASSERT_EQ(Int(64), *StringToType("Int64"));

  ASSERT_EQ(Float(16), *StringToType("Float16"));
  ASSERT_EQ(Float(32), *StringToType("Float32"));
  ASSERT_EQ(Float(64), *StringToType("Float64"));
}

TEST_F(TestDType, TestStringToType2) {
  ASSERT_TRUE(IsIdentidityOrSubclass(StringToType("Tensor"), std::make_shared<TensorType>()));
  ASSERT_TRUE(IsIdentidityOrSubclass(StringToType("List"), std::make_shared<List>()));
  ASSERT_TRUE(IsIdentidityOrSubclass(StringToType("Tuple"), std::make_shared<Tuple>()));
  ASSERT_TRUE(IsIdentidityOrSubclass(StringToType("Tensor[Float32]"), StringToType("Tensor")));
  ASSERT_TRUE(IsIdentidityOrSubclass(StringToType("List[Float32]"), StringToType("List")));
  ASSERT_TRUE(IsIdentidityOrSubclass(StringToType("Tuple[Float64,Int64]"), StringToType("Tuple")));
  ASSERT_TRUE(IsIdentidityOrSubclass(StringToType("Function[(),Int64]"), StringToType("Function")));
  std::cout << StringToType("Function[(Tuple[Float64],Int64),Int32]") << std::endl;
}

TEST_F(TestDType, TestTypeIdNormalize) {
  ASSERT_EQ(kNumberTypeInt, NormalizeTypeId(kNumberTypeInt));
  ASSERT_EQ(kNumberTypeInt, NormalizeTypeId(kNumberTypeInt8));
  ASSERT_EQ(kNumberTypeInt, NormalizeTypeId(kNumberTypeInt16));
  ASSERT_EQ(kNumberTypeInt, NormalizeTypeId(kNumberTypeInt32));
  ASSERT_EQ(kNumberTypeInt, NormalizeTypeId(kNumberTypeInt64));

  ASSERT_TRUE(kNumberTypeInt != NormalizeTypeId(kNumberTypeUInt));
  ASSERT_EQ(kNumberTypeUInt, NormalizeTypeId(kNumberTypeUInt));

  ASSERT_EQ(kNumberTypeFloat, NormalizeTypeId(kNumberTypeFloat));
  ASSERT_EQ(kNumberTypeFloat, NormalizeTypeId(kNumberTypeFloat16));
  ASSERT_EQ(kNumberTypeFloat, NormalizeTypeId(kNumberTypeFloat32));
  ASSERT_EQ(kNumberTypeFloat, NormalizeTypeId(kNumberTypeFloat64));

  ASSERT_EQ(kNumberTypeBool, NormalizeTypeId(kNumberTypeBool));
}
}  // namespace mindspore

// test_type_cloner()

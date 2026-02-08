/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include <string>

#include "common/common_test.h"

#include "ir/anf.h"
#include "ir/tensor.h"
#include "ir/value.h"
#include "ir/tensor_new.h"
#include "utils/anf_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/utils.h"
#include "mindspore/core/ops/infer_info/value_infer_info_adapter.h"
#include "ops/infer_info/abstract_infer_info_adapter.h"

namespace mindspore::ops {
namespace {
const std::string kPrim = "Foo";
const std::string kArg0 = "Arg0";
}  // namespace

using Named = Named;
using abstract::AbstractList;
using abstract::AbstractListPtr;
using tensor::Tensor;
using tensor::TensorPtr;
using tensor::TensorPtrList;

class TestInferInfo : public UT::Common {
 public:
  TestInferInfo() {}
};

template <typename T>
class TypedTestInferInfo : public UT::Common {
 public:
  TypedTestInferInfo() {}
  static const bool is_value_infer_{std::is_same_v<T, ValueInferInfoAdapter>};
};

using InferInfoTypes = ::testing::Types<ValueInferInfoAdapter, AbstractInferInfoAdapter>;
TYPED_TEST_SUITE(TypedTestInferInfo, InferInfoTypes);

static std::unordered_map<TypeId, std::function<ValuePtr(float)>> scalar_creator{
  {kNumberTypeFloat32, [](float x) { return std::make_shared<FP32Imm>(x); }},
  {kNumberTypeInt64, [](float x) { return std::make_shared<Int64Imm>(static_cast<int64_t>(x)); }}};

template <typename T>
ValuePtr make_value(ShapeVector shape, TypeId type, std::vector<T> values) {
  ValuePtr ret;
  if (shape.empty()) {
    ret = scalar_creator[type](values[0]);
  } else {
    ret = tensor::from_buffer(type, shape, values.data(), values.size() * sizeof(T));
  }
  return ret;
}

ValuePtr make_sequence_value(ValuePtrList values) { return std::make_shared<ValueSequence>(values); }

AbstractBasePtr make_abstract(ValuePtr value, bool with_value = true) {
  MS_EXCEPTION_IF_NULL(value);
  AbstractBasePtr abs = value->ToAbstract();
  if (!with_value) {
    abs->set_value(kValueAny);
  }
  return abs;
}

AbstractListPtr make_abstract_list(ValuePtrList values) {
  AbstractBasePtrList abs_list;
  std::transform(values.begin(), values.end(), std::back_inserter(abs_list),
                 [](const ValuePtr &value) { return value->ToAbstract(); });
  auto abs_list_ptr = std::make_shared<AbstractList>(abs_list);
  return abs_list_ptr;
}

#define make_infer_ptr(name, prim_name, arg_name, shape, type, value, with_value)                  \
  if constexpr (this->is_value_infer_) {                                                           \
    auto value_ptr = make_value(shape, type, value);                                               \
    name = std::make_unique<TypeParam>(value_ptr, prim_name, arg_name);                            \
  } else {                                                                                         \
    auto value_ptr = make_value(shape, type, value);                                               \
    name = std::make_unique<TypeParam>(make_abstract(value_ptr, with_value), prim_name, arg_name); \
  }

#define make_infer_ptr_from_seq_value(name, prim_name, arg_name, values, with_value) \
  if constexpr (this->is_value_infer_) {                                             \
    auto seq_value = make_sequence_value(values);                                    \
    name = std::make_unique<TypeParam>(seq_value, prim_name, arg_name);              \
  } else {                                                                           \
    auto abs = make_abstract_list(values);                                           \
    name = std::make_unique<TypeParam>(abs, prim_name, arg_name);                    \
  }

#define expect_inferinfo_throw(statement, prim_name, arg_name) \
  try {                                                        \
    EXPECT_ANY_THROW(statement);                               \
  } catch (const std::runtime_error &e) {                      \
    std::string msg = e.what();                                \
    EXPECT_TRUE(msg.find(prim_name) != std::string::npos);     \
    EXPECT_TRUE(msg.find(arg_name) != std::string::npos);      \
  }

/// Feature: GetShape, GetType
/// Description: Normal input for two infers
/// Expectation: Success
TYPED_TEST(TypedTestInferInfo, test_get_shape_type) {
  ShapeVector shape{1, 3};
  TypeId type = kNumberTypeFloat32;
  std::vector<float> value{1., 2., 3.};

  InferInfoPtr infer_info;
  make_infer_ptr(infer_info, kPrim, kArg0, shape, type, value, true);

  auto get_shape = infer_info->GetShape();
  auto get_type = infer_info->GetType();
  EXPECT_EQ(get_shape, shape);
  EXPECT_EQ(get_type, type);

  EXPECT_FALSE(infer_info->IsSequence());
  EXPECT_FALSE(infer_info->IsNone());
}

/// Feature: IsDynamic, IsDynamicRank
/// Description: Dynamic input for AbstractInfer
/// Expectation: Success
TEST_F(TestInferInfo, test_dynamic_shape) {
  // dynamic shape
  ShapeVector shape{-1, -1};
  TypeId type = kNumberTypeFloat32;
  std::vector<float> value{1., 2., 3.};

  auto abs = abstract::MakeAbstract({shape}, {type});
  AbstractInferInfoAdapter infer_info(abs, kPrim, kArg0);

  EXPECT_TRUE(infer_info.IsDynamic());
  EXPECT_FALSE(infer_info.IsDynamicRank());

  shape = {-2};
  abs = abstract::MakeAbstract({shape}, {type});
  AbstractInferInfoAdapter infer_info2(abs, kPrim, kArg0);

  EXPECT_TRUE(infer_info2.IsDynamic());
  EXPECT_TRUE(infer_info2.IsDynamicRank());
}

/// Feature: GetValue
/// Description: Scalar and Tensor input for two infers
/// Expectation: Success
TYPED_TEST(TypedTestInferInfo, test_get_value) {
  // test scalar value
  ShapeVector shape{};
  TypeId type = kNumberTypeInt64;
  std::vector<int64_t> value{1};

  InferInfoPtr infer_info;
  make_infer_ptr(infer_info, kPrim, kArg0, shape, type, value, true);

  auto get_value = infer_info->GetScalarValue<int64_t>();
  EXPECT_TRUE(get_value.has_value());
  EXPECT_EQ(get_value.value(), value[0]);
  EXPECT_ANY_THROW(infer_info->GetArrayValue<int64_t>());

  // test array value
  shape = {1, 3};
  type = kNumberTypeFloat32;
  std::vector<float> value2{1., 2., 3.};
  make_infer_ptr(infer_info, kPrim, kArg0, shape, type, value2, true);

  auto get_value2 = infer_info->GetArrayValue<float>();
  EXPECT_TRUE(get_value2.has_value());
  EXPECT_EQ(get_value2.value().ToVector(), value2);
  EXPECT_ANY_THROW(infer_info->GetScalarValue<float>());

  // test value unavailable
  if constexpr (!this->is_value_infer_) {
    make_infer_ptr(infer_info, kPrim, kArg0, shape, type, value2, false);
    get_value2 = infer_info->GetArrayValue<float>();
    EXPECT_FALSE(get_value2.has_value());
  }
}

/// Feature: GetValue
/// Description: Abstract input without value
/// Expectation: Success
TEST_F(TestInferInfo, test_abstract_without_value) {
  ShapeVector shape{2, 2};
  TypeId type = kNumberTypeFloat32;
  auto abs = abstract::MakeAbstract(shape, type);
  AbstractInferInfoAdapter infer_info(abs, kPrim, kArg0);
  auto value = infer_info.GetArrayValue<float>();
  EXPECT_FALSE(value.has_value());
}

/// Feature: Sequence
/// Description: Sequence input for two infers
/// Expectation: Success
TYPED_TEST(TypedTestInferInfo, test_sequence) {
  ShapeVector shape1{1, 3};
  TypeId type1 = kNumberTypeFloat32;
  std::vector<float> value1{1., 2., 3.};
  auto value_ptr1 = make_value(shape1, type1, value1);

  ShapeVector shape2{1};
  TypeId type2 = kNumberTypeFloat32;
  std::vector<float> value2{4.};
  auto value_ptr2 = make_value(shape2, type2, value2);

  InferInfoPtr infer_info;
  ValuePtrList values{value_ptr1, value_ptr2};
  make_infer_ptr_from_seq_value(infer_info, kPrim, kArg0, values, true);

  EXPECT_TRUE(infer_info->IsSequence());
  EXPECT_FALSE(infer_info->IsDynamicSequence());
  EXPECT_FALSE(infer_info->IsNone());

  // shape and type
  auto elements = infer_info->GetSequenceElements();
  EXPECT_EQ(elements.size(), 2);
  auto get_shape1 = elements[0]->GetShape();
  auto get_shape2 = elements[1]->GetShape();
  auto get_type1 = elements[0]->GetType();
  auto get_type2 = elements[1]->GetType();
  EXPECT_EQ(get_shape1, shape1);
  EXPECT_EQ(get_shape2, shape2);
  EXPECT_EQ(get_type1, type1);
  EXPECT_EQ(get_type2, type2);

  // value
  auto get_value1 = elements[0]->GetArrayValue<float>();
  auto get_value2 = elements[1]->GetArrayValue<float>();
  EXPECT_TRUE(get_value1.has_value());
  EXPECT_TRUE(get_value2.has_value());
  EXPECT_EQ(get_value1.value().ToVector(), value1);
  EXPECT_EQ(get_value2.value().ToVector(), value2);

  // throw
  expect_inferinfo_throw(infer_info->GetShape(), kPrim, kArg0);
  expect_inferinfo_throw(infer_info->GetType(), kPrim, kArg0);
  EXPECT_ANY_THROW(infer_info->GetScalarValue<int64_t>());
  EXPECT_ANY_THROW(infer_info->GetArrayValue<int64_t>());
}

/// Feature: Dynamic Sequence
/// Description: Dynamic Sequence input for AbstractInfer
/// Expectation: Success
TEST_F(TestInferInfo, test_dynamic_sequence) {
  ShapeVector shape{1, 3};
  TypeId type = kNumberTypeFloat32;
  std::vector<float> value{1., 2., 3.};
  auto value1 = make_value(shape, type, value);
  auto value2 = make_value(shape, type, value);

  auto sequence_abs = make_abstract_list({value1, value2});
  sequence_abs->CheckAndConvertToDynamicLenSequence();
  AbstractInferInfoAdapter infer_info(sequence_abs, kPrim, kArg0);

  EXPECT_TRUE(infer_info.IsSequence());
  EXPECT_TRUE(infer_info.IsDynamicSequence());
  auto element_info = infer_info.GetDynamicSequenceElement();
  auto get_shape = element_info->GetShape();
  auto get_type = element_info->GetType();
  EXPECT_EQ(get_shape, shape);
  EXPECT_EQ(get_type, type);

  expect_inferinfo_throw(infer_info.GetSequenceElements(), kPrim, kArg0);
}

/// Feature: None as input
/// Description: GetShape, GetType etc. throw for None
/// Expectation: Error thrown
TYPED_TEST(TypedTestInferInfo, test_none) {
  InferInfoPtr infer_info;
  if constexpr (this->is_value_infer_) {
    infer_info = std::make_unique<TypeParam>(kNone, kPrim, kArg0);
  } else {
    infer_info = std::make_unique<TypeParam>(std::make_shared<abstract::AbstractNone>(), kPrim, kArg0);
  }
  ASSERT_TRUE(infer_info->IsNone());
  EXPECT_FALSE(infer_info->IsSequence());
  expect_inferinfo_throw(infer_info->GetShape(), kPrim, kArg0);
  expect_inferinfo_throw(infer_info->GetType(), kPrim, kArg0);
  EXPECT_ANY_THROW(infer_info->GetScalarValue<int64_t>());
  EXPECT_ANY_THROW(infer_info->GetArrayValue<float>());
}
}  // namespace mindspore::ops

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
#include <iostream>
#include <memory>

#include "common/common_test.h"

#include "abstract/dshape.h"
#include "abstract/abstract_value.h"
#include "utils/log_adapter.h"
namespace mindspore {
namespace abstract {
class DynamicSequenceTestUtils : public UT::Common {
 public:
  DynamicSequenceTestUtils() {}
  virtual void SetUp() {}
  virtual void TearDown() {}
};

AbstractListPtr BuildDynamicAbstractList(const AbstractBasePtrList &elements) {
  AbstractListPtr ret = std::make_shared<AbstractList>(elements);
  ret->CheckAndConvertToDynamicLenSequence();
  return ret;
}

AbstractTuplePtr BuildDynamicAbstractTuple(const AbstractBasePtrList &elements) {
  AbstractTuplePtr ret = std::make_shared<AbstractTuple>(elements);
  ret->CheckAndConvertToDynamicLenSequence();
  return ret;
}

/// Feature: AbstractList with dynamic length.
/// Description: Generate abstract list with dynamic length.
/// Expectation: No exception.
TEST_F(DynamicSequenceTestUtils, test_generate_dynamic_length_list) {
  AbstractScalarPtr int_num1 = std::make_shared<AbstractScalar>(1);
  AbstractScalarPtr int_num2 = std::make_shared<AbstractScalar>(2);
  AbstractScalarPtr float_num1 = std::make_shared<AbstractScalar>(1.0);
  AbstractScalarPtr float_num2 = std::make_shared<AbstractScalar>(2.0);
  AbstractBasePtrList elements1{int_num1, int_num2};
  AbstractBasePtrList elements2{float_num1, float_num2};
  auto abs_list_1 = BuildDynamicAbstractList(elements1);
  ASSERT_TRUE(abs_list_1->isa<AbstractList>());
  ASSERT_TRUE(abs_list_1->dynamic_len());
  ASSERT_TRUE(abs_list_1->dynamic_len_element_abs() != nullptr);
  ASSERT_TRUE(*(abs_list_1->dynamic_len_element_abs()->BuildType()) == *(int_num2->BuildType()));
  ASSERT_TRUE(*(abs_list_1->dynamic_len_element_abs()->BuildShape()) == *(int_num2->BuildShape()));
  auto abs_list_2 = BuildDynamicAbstractList(elements2);
  ASSERT_TRUE(abs_list_2->isa<AbstractList>());
  ASSERT_TRUE(abs_list_2->dynamic_len());
  ASSERT_TRUE(abs_list_2->dynamic_len_element_abs() != nullptr);
  ASSERT_TRUE(*(abs_list_2->dynamic_len_element_abs()->BuildType()) == *(float_num2->BuildType()));
  ASSERT_TRUE(*(abs_list_2->dynamic_len_element_abs()->BuildShape()) == *(float_num2->BuildShape()));
}

/// Feature: AbstractList with dynamic length.
/// Description: Generate abstract list with dynamic length.
/// Expectation: Runtime error.
TEST_F(DynamicSequenceTestUtils, test_generate_dynamic_length_list_error) {
  AbstractScalarPtr int_num = std::make_shared<AbstractScalar>(1);
  AbstractScalarPtr float_num = std::make_shared<AbstractScalar>(1.0);
  AbstractBasePtrList elements{int_num, float_num};
  try {
    BuildDynamicAbstractList(elements);
    FAIL();
  } catch (std::runtime_error const &err) {
    ASSERT_TRUE(std::string(err.what()).find("The element type do not match") != std::string::npos);
  }
}

/// Feature: AbstractList with dynamic length.
/// Description: Generate empty abstract list with dynamic length.
/// Expectation: Runtime error.
TEST_F(DynamicSequenceTestUtils, test_generate_dynamic_length_empty_list) {
  auto abs_list = BuildDynamicAbstractList(AbstractBasePtrList{});
  ASSERT_TRUE(abs_list->dynamic_len());
  ASSERT_TRUE(abs_list->dynamic_len_element_abs() == nullptr);
  AbstractScalarPtr int_abs = std::make_shared<AbstractScalar>(1);
  abs_list->set_dynamic_len_element_abs(int_abs);
  ASSERT_TRUE(abs_list->dynamic_len_element_abs() != nullptr);
  AbstractScalarPtr test_abs = std::make_shared<AbstractScalar>(1);
  ASSERT_TRUE(*(abs_list->dynamic_len_element_abs()->BuildType()) == *(test_abs->BuildType()));
  ASSERT_TRUE(*(abs_list->dynamic_len_element_abs()->BuildShape()) == *(test_abs->BuildShape()));
  AbstractScalarPtr float_abs = std::make_shared<AbstractScalar>(1.0);
  abs_list->set_dynamic_len_element_abs(float_abs);
  ASSERT_TRUE(abs_list->dynamic_len_element_abs() != nullptr);
  test_abs = std::make_shared<AbstractScalar>(2.0);
  ASSERT_TRUE(*(abs_list->dynamic_len_element_abs()->BuildType()) == *(test_abs->BuildType()));
  ASSERT_TRUE(*(abs_list->dynamic_len_element_abs()->BuildShape()) == *(test_abs->BuildShape()));
}

/// Feature: AbstractList with dynamic length.
/// Description: Abstract list with dynamic length build value.
/// Expectation: No exception.
TEST_F(DynamicSequenceTestUtils, test_dynamic_length_list_build_value) {
  auto abs_empty_list = BuildDynamicAbstractList(AbstractBasePtrList{});
  ASSERT_TRUE(abs_empty_list->BuildValue() == kValueAny);
  AbstractScalarPtr int_num1 = std::make_shared<AbstractScalar>(1);
  AbstractScalarPtr int_num2 = std::make_shared<AbstractScalar>(2);
  AbstractBasePtrList elements1{int_num1, int_num2};
  auto abs_list = BuildDynamicAbstractList(elements1);
  ASSERT_TRUE(abs_list->BuildValue() == kValueAny);
}

/// Feature: AbstractList with dynamic length.
/// Description: Abstract list with dynamic length build shape.
/// Expectation: No exception.
TEST_F(DynamicSequenceTestUtils, test_dynamic_length_list_build_shape) {
  auto abs_empty_list = BuildDynamicAbstractList(AbstractBasePtrList{});
  ASSERT_TRUE(abs_empty_list->BuildShape() == kDynamicSequenceShape);
  AbstractScalarPtr int_num1 = std::make_shared<AbstractScalar>(1);
  AbstractScalarPtr int_num2 = std::make_shared<AbstractScalar>(2);
  AbstractBasePtrList elements1{int_num1, int_num2};
  auto abs_list = BuildDynamicAbstractList(elements1);
  ASSERT_TRUE(abs_list->BuildShape() == kDynamicSequenceShape);
}

/// Feature: AbstractList with dynamic length.
/// Description: Abstract list with dynamic length build type.
/// Expectation: No exception.
TEST_F(DynamicSequenceTestUtils, test_dynamic_length_list_build_type) {
  auto abs_empty_list = BuildDynamicAbstractList(AbstractBasePtrList{});
  auto abs_empty_list_type = dyn_cast<List>(abs_empty_list->BuildType());
  ASSERT_TRUE(abs_empty_list_type != nullptr);
  ASSERT_TRUE(abs_empty_list_type->dynamic_len());
  ASSERT_TRUE(abs_empty_list_type->dynamic_element_type() == nullptr);
  AbstractScalarPtr int_num1 = std::make_shared<AbstractScalar>(1);
  AbstractScalarPtr int_num2 = std::make_shared<AbstractScalar>(2);
  AbstractBasePtrList elements1{int_num1, int_num2};
  auto abs_list = BuildDynamicAbstractList(elements1);
  auto abs_list_type = dyn_cast<List>(abs_list->BuildType());
  ASSERT_TRUE(abs_list_type != nullptr);
  ASSERT_TRUE(abs_list_type->dynamic_len());
  ASSERT_TRUE(abs_list_type->dynamic_element_type() != nullptr);
  ASSERT_TRUE(*(abs_list_type->dynamic_element_type()) == *(int_num2->BuildType()));
}

/// Feature: AbstractTuple with dynamic length.
/// Description: Generate abstract tuple with dynamic length.
/// Expectation: No exception.
TEST_F(DynamicSequenceTestUtils, test_generate_dynamic_length_tuple) {
  AbstractScalarPtr int_num1 = std::make_shared<AbstractScalar>(1);
  AbstractScalarPtr int_num2 = std::make_shared<AbstractScalar>(2);
  AbstractScalarPtr float_num1 = std::make_shared<AbstractScalar>(1.0);
  AbstractScalarPtr float_num2 = std::make_shared<AbstractScalar>(2.0);
  AbstractBasePtrList elements1{int_num1, int_num2};
  AbstractBasePtrList elements2{float_num1, float_num2};
  auto abs_tuple_1 = BuildDynamicAbstractTuple(elements1);
  ASSERT_TRUE(abs_tuple_1->isa<AbstractTuple>());
  ASSERT_TRUE(abs_tuple_1->dynamic_len());
  ASSERT_TRUE(abs_tuple_1->dynamic_len_element_abs() != nullptr);
  ASSERT_TRUE(*(abs_tuple_1->dynamic_len_element_abs()->BuildType()) == *(int_num2->BuildType()));
  ASSERT_TRUE(*(abs_tuple_1->dynamic_len_element_abs()->BuildShape()) == *(int_num2->BuildShape()));
  auto abs_tuple_2 = BuildDynamicAbstractTuple(elements2);
  ASSERT_TRUE(abs_tuple_2->isa<AbstractTuple>());
  ASSERT_TRUE(abs_tuple_2->dynamic_len());
  ASSERT_TRUE(abs_tuple_2->dynamic_len_element_abs() != nullptr);
  ASSERT_TRUE(*(abs_tuple_2->dynamic_len_element_abs()->BuildType()) == *(float_num2->BuildType()));
  ASSERT_TRUE(*(abs_tuple_2->dynamic_len_element_abs()->BuildShape()) == *(float_num2->BuildShape()));
}

/// Feature: AbstractTuple with dynamic length.
/// Description: Generate abstract tuple with dynamic length.
/// Expectation: Runtime error.
TEST_F(DynamicSequenceTestUtils, test_generate_dynamic_length_tuple_error) {
  AbstractScalarPtr int_num = std::make_shared<AbstractScalar>(1);
  AbstractScalarPtr float_num = std::make_shared<AbstractScalar>(1.0);
  AbstractBasePtrList elements{int_num, float_num};
  try {
    BuildDynamicAbstractTuple(elements);
    FAIL();
  } catch (std::runtime_error const &err) {
    ASSERT_TRUE(std::string(err.what()).find("The element type do not match") != std::string::npos);
  }
}

/// Feature: AbstractTuple with dynamic length.
/// Description: Generate empty abstract tuple with dynamic length.
/// Expectation: Runtime error.
TEST_F(DynamicSequenceTestUtils, test_generate_dynamic_length_empty_tuple) {
  auto abs_tuple = BuildDynamicAbstractTuple(AbstractBasePtrList{});
  ASSERT_TRUE(abs_tuple->dynamic_len());
  ASSERT_TRUE(abs_tuple->dynamic_len_element_abs() == nullptr);
  AbstractScalarPtr int_abs = std::make_shared<AbstractScalar>(1);
  abs_tuple->set_dynamic_len_element_abs(int_abs);
  ASSERT_TRUE(abs_tuple->dynamic_len_element_abs() != nullptr);
  AbstractScalarPtr test_abs = std::make_shared<AbstractScalar>(1);
  ASSERT_TRUE(*(abs_tuple->dynamic_len_element_abs()->BuildType()) == *(test_abs->BuildType()));
  ASSERT_TRUE(*(abs_tuple->dynamic_len_element_abs()->BuildShape()) == *(test_abs->BuildShape()));
  AbstractScalarPtr float_abs = std::make_shared<AbstractScalar>(1.0);
  abs_tuple->set_dynamic_len_element_abs(float_abs);
  ASSERT_TRUE(abs_tuple->dynamic_len_element_abs() != nullptr);
  test_abs = std::make_shared<AbstractScalar>(2.0);
  ASSERT_TRUE(*(abs_tuple->dynamic_len_element_abs()->BuildType()) == *(test_abs->BuildType()));
  ASSERT_TRUE(*(abs_tuple->dynamic_len_element_abs()->BuildShape()) == *(test_abs->BuildShape()));
}

/// Feature: AbstractTuple with dynamic length.
/// Description: Abstract tuple with dynamic length build value.
/// Expectation: No exception.
TEST_F(DynamicSequenceTestUtils, test_dynamic_length_tuple_build_value) {
  auto abs_empty_tuple = BuildDynamicAbstractTuple(AbstractBasePtrList{});
  ASSERT_TRUE(abs_empty_tuple->BuildValue() == kValueAny);
  AbstractScalarPtr int_num1 = std::make_shared<AbstractScalar>(1);
  AbstractScalarPtr int_num2 = std::make_shared<AbstractScalar>(2);
  AbstractBasePtrList elements1{int_num1, int_num2};
  auto abs_tuple = BuildDynamicAbstractTuple(elements1);
  ASSERT_TRUE(abs_tuple->BuildValue() == kValueAny);
}

/// Feature: AbstractTuple with dynamic length.
/// Description: Abstract tuple with dynamic length build shape.
/// Expectation: No exception.
TEST_F(DynamicSequenceTestUtils, test_dynamic_length_tuple_build_shape) {
  auto abs_empty_tuple = BuildDynamicAbstractTuple(AbstractBasePtrList{});
  ASSERT_TRUE(abs_empty_tuple->BuildShape() == kDynamicSequenceShape);
  AbstractScalarPtr int_num1 = std::make_shared<AbstractScalar>(1);
  AbstractScalarPtr int_num2 = std::make_shared<AbstractScalar>(2);
  AbstractBasePtrList elements1{int_num1, int_num2};
  auto abs_tuple = BuildDynamicAbstractTuple(elements1);
  ASSERT_TRUE(abs_tuple->BuildShape() == kDynamicSequenceShape);
}

/// Feature: AbstractTuple with dynamic length.
/// Description: Abstract tuple with dynamic length build type.
/// Expectation: No exception.
TEST_F(DynamicSequenceTestUtils, test_dynamic_length_tuple_build_type) {
  auto abs_empty_tuple = BuildDynamicAbstractTuple(AbstractBasePtrList{});
  auto abs_empty_tuple_type = dyn_cast<Tuple>(abs_empty_tuple->BuildType());
  ASSERT_TRUE(abs_empty_tuple_type != nullptr);
  ASSERT_TRUE(abs_empty_tuple_type->dynamic_len());
  ASSERT_TRUE(abs_empty_tuple_type->dynamic_element_type() == nullptr);
  AbstractScalarPtr int_num1 = std::make_shared<AbstractScalar>(1);
  AbstractScalarPtr int_num2 = std::make_shared<AbstractScalar>(2);
  AbstractBasePtrList elements1{int_num1, int_num2};
  auto abs_tuple = BuildDynamicAbstractTuple(elements1);
  auto abs_tuple_type = dyn_cast<Tuple>(abs_tuple->BuildType());
  ASSERT_TRUE(abs_tuple_type != nullptr);
  ASSERT_TRUE(abs_tuple_type->dynamic_len());
  ASSERT_TRUE(abs_tuple_type->dynamic_element_type() != nullptr);
  ASSERT_TRUE(*(abs_tuple_type->dynamic_element_type()) == *(int_num2->BuildType()));
}

/// Feature: AbstractSequence with dynamic length.
/// Description: AbstractSequence call operator[].
/// Expectation: Runtime error.
TEST_F(DynamicSequenceTestUtils, test_dynamic_length_sequence_operator_getitem) {
  AbstractScalarPtr int_num1 = std::make_shared<AbstractScalar>(1);
  AbstractScalarPtr int_num2 = std::make_shared<AbstractScalar>(2);
  AbstractBasePtrList elements1{int_num1, int_num2};
  auto abs_tuple = BuildDynamicAbstractTuple(elements1);
  try {
    (void)(*abs_tuple)[0];
    FAIL();
  } catch (std::runtime_error const &err) {
    ASSERT_TRUE(std::string(err.what()).find("Can not get element from dynamic length sequence") != std::string::npos);
  }
}

/// Feature: AbstractSequence with dynamic length.
/// Description: AbstractSequence call function empty().
/// Expectation: Runtime error.
TEST_F(DynamicSequenceTestUtils, test_dynamic_length_sequence_function_empty) {
  AbstractScalarPtr int_num1 = std::make_shared<AbstractScalar>(1);
  AbstractScalarPtr int_num2 = std::make_shared<AbstractScalar>(2);
  AbstractBasePtrList elements1{int_num1, int_num2};
  auto abs_tuple = BuildDynamicAbstractTuple(elements1);
  try {
    (void)abs_tuple->empty();
    FAIL();
  } catch (std::runtime_error const &err) {
    ASSERT_TRUE(std::string(err.what()).find("Can not call function empty() for dynamic length") != std::string::npos);
  }
}

/// Feature: AbstractSequence with dynamic length.
/// Description: AbstractSequence call function size().
/// Expectation: Runtime error.
TEST_F(DynamicSequenceTestUtils, test_dynamic_length_sequence_function_size) {
  auto abs_empty_tuple = BuildDynamicAbstractTuple(AbstractBasePtrList{});
  ASSERT_TRUE(abs_empty_tuple->size() == 0);
  AbstractScalarPtr int_num1 = std::make_shared<AbstractScalar>(1);
  AbstractScalarPtr int_num2 = std::make_shared<AbstractScalar>(2);
  AbstractBasePtrList elements1{int_num1, int_num2};
  auto abs_tuple = BuildDynamicAbstractTuple(elements1);
  try {
    (void)abs_tuple->size();
    FAIL();
  } catch (std::runtime_error const &err) {
    ASSERT_TRUE(std::string(err.what()).find("Can not get size for dynamic length sequence") != std::string::npos);
  }
}

/// Feature: AbstractSequence with dynamic length.
/// Description: AbstractSequence call operator==.
/// Expectation: No exception.
TEST_F(DynamicSequenceTestUtils, test_dynamic_length_sequence_operator_equal) {
  AbstractScalarPtr int_num1 = std::make_shared<AbstractScalar>(1);
  AbstractScalarPtr int_num2 = std::make_shared<AbstractScalar>(2);
  AbstractScalarPtr int_num3 = std::make_shared<AbstractScalar>(2);
  AbstractScalarPtr float_num1 = std::make_shared<AbstractScalar>(1.0);
  AbstractScalarPtr float_num2 = std::make_shared<AbstractScalar>(2.0);
  AbstractScalarPtr float_num3 = std::make_shared<AbstractScalar>(2.0);
  AbstractBasePtrList int_elements1{int_num1, int_num2};
  AbstractBasePtrList int_elements2{int_num3};
  AbstractBasePtrList float_elements1{float_num1, float_num2};
  AbstractBasePtrList float_elements2{float_num3};
  AbstractListPtr int_abs_list_1 = BuildDynamicAbstractList(int_elements1);
  AbstractListPtr int_abs_list_2 = BuildDynamicAbstractList(int_elements2);
  AbstractListPtr float_abs_list_1 = BuildDynamicAbstractList(float_elements1);
  AbstractListPtr float_abs_list_2 = BuildDynamicAbstractList(float_elements2);
  ASSERT_TRUE(*int_abs_list_1 == *int_abs_list_2);
  ASSERT_TRUE(*float_abs_list_1 == *float_abs_list_2);
  ASSERT_FALSE(*float_abs_list_2 == *int_abs_list_2);
  AbstractTuplePtr int_abs_tuple_2 = BuildDynamicAbstractTuple(int_elements2);
  ASSERT_FALSE(*int_abs_list_1 == *int_abs_tuple_2);
  auto abs_empty_list = BuildDynamicAbstractList(AbstractBasePtrList{});
  ASSERT_FALSE(*abs_empty_list == *int_abs_list_1);
  ASSERT_FALSE(*int_abs_list_1 == *abs_empty_list);
}
}  // namespace abstract
}  // namespace mindspore

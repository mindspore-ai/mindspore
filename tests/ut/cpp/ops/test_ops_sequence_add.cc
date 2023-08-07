/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include <vector>
#include <memory>
#include "common/common_test.h"
#include "ops/sequence_op_name.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "utils/ms_context.h"
#include "ops/test_ops.h"
#include "include/backend/optimizer/helper.h"

namespace mindspore {
namespace ops {
struct TestSequenceAddParams {
  bool is_input_tensor;
  bool is_input_dyn_len;
  bool is_input_tuple;
  std::vector<ShapeVector> x1;
  TypePtr x1_type;
  std::vector<ShapeVector> x2;
  TypePtr x2_type;
  std::vector<ShapeVector> out;
  TypePtr out_type;
};

class TestSequenceAddDyn : public TestOps, public testing::WithParamInterface<TestSequenceAddParams> {};

void compare_tuple(AbstractBasePtrList &list1, AbstractBasePtrList &list2, AbstractBasePtrList &out_list,
                   bool is_input_dyn_len) {
  auto input_1 = std::make_shared<abstract::AbstractTuple>(list1);
  auto input_2 = std::make_shared<abstract::AbstractTuple>(list2);
  auto expect = std::make_shared<abstract::AbstractTuple>(out_list);
  if (is_input_dyn_len) {
    input_1->CheckAndConvertToDynamicLenSequence();
    input_2->CheckAndConvertToDynamicLenSequence();
    expect->CheckAndConvertToDynamicLenSequence();
  }
  auto prim = std::make_shared<Primitive>(kSequenceAddOpName);
  auto out_abstract = opt::CppInferShapeAndType(prim, {input_1, input_2});
  ASSERT_TRUE(*out_abstract == *expect);
}

void compare_list(AbstractBasePtrList &list1, AbstractBasePtrList &list2, AbstractBasePtrList &out_list,
                  bool is_input_dyn_len) {
  auto input_1 = std::make_shared<abstract::AbstractList>(list1);
  auto input_2 = std::make_shared<abstract::AbstractList>(list2);
  auto prim = std::make_shared<Primitive>(kSequenceAddOpName);
  if (is_input_dyn_len) {
    input_1->CheckAndConvertToDynamicLenSequence();
    input_2->CheckAndConvertToDynamicLenSequence();
    auto out_abstract = opt::CppInferShapeAndType(prim, {input_1, input_2});
    auto expect = std::make_shared<abstract::AbstractList>(out_list);
    expect->CheckAndConvertToDynamicLenSequence();
    ASSERT_TRUE(*out_abstract == *expect);
  } else {
    auto out_abstract = opt::CppInferShapeAndType(prim, {input_1, input_2});
    auto expect = std::make_shared<abstract::AbstractTuple>(out_list);
    ASSERT_TRUE(*out_abstract == *expect);
  }
}

TEST_P(TestSequenceAddDyn, sequence_add_dyn_shape) {
  const auto &param = GetParam();
  AbstractBasePtrList list1;
  if (!param.is_input_tensor) {
    for (auto elem : param.x1[0]) {
      auto s_elem = std::make_shared<abstract::AbstractScalar>(elem);
      list1.push_back(s_elem);
    }
  } else {
    for (auto elem : param.x1) {
      auto t_elem = std::make_shared<abstract::AbstractTensor>(param.x1_type, elem);
      list1.push_back(t_elem);
    }
  }

  AbstractBasePtrList list2;
  if (!param.is_input_tensor) {
    for (auto elem : param.x2[0]) {
      auto s_elem = std::make_shared<abstract::AbstractScalar>(elem);
      list2.push_back(s_elem);
    }
  } else {
    for (auto elem : param.x2) {
      auto t_elem = std::make_shared<abstract::AbstractTensor>(param.x2_type, elem);
      list2.push_back(t_elem);
    }
  }

  AbstractBasePtrList out_list;
  if (!param.is_input_tensor) {
    for (auto elem : param.out[0]) {
      auto o_elem = std::make_shared<abstract::AbstractScalar>(elem);
      out_list.push_back(o_elem);
    }
  } else {
    for (auto elem : param.out) {
      auto t_elem = std::make_shared<abstract::AbstractTensor>(param.out_type, elem);
      out_list.push_back(t_elem);
    }
  }

  if (param.is_input_tuple) {
    compare_tuple(list1, list2, out_list, param.is_input_dyn_len);
  } else {
    compare_list(list1, list2, out_list, param.is_input_dyn_len);
  }
}

INSTANTIATE_TEST_CASE_P(TestSequenceAddDyn, TestSequenceAddDyn,
                        testing::Values(TestSequenceAddParams {true, true, false,
                                                               std::vector<ShapeVector>{{2, 3}, {2, 3}},
                                                               kInt32,
                                                               std::vector<ShapeVector>{{2, 3}, {2, 3}},
                                                               kInt32,
                                                               std::vector<ShapeVector>{{2, 3}, {2, 3}, {2, 3}, {2, 3}},
                                                               kInt32},
                                        TestSequenceAddParams {true, false, true,
                                                               std::vector<ShapeVector>{{2, 3}, {2, 3}},
                                                               kInt32,
                                                               std::vector<ShapeVector>{{2, 3}, {2, 3}},
                                                               kInt32,
                                                               std::vector<ShapeVector>{{2, 3}, {2, 3}, {2, 3}, {2, 3}},
                                                               kInt32},
                                        TestSequenceAddParams {false, true, false,
                                                               std::vector<ShapeVector>{{1, 2, 3}}, kInt32,
                                                               std::vector<ShapeVector>{{4, 5, 6}}, kInt32,
                                                               std::vector<ShapeVector>{{1, 2, 3, 4, 5, 6}}, kInt32},
                                        TestSequenceAddParams {false, false, true,
                                                               std::vector<ShapeVector>{{1, 2, 3}}, kInt32,
                                                               std::vector<ShapeVector>{{4, 5, 6}}, kInt32,
                                                               std::vector<ShapeVector>{{1, 2, 3, 4, 5, 6}}, kInt32}
                                        ));
}  // namespace ops
}  // namespace mindspore

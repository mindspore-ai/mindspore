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
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"
#include "ops/ops_func_impl/tuple_to_list.h"
#include "ops/ops_func_impl/list_to_tuple.h"
#include "ops/ops_frontend_func_impl.h"
#include "include/backend/optimizer/helper.h"
#include "ops/test_ops_cmp_utils.h"

namespace mindspore {
namespace ops {
struct SeqToSeqParams {
  bool is_input_tuple;
  bool is_input_ele_tensor;
  bool is_input_dyn_len;
  std::vector<ShapeVector> input_ele;
  TypePtr input_type;
};

class TestSeqToSeq : public TestOps, public testing::WithParamInterface<SeqToSeqParams> {};

void compare_tuple_to_list(AbstractBasePtrList &input_abs, bool is_input_dyn_len) {
  auto input_x = std::make_shared<abstract::AbstractTuple>(input_abs);
  auto expect = std::make_shared<abstract::AbstractList>(input_abs);
  if (is_input_dyn_len) {
    input_x->CheckAndConvertToDynamicLenSequence();
    expect->CheckAndConvertToDynamicLenSequence();
  }
  std::vector<AbstractBasePtr> input_args{input_x};
  auto expect_shape = expect->GetShape();
  auto expect_type = expect->GetType();
  auto prim = std::make_shared<Primitive>("TupleToList");
  auto infer_impl = GetOpFrontendFuncImplPtr("TupleToList");
  ASSERT_NE(infer_impl, nullptr);
  auto tuple_to_list_func_impl = infer_impl->InferAbstract(prim, input_args);
  ASSERT_NE(tuple_to_list_func_impl, nullptr);

  auto infer_shape = tuple_to_list_func_impl->GetShape();
  ASSERT_NE(infer_shape, nullptr);
  ASSERT_TRUE(*infer_shape == *expect_shape);
  auto infer_dtype = tuple_to_list_func_impl->GetType();
  ASSERT_NE(infer_dtype, nullptr);
  ASSERT_TRUE(*infer_shape == *expect_shape);
}

void compare_list_to_tuple(AbstractBasePtrList &input_abs, bool is_input_dyn_len) {
  auto input_x = std::make_shared<abstract::AbstractList>(input_abs);
  auto expect = std::make_shared<abstract::AbstractTuple>(input_abs);
  if (is_input_dyn_len) {
    input_x->CheckAndConvertToDynamicLenSequence();
    expect->CheckAndConvertToDynamicLenSequence();
  }
  std::vector<AbstractBasePtr> input_args{input_x};
  auto expect_shape = expect->GetShape();
  auto expect_type = expect->GetType();
  auto prim = std::make_shared<Primitive>("ListToTuple");
  auto infer_impl = GetOpFrontendFuncImplPtr("ListToTuple");
  ASSERT_NE(infer_impl, nullptr);
  auto list_to_tuple_func_impl = infer_impl->InferAbstract(prim, input_args);
  ASSERT_NE(list_to_tuple_func_impl, nullptr);

  auto infer_shape = list_to_tuple_func_impl->GetShape();
  ASSERT_NE(infer_shape, nullptr);
  ASSERT_TRUE(*infer_shape == *expect_shape);
  auto infer_dtype = list_to_tuple_func_impl->GetType();
  ASSERT_NE(infer_dtype, nullptr);
  ASSERT_TRUE(*infer_shape == *expect_shape);
}

TEST_P(TestSeqToSeq, seq_to_seq_dyn_shape) {
  const auto &param = GetParam();
  AbstractBasePtrList input_abs;
  if (!param.is_input_ele_tensor && !param.input_ele.empty()) {
    for (auto elem : param.input_ele[0]) {
      auto s_elem = std::make_shared<abstract::AbstractScalar>(elem);
      input_abs.push_back(s_elem);
    }
  } else {
    for (auto elem : param.input_ele) {
      auto t_elem = std::make_shared<abstract::AbstractTensor>(param.input_type, elem);
      input_abs.push_back(t_elem);
    }
  }
  if (param.is_input_tuple) {
    compare_tuple_to_list(input_abs, param.is_input_dyn_len);
  } else {
    compare_list_to_tuple(input_abs, param.is_input_dyn_len);
  }
}

INSTANTIATE_TEST_CASE_P(TestSeqToSeqGroup, TestSeqToSeq,
                        testing::Values(
                          // TupleToList
                          SeqToSeqParams{true, true, true, std::vector<ShapeVector>{}, kInt32},
                          SeqToSeqParams{true, true, false, std::vector<ShapeVector>{}, kInt32},
                          SeqToSeqParams{true, false, true, std::vector<ShapeVector>{}, kInt32},
                          SeqToSeqParams{true, false, false, std::vector<ShapeVector>{}, kInt32},
                          SeqToSeqParams{true, true, true, std::vector<ShapeVector>{{2, 3}, {2, 3}}, kInt32},
                          SeqToSeqParams{true, true, false, std::vector<ShapeVector>{{2, 3}, {2, 3}}, kInt32},
                          SeqToSeqParams{true, false, true, std::vector<ShapeVector>{{2, 3}, {2, 3}}, kInt32},
                          SeqToSeqParams{true, false, false, std::vector<ShapeVector>{{2, 3}, {2, 3}}, kInt32},
                          SeqToSeqParams{true, true, true, std::vector<ShapeVector>{{1, 2, 3}}, kInt32},
                          SeqToSeqParams{true, true, false, std::vector<ShapeVector>{{1, 2, 3}}, kInt32},
                          SeqToSeqParams{true, false, true, std::vector<ShapeVector>{{1, 2, 3}}, kInt32},
                          SeqToSeqParams{true, false, false, std::vector<ShapeVector>{{1, 2, 3}}, kInt32},
                          // ListToTuple
                          SeqToSeqParams{false, true, true, std::vector<ShapeVector>{}, kInt32},
                          SeqToSeqParams{false, true, false, std::vector<ShapeVector>{}, kInt32},
                          SeqToSeqParams{false, false, true, std::vector<ShapeVector>{}, kInt32},
                          SeqToSeqParams{false, false, false, std::vector<ShapeVector>{}, kInt32},
                          SeqToSeqParams{false, true, true, std::vector<ShapeVector>{{2, 3}, {2, 3}}, kInt32},
                          SeqToSeqParams{false, true, false, std::vector<ShapeVector>{{2, 3}, {2, 3}}, kInt32},
                          SeqToSeqParams{false, false, true, std::vector<ShapeVector>{{2, 3}, {2, 3}}, kInt32},
                          SeqToSeqParams{false, false, false, std::vector<ShapeVector>{{2, 3}, {2, 3}}, kInt32},
                          SeqToSeqParams{false, true, true, std::vector<ShapeVector>{{1, 2, 3}}, kInt32},
                          SeqToSeqParams{false, true, false, std::vector<ShapeVector>{{1, 2, 3}}, kInt32},
                          SeqToSeqParams{false, false, true, std::vector<ShapeVector>{{1, 2, 3}}, kInt32},
                          SeqToSeqParams{false, false, false, std::vector<ShapeVector>{{1, 2, 3}}, kInt32}));
}  // namespace ops
}  // namespace mindspore

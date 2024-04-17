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
#include "ops/linear_sum_assignment.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"

namespace mindspore {
namespace ops {
struct LinearSumAssignmentParams {
  ShapeVector cost_matrix_shape;
  TypePtr cost_matrix_type;
  ShapeVector dimension_limit_shape;
  TypePtr dimension_limit_type;
  ValuePtr maximize_value;
  TypePtr maximize_type;

  ShapeVector row_ind_shape;
  TypePtr row_ind_type;
  ShapeVector col_ind_shape;
  TypePtr col_ind_type;
};

class TestLinearSumAssignment : public TestOps, public testing::WithParamInterface<LinearSumAssignmentParams> {};

TEST_P(TestLinearSumAssignment, linear_sum_assignment_dyn_shape) {
  const auto &param = GetParam();
  auto cost_matrix = std::make_shared<abstract::AbstractTensor>(param.cost_matrix_type,
                                                                param.cost_matrix_shape);
  ASSERT_NE(cost_matrix, nullptr);
  auto dimension_limit = std::make_shared<abstract::AbstractTensor>(param.dimension_limit_type,
                                                                    param.dimension_limit_shape);
  ASSERT_NE(dimension_limit, nullptr);
  auto maximize = std::make_shared<abstract::AbstractScalar>(param.maximize_value,
                                                             param.maximize_type);
  ASSERT_NE(maximize, nullptr);
  auto expect_row_ind = std::make_shared<abstract::AbstractTensor>(param.row_ind_type, param.row_ind_shape);
  ASSERT_NE(expect_row_ind, nullptr);
  auto expect_col_ind = std::make_shared<abstract::AbstractTensor>(param.col_ind_type, param.col_ind_shape);
  ASSERT_NE(expect_col_ind, nullptr);

  AbstractBasePtrList abstract_list{expect_row_ind, expect_col_ind};
  auto expect = std::make_shared<abstract::AbstractTuple>(abstract_list);

  auto lsa = std::make_shared<LinearSumAssignment>();
  auto prim = std::make_shared<Primitive>(kNameLinearSumAssignment);
  auto out_abstract = LinearSumAssignmentInfer(nullptr, prim, {cost_matrix, dimension_limit, maximize});
  ASSERT_NE(out_abstract, nullptr);
  ASSERT_TRUE(*out_abstract == *expect);
}

INSTANTIATE_TEST_CASE_P(TestLinearSumAssignment, TestLinearSumAssignment,
                        testing::Values(LinearSumAssignmentParams{{3, 3}, kFloat16, {}, kInt64, MakeValue(False), kBool,
                                                                  {3}, kInt64, {3}, kInt64},
                                        LinearSumAssignmentParams{{3, 3}, kFloat32, {}, kInt64, MakeValue(False), kBool,
                                                                  {3}, kInt64, {3}, kInt64},
                                        LinearSumAssignmentParams{{3, 3}, kFloat64, {}, kInt64, MakeValue(False), kBool,
                                                                  {3}, kInt64, {3}, kInt64},
                                        LinearSumAssignmentParams{{3, 3}, kBool, {}, kInt64, MakeValue(False), kBool,
                                                                  {3}, kInt64, {3}, kInt64},
                                        LinearSumAssignmentParams{{3, 3}, kInt16, {}, kInt64, MakeValue(False), kBool,
                                                                  {3}, kInt64, {3}, kInt64},
                                        LinearSumAssignmentParams{{3, 3}, kInt32, {}, kInt64, MakeValue(False), kBool,
                                                                  {3}, kInt64, {3}, kInt64},
                                        LinearSumAssignmentParams{{3, 3}, kInt64, {}, kInt64, MakeValue(False), kBool,
                                                                  {3}, kInt64, {3}, kInt64},
                                        LinearSumAssignmentParams{{3, 3}, kInt8, {}, kInt64, MakeValue(False), kBool,
                                                                  {3}, kInt64, {3}, kInt64},
                                        LinearSumAssignmentParams{{3, 3}, kUInt16, {}, kInt64, MakeValue(False), kBool,
                                                                  {3}, kInt64, {3}, kInt64},
                                        LinearSumAssignmentParams{{3, 3}, kUInt32, {}, kInt64, MakeValue(False), kBool,
                                                                  {3}, kInt64, {3}, kInt64},
                                        LinearSumAssignmentParams{{3, 3}, kUInt64, {}, kInt64, MakeValue(False), kBool,
                                                                  {3}, kInt64, {3}, kInt64},
                                        LinearSumAssignmentParams{{3, 3}, kUInt8, {}, kInt64, MakeValue(False), kBool,
                                                                  {3}, kInt64, {3}, kInt64},
                                        LinearSumAssignmentParams{{3, 4}, kFloat16, {}, kInt64, MakeValue(False), kBool,
                                                                  {3}, kInt64, {3}, kInt64},
                                        LinearSumAssignmentParams{{3, -1}, kFloat16, {}, kInt64, MakeValue(False),
                                                                  kBool, {-1}, kInt64, {-1}, kInt64}));
}  // namespace ops
}  // namespace mindspore

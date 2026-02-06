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
#include <vector>
#include <memory>
#include "common/common_test.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "include/backend/common/pass_manager/helper.h"
#include "ops/test_ops.h"
#include "infer/ops_func_impl/group_topk.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
struct GroupTopkShapeParams {
  ShapeVector token_shape;
  TypePtr token_type;
  ShapeVector idx_arr_shape;
  TypePtr idx_arr_type;
  ValuePtr group_num;
  ValuePtr k;
  ValuePtr k_inner;
};

class TestGroupTopk : public TestOps, public testing::WithParamInterface<GroupTopkShapeParams> {};

TEST_P(TestGroupTopk, DynShape) {
  const auto &param = GetParam();
  auto token = std::make_shared<abstract::AbstractTensor>(param.token_type, param.token_shape);
  auto idx_arr = std::make_shared<abstract::AbstractTensor>(param.idx_arr_type, param.idx_arr_shape);
  auto group_num = param.group_num->ToAbstract();
  auto k = param.k->ToAbstract();
  auto k_inner = param.k_inner->ToAbstract();

  auto token_shape = std::make_shared<abstract::Shape>(param.token_shape);
  auto expect_shape = token_shape;
  auto expect_type = std::make_shared<TensorType>(param.token_type);

  GroupTopkFuncImpl func_impl;
  auto prim = std::make_shared<Primitive>("GroupTopk");

  auto out_dtype = func_impl.InferType(prim, {token, idx_arr, group_num, k, k_inner});
  ASSERT_TRUE(*out_dtype == *expect_type);
  auto out_shape = func_impl.InferShape(prim, {token, idx_arr, group_num, k, k_inner});
  ASSERT_TRUE(*out_shape == *expect_shape);
}

INSTANTIATE_TEST_CASE_P(TestGroupTopk, TestGroupTopk,
                        testing::Values(GroupTopkShapeParams{{64, 256},
                                                             kFloat16,
                                                             {1024},
                                                             kInt32,
                                                             CreateScalar<int64_t>(8),
                                                             CreateScalar<int64_t>(4),
                                                             CreateScalar<int64_t>(2)},
                                        GroupTopkShapeParams{{64, 256},
                                                             kBFloat16,
                                                             {1024},
                                                             kInt32,
                                                             CreateScalar<int64_t>(8),
                                                             CreateScalar<int64_t>(4),
                                                             CreateScalar<int64_t>(2)},
                                        GroupTopkShapeParams{{-1, 256},
                                                             kFloat16,
                                                             {1024},
                                                             kInt32,
                                                             CreateScalar<int64_t>(8),
                                                             CreateScalar<int64_t>(4),
                                                             CreateScalar<int64_t>(2)},
                                        GroupTopkShapeParams{{-1, 256},
                                                             kBFloat16,
                                                             {1024},
                                                             kInt32,
                                                             CreateScalar<int64_t>(8),
                                                             CreateScalar<int64_t>(4),
                                                             CreateScalar<int64_t>(2)}));
}  // namespace ops
}  // namespace mindspore

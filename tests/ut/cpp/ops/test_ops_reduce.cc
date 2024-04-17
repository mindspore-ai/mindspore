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
#include "ops/ops_func_impl/reduce_all.h"
#include "ops/ops_func_impl/reduce_any.h"
#include "ops/ops_func_impl/reduce_max.h"
#include "ops/ops_func_impl/reduce_min.h"
#include "ops/ops_func_impl/reduce_mean.h"
#include "ops/ops_func_impl/reduce_prod.h"
#include "ops/auto_generate/gen_ops_name.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"
#include "ops/test_value_utils.h"
#include "ops/test_ops_cmp_utils.h"

namespace mindspore {
namespace ops {
struct ReduceParams {
  ShapeVector x_shape;
  ValuePtr axis;
  ValuePtr keep_dims;
  ShapeVector out_shape;
};

static std::map<std::string, std::pair<TypePtr, OpFuncImplPtr>> reduce_func_impl = {
  {kNameReduceAll, std::make_pair(kBool, std::make_shared<ReduceAllFuncImpl>())},
  {kNameReduceAny, std::make_pair(kBool, std::make_shared<ReduceAnyFuncImpl>())},
  {kNameReduceMax, std::make_pair(kFloat32, std::make_shared<ReduceMaxFuncImpl>())},
  {kNameReduceMin, std::make_pair(kFloat32, std::make_shared<ReduceMinFuncImpl>())},
  {kNameReduceMean, std::make_pair(kFloat32, std::make_shared<ReduceMeanFuncImpl>())},
  {kNameReduceProd, std::make_pair(kFloat32, std::make_shared<ReduceProdFuncImpl>())},
};

class TestReduce : public TestOps, public testing::WithParamInterface<std::tuple<const char *, ReduceParams>> {};

TEST_P(TestReduce, dyn_shape) {
  const auto &reduce_mode = std::get<0>(GetParam());
  const auto &param = std::get<1>(GetParam());
  auto reduce_op_itr = reduce_func_impl.find(reduce_mode);
  ASSERT_TRUE(reduce_op_itr != reduce_func_impl.end());
  auto reduce_op_map = reduce_op_itr->second;
  auto type = reduce_op_map.first;
  ASSERT_NE(type, nullptr);

  auto x = std::make_shared<abstract::AbstractTensor>(type, param.x_shape);
  ASSERT_NE(x, nullptr);
  auto axis = param.axis->ToAbstract();
  ASSERT_NE(axis, nullptr);
  auto keep_dims = param.keep_dims->ToAbstract();
  ASSERT_NE(keep_dims, nullptr);
  auto prim = std::make_shared<Primitive>(reduce_mode);
  auto op_impl = reduce_op_map.second;
  ASSERT_NE(op_impl, nullptr);
  auto expect_shape = std::make_shared<abstract::Shape>(param.out_shape);
  auto expect_type = std::make_shared<TensorType>(type);
  auto inferred_shape = op_impl->InferShape(prim, {x, axis, keep_dims});
  auto inferred_type = op_impl->InferType(prim, {x, axis, keep_dims});
  ShapeCompare(inferred_shape, expect_shape);
  TypeCompare(inferred_type, expect_type);
}

namespace {
auto ReduceDynTestCase = testing::ValuesIn(
  {ReduceParams{{2, 3, 4}, CreatePyIntTuple({1}), CreateScalar(true), {2, 1, 4}},
   ReduceParams{{2, 3, 4}, CreatePyIntTuple({1}), CreateScalar(false), {2, 4}},
   ReduceParams{{2, 3, 4}, CreatePyIntTuple({0, 1}), CreateScalar(true), {1, 1, 4}},
   ReduceParams{{2, 3, 4}, CreatePyIntTuple({0, 1}), CreateScalar(false), {4}},
   ReduceParams{{2, 3, 4}, CreatePyIntTuple({-1}), CreateScalar(true), {2, 3, 1}},
   ReduceParams{{2, 3, 4}, CreatePyIntTuple({-2}), CreateScalar(false), {2, 4}},
   ReduceParams{{2, 3, 4}, CreatePyIntTuple({-1, -2}), CreateScalar(true), {2, 1, 1}},
   ReduceParams{{2, 3, 4}, CreatePyIntTuple({-2, -3}), CreateScalar(false), {4}},
   ReduceParams{{2, 3, 4}, CreatePyIntTuple({kValueAny, 1}), CreateScalar(true), {-1, 1, -1}},
   ReduceParams{{2, 3, 4}, CreatePyIntTuple({kValueAny, 1}), CreateScalar(false), {-1}},
   ReduceParams{{2, 3, 4}, CreatePyIntTuple({kValueAny, kValueAny}), CreateScalar(true), {-1, -1, -1}},
   ReduceParams{{2, 3, 4}, CreatePyIntTuple({kValueAny, kValueAny}), CreateScalar(false), {-1}},
   ReduceParams{{2, 3, 4}, kValueAny, CreateScalar(true), {-1, -1, -1}},
   ReduceParams{{2, 3, 4}, kValueAny, CreateScalar(false), {-2}},
   ReduceParams{{2, 3, 4}, CreatePyIntTuple({}), CreateScalar(false), {}},
   ReduceParams{{2, 3, 4}, CreatePyIntTuple({1}), kValueAny, {-2}},
   ReduceParams{{2, 3, 4}, CreatePyIntTuple({1, 2}), kValueAny, {-2}},
   ReduceParams{{-1, -1, 4}, CreatePyIntTuple({1}), CreateScalar(true), {-1, 1, 4}},
   ReduceParams{{-1, -1, 4}, CreatePyIntTuple({1}), CreateScalar(false), {-1, 4}},
   ReduceParams{{-1, 3, 4}, CreatePyIntTuple({0, 2}), CreateScalar(true), {1, 3, 1}},
   ReduceParams{{-1, 3, 4}, CreatePyIntTuple({0, 2}), CreateScalar(false), {3}},
   ReduceParams{{-1, -1, 4}, CreatePyIntTuple({kValueAny, 1}), CreateScalar(true), {-1, 1, -1}},
   ReduceParams{{-1, -1, 4}, CreatePyIntTuple({kValueAny, 1}), CreateScalar(false), {-1}},
   ReduceParams{{-1, -1, 4}, CreatePyIntTuple({kValueAny, kValueAny}), CreateScalar(true), {-1, -1, -1}},
   ReduceParams{{-1, -1, 4}, CreatePyIntTuple({kValueAny, kValueAny}), CreateScalar(false), {-1}},
   ReduceParams{{-1, -1, 4}, kValueAny, CreateScalar(true), {-1, -1, -1}},
   ReduceParams{{-1, -1, 4}, kValueAny, CreateScalar(false), {-2}},
   ReduceParams{{-1, -1, 4}, CreatePyIntTuple({}), CreateScalar(false), {}},
   ReduceParams{{-1, -1, 4}, CreatePyIntTuple({1}), kValueAny, {-2}},
   ReduceParams{{-1, -1, 4}, CreatePyIntTuple({1, 2}), kValueAny, {-2}},
   ReduceParams{{-2}, CreatePyIntTuple({1}), CreateScalar(true), {-2}},
   ReduceParams{{-2}, CreatePyIntTuple({0, 2}), CreateScalar(false), {-2}},
   ReduceParams{{-2}, CreatePyIntTuple({kValueAny, 1}), CreateScalar(true), {-2}},
   ReduceParams{{-2}, kValueAny, CreateScalar(true), {-2}},
   ReduceParams{{-2}, CreatePyIntTuple({}), CreateScalar(false), {}}});
}

INSTANTIATE_TEST_CASE_P(TestReduceAllGroup, TestReduce,
                        testing::Combine(testing::ValuesIn({kNameReduceAll}), ReduceDynTestCase));
INSTANTIATE_TEST_CASE_P(TestReduceAnyGroup, TestReduce,
                        testing::Combine(testing::ValuesIn({kNameReduceAny}), ReduceDynTestCase));
INSTANTIATE_TEST_CASE_P(TestReduceMaxGroup, TestReduce,
                        testing::Combine(testing::ValuesIn({kNameReduceMax}), ReduceDynTestCase));
INSTANTIATE_TEST_CASE_P(TestReduceMinGroup, TestReduce,
                        testing::Combine(testing::ValuesIn({kNameReduceMin}), ReduceDynTestCase));
INSTANTIATE_TEST_CASE_P(TestReduceMeanGroup, TestReduce,
                        testing::Combine(testing::ValuesIn({kNameReduceMean}), ReduceDynTestCase));
INSTANTIATE_TEST_CASE_P(TestReduceProdGroup, TestReduce,
                        testing::Combine(testing::ValuesIn({kNameReduceProd}), ReduceDynTestCase));
}  // namespace ops
}  // namespace mindspore

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
#include "ops/ops_func_impl/reduce_sum.h"
#include "ops/gen_ops_name.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"
#include "ops/test_value_utils.h"
#include "ops/test_ops_cmp_utils.h"

namespace mindspore {
namespace ops {
struct ReduceSumParams {
  ShapeVector x_shape;
  TypePtr x_type;
  ValuePtr axis;
  ValuePtr keep_dims;
  ValuePtr skip_mode;
  ShapeVector out_shape;
  TypePtr out_type;
};

class TestReduceSum : public TestOps, public testing::WithParamInterface<ReduceSumParams> {};

TEST_P(TestReduceSum, dyn_shape) {
  const auto &param = GetParam();
  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  ASSERT_NE(x, nullptr);
  auto axis = param.axis->ToAbstract();
  ASSERT_NE(axis, nullptr);
  auto keep_dims = param.keep_dims->ToAbstract();
  ASSERT_NE(keep_dims, nullptr);
  auto skip_mode = param.skip_mode->ToAbstract();
  ASSERT_NE(skip_mode, nullptr);

  auto expect_shape = std::make_shared<abstract::Shape>(param.out_shape);
  auto expect_type = std::make_shared<TensorType>(param.out_type);
  DoFuncImplInferAndCompare<ReduceSumFuncImpl>(kNameReduceSum, {x, axis, keep_dims, skip_mode}, expect_shape,
                                               expect_type);
}

INSTANTIATE_TEST_CASE_P(
  TestReduceSumGroup, TestReduceSum,
  testing::Values(
    ReduceSumParams{
      {2, 3, 4}, kFloat32, CreatePyIntTuple({1}), CreateScalar(true), CreateScalar(false), {2, 1, 4}, kFloat32},
    ReduceSumParams{
      {2, 3, 4}, kFloat32, CreatePyIntTuple({1}), CreateScalar(false), CreateScalar(false), {2, 4}, kFloat32},
    ReduceSumParams{
      {2, 3, 4}, kFloat32, CreatePyIntTuple({0, 1}), CreateScalar(true), CreateScalar(false), {1, 1, 4}, kFloat32},
    ReduceSumParams{
      {2, 3, 4}, kFloat32, CreatePyIntTuple({0, 1}), CreateScalar(false), CreateScalar(false), {4}, kFloat32},
    ReduceSumParams{{2, 3, 4},
                    kFloat32,
                    CreatePyIntTuple({kValueAny, 1}),
                    CreateScalar(true),
                    CreateScalar(false),
                    {-1, 1, -1},
                    kFloat32},
    ReduceSumParams{
      {2, 3, 4}, kFloat32, CreatePyIntTuple({kValueAny, 1}), CreateScalar(false), CreateScalar(true), {-1}, kFloat32},
    ReduceSumParams{{2, 3, 4},
                    kFloat32,
                    CreatePyIntTuple({kValueAny, kValueAny}),
                    CreateScalar(true),
                    CreateScalar(false),
                    {-1, -1, -1},
                    kFloat32},
    ReduceSumParams{{2, 3, 4},
                    kFloat32,
                    CreatePyIntTuple({kValueAny, kValueAny}),
                    CreateScalar(false),
                    CreateScalar(true),
                    {-1},
                    kFloat32},
    ReduceSumParams{{2, 3, 4}, kFloat32, kValueAny, CreateScalar(true), CreateScalar(false), {-1, -1, -1}, kFloat32},
    ReduceSumParams{{2, 3, 4}, kFloat32, kValueAny, CreateScalar(false), CreateScalar(false), {-2}, kFloat32},
    ReduceSumParams{{2, 3, 4}, kFloat32, CreatePyIntTuple({}), CreateScalar(false), CreateScalar(false), {}, kFloat32},
    ReduceSumParams{{2, 3, 4}, kFloat32, CreatePyIntTuple({1}), kValueAny, CreateScalar(false), {-2}, kFloat32},
    ReduceSumParams{{2, 3, 4}, kFloat32, CreatePyIntTuple({1, 2}), kValueAny, CreateScalar(false), {-2}, kFloat32},
    ReduceSumParams{
      {-1, -1, 4}, kFloat32, CreatePyIntTuple({1}), CreateScalar(true), CreateScalar(true), {-1, 1, 4}, kFloat32},
    ReduceSumParams{
      {-1, -1, 4}, kFloat32, CreatePyIntTuple({1}), CreateScalar(false), CreateScalar(false), {-1, 4}, kFloat32},
    ReduceSumParams{
      {-1, 3, 4}, kFloat32, CreatePyIntTuple({0, 2}), CreateScalar(true), CreateScalar(false), {1, 3, 1}, kFloat32},
    ReduceSumParams{
      {-1, 3, 4}, kFloat32, CreatePyIntTuple({0, 2}), CreateScalar(false), CreateScalar(false), {3}, kFloat32},
    ReduceSumParams{{-1, -1, 4},
                    kFloat32,
                    CreatePyIntTuple({kValueAny, 1}),
                    CreateScalar(true),
                    CreateScalar(false),
                    {-1, 1, -1},
                    kFloat32},
    ReduceSumParams{{-1, -1, 4},
                    kFloat32,
                    CreatePyIntTuple({kValueAny, 1}),
                    CreateScalar(false),
                    CreateScalar(false),
                    {-1},
                    kFloat32},
    ReduceSumParams{{-1, -1, 4},
                    kFloat32,
                    CreatePyIntTuple({kValueAny, kValueAny}),
                    CreateScalar(true),
                    CreateScalar(false),
                    {-1, -1, -1},
                    kFloat32},
    ReduceSumParams{{-1, -1, 4},
                    kFloat32,
                    CreatePyIntTuple({kValueAny, kValueAny}),
                    CreateScalar(false),
                    CreateScalar(false),
                    {-1},
                    kFloat32},
    ReduceSumParams{{-1, -1, 4}, kFloat32, kValueAny, CreateScalar(true), CreateScalar(false), {-1, -1, -1}, kFloat32},
    ReduceSumParams{{-1, -1, 4}, kFloat32, kValueAny, CreateScalar(false), CreateScalar(false), {-2}, kFloat32},
    ReduceSumParams{
      {-1, -1, 4}, kFloat32, CreatePyIntTuple({}), CreateScalar(false), CreateScalar(false), {}, kFloat32},
    ReduceSumParams{{-1, -1, 4}, kFloat32, CreatePyIntTuple({1}), kValueAny, CreateScalar(true), {-2}, kFloat32},
    ReduceSumParams{{-1, -1, 4}, kFloat32, CreatePyIntTuple({1, 2}), kValueAny, CreateScalar(false), {-2}, kFloat32},
    ReduceSumParams{{-2}, kFloat32, CreatePyIntTuple({1}), CreateScalar(true), CreateScalar(false), {-2}, kFloat32},
    ReduceSumParams{{-2}, kFloat32, CreatePyIntTuple({0, 2}), CreateScalar(false), CreateScalar(false), {-2}, kFloat32},
    ReduceSumParams{
      {-2}, kFloat32, CreatePyIntTuple({kValueAny, 1}), CreateScalar(true), CreateScalar(true), {-2}, kFloat32},
    ReduceSumParams{{-2}, kFloat32, kValueAny, CreateScalar(true), CreateScalar(false), {-2}, kFloat32},
    ReduceSumParams{{-2}, kFloat32, CreatePyIntTuple({}), CreateScalar(false), CreateScalar(false), {}, kFloat32},
    ReduceSumParams{
      {2, 3, 4}, kFloat32, CreatePyIntTuple({}), CreateScalar(true), CreateScalar(true), {2, 3, 4}, kFloat32},
    ReduceSumParams{
      {2, 3, 4}, kFloat32, CreatePyIntTuple({}), CreateScalar(false), CreateScalar(true), {2, 3, 4}, kFloat32},
    ReduceSumParams{
      {-1, -1, 4}, kFloat32, CreatePyIntTuple({}), CreateScalar(true), CreateScalar(true), {-1, -1, 4}, kFloat32},
    ReduceSumParams{
      {-1, -1, 4}, kFloat32, CreatePyIntTuple({}), CreateScalar(false), CreateScalar(true), {-1, -1, 4}, kFloat32},
    ReduceSumParams{{-2}, kFloat32, CreatePyIntTuple({}), CreateScalar(true), CreateScalar(true), {-2}, kFloat32},
    ReduceSumParams{{-2}, kFloat32, CreatePyIntTuple({}), CreateScalar(false), CreateScalar(true), {-2}, kFloat32},
    ReduceSumParams{{2, 3, 4}, kFloat32, CreatePyIntTuple({}), CreateScalar(true), kValueAny, {-2}, kFloat32}));
}  // namespace ops
}  // namespace mindspore

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
#include "ops/test_ops_cmp_utils.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"
#include "ops/ops_func_impl/split.h"
#include "ops/test_value_utils.h"

// #include "ops/test_ops_dyn_cases.h"
// #include "include/backend/optimizer/helper.h"

namespace mindspore {
namespace ops {
struct SplitParams {
  ShapeVector x_shape;
  TypePtr x_type;
  ValuePtr axis;
  ValuePtr output_num;  // can not be kValueAny.
  std::vector<ShapeVector> out_shape;
  std::vector<TypePtr> out_type;
};
class TestSplit : public TestOps, public testing::WithParamInterface<SplitParams> {};

TEST_P(TestSplit, split_dyn_shape) {
  auto primitive = std::make_shared<Primitive>("Split");
  ASSERT_NE(primitive, nullptr);
  const auto &param = GetParam();
  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  ASSERT_NE(x, nullptr);
  std::vector<abstract::AbstractBasePtr> input_args{std::move(x), std::move(param.axis->ToAbstract()),
                                                    std::move(param.output_num->ToAbstract())};
  auto infer_impl = std::make_shared<SplitFuncImpl>();
  ASSERT_NE(infer_impl, nullptr);
  auto infer_shape = infer_impl->InferShape(primitive, input_args);
  ASSERT_NE(infer_shape, nullptr);
  auto infer_type = infer_impl->InferType(primitive, input_args);
  ASSERT_NE(infer_type, nullptr);

  auto expect = MakeOutputTupleShapeAndType(param.out_shape, param.out_type);
  auto expect_shape = expect.first;
  ASSERT_NE(expect_shape, nullptr);
  auto expect_type = expect.second;
  ASSERT_NE(expect_type, nullptr);
  ShapeCompare(infer_shape, expect_shape);
  TypeCompare(infer_type, expect_type);
};

INSTANTIATE_TEST_CASE_P(
  TestSplitGroup, TestSplit,
  testing::Values(
    SplitParams{{2, 2}, kFloat32, CreateScalar<int64_t>(1), CreateScalar<int64_t>(1), {{2, 2}}, {kFloat32}},
    SplitParams{
      {2, 2}, kFloat32, CreateScalar<int64_t>(1), CreateScalar<int64_t>(2), {{2, 1}, {2, 1}}, {kFloat32, kFloat32}},
    SplitParams{
      {-1, 2}, kFloat32, CreateScalar<int64_t>(1), CreateScalar<int64_t>(2), {{-1, 1}, {-1, 1}}, {kFloat32, kFloat32}},
    SplitParams{
      {2, -1}, kFloat32, CreateScalar<int64_t>(1), CreateScalar<int64_t>(2), {{2, -1}, {2, -1}}, {kFloat32, kFloat32}},
    SplitParams{{-1, -1},
                kFloat32,
                CreateScalar<int64_t>(1),
                CreateScalar<int64_t>(2),
                {{-1, -1}, {-1, -1}},
                {kFloat32, kFloat32}},
    SplitParams{{-2}, kFloat32, CreateScalar<int64_t>(1), CreateScalar<int64_t>(2), {{-2}, {-2}}, {kFloat32, kFloat32}},
    SplitParams{
      {2, 2}, kFloat32, CreateScalar(kValueAny), CreateScalar<int64_t>(2), {{-1, -1}, {-1, -1}}, {kFloat32, kFloat32}},
    SplitParams{
      {-1, 2}, kFloat32, CreateScalar(kValueAny), CreateScalar<int64_t>(2), {{-1, -1}, {-1, -1}}, {kFloat32, kFloat32}},
    SplitParams{
      {2, -1}, kFloat32, CreateScalar(kValueAny), CreateScalar<int64_t>(2), {{-1, -1}, {-1, -1}}, {kFloat32, kFloat32}},
    SplitParams{{-1, -1},
                kFloat32,
                CreateScalar(kValueAny),
                CreateScalar<int64_t>(2),
                {{-1, -1}, {-1, -1}},
                {kFloat32, kFloat32}},
    SplitParams{
      {-2}, kFloat32, CreateScalar(kValueAny), CreateScalar<int64_t>(2), {{-2}, {-2}}, {kFloat32, kFloat32}}));
}  // namespace ops
}  // namespace mindspore

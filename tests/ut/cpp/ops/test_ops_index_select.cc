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
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "common/common_test.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "ops/ops_func_impl/index_select.h"
#include "ops/test_ops.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
struct IndexSelectOpParams {
  ShapeVector input_shape;
  TypePtr input_type;
  ValuePtr axis;
  ShapeVector index_shape;
  TypePtr index_type;
  ShapeVector output_shape;
  TypePtr output_type;
};

class TestIndexSelect : public TestOps, public testing::WithParamInterface<IndexSelectOpParams> {};

TEST_P(TestIndexSelect, dyn_shape) {
  auto primitive = std::make_shared<Primitive>("IndexSelect");
  ASSERT_NE(primitive, nullptr);
  const auto &param = GetParam();
  auto input = std::make_shared<abstract::AbstractTensor>(param.input_type, param.input_shape);
  auto index = std::make_shared<abstract::AbstractTensor>(param.index_type, param.index_shape);
  ASSERT_NE(input, nullptr);
  ASSERT_NE(index, nullptr);
  auto axis = param.axis->ToAbstract();
  ASSERT_NE(axis, nullptr);

  std::vector<abstract::AbstractBasePtr> input_args{std::move(input), std::move(axis), std::move(index)};
  auto infer_impl = std::make_shared<IndexSelectFuncImpl>();
  ASSERT_NE(infer_impl, nullptr);
  auto infer_shape = infer_impl->InferShape(primitive, input_args);
  ASSERT_NE(infer_shape, nullptr);
  auto infer_type = infer_impl->InferType(primitive, input_args);
  ASSERT_NE(infer_type, nullptr);

  auto expect_shape = std::make_shared<abstract::TensorShape>(param.output_shape);
  auto expect_type = std::make_shared<TensorType>(param.output_type);
  ASSERT_NE(expect_shape, nullptr);
  ASSERT_NE(expect_type, nullptr);

  ShapeCompare(infer_shape, expect_shape);
  TypeCompare(infer_type, expect_type);
}

INSTANTIATE_TEST_CASE_P(TestIndexSelectGroup, TestIndexSelect,
  testing::Values(
    IndexSelectOpParams{{2, 2, 3}, kFloat32, CreateScalar<int64_t>(-1), {4}, kInt64, {2, 2, 4}, kFloat32},
    IndexSelectOpParams{{2, 2, -1}, kFloat32, CreateScalar<int64_t>(-1), {4}, kInt64, {2, 2, 4}, kFloat32},
    IndexSelectOpParams{{-1, -1, -1}, kFloat32, CreateScalar<int64_t>(-1), {4}, kInt64, {-1, -1, 4}, kFloat32},
    IndexSelectOpParams{{-2}, kFloat32, CreateScalar<int64_t>(-1), {4}, kInt64, {-2}, kFloat32},
    IndexSelectOpParams{{2, 2, 3}, kFloat32, CreateScalar<int64_t>(-1), {-1}, kInt64, {2, 2, -1}, kFloat32},
    IndexSelectOpParams{{2, 2, 3}, kFloat32, CreateScalar<int64_t>(-1), {-2}, kInt64, {2, 2, -1}, kFloat32},
    IndexSelectOpParams{{-2}, kFloat32, CreateScalar<int64_t>(-1), {-1}, kInt64, {-2}, kFloat32},
    IndexSelectOpParams{{2, 2, 3}, kFloat32, kValueAny, {4}, kInt64, {-1, -1, -1}, kFloat32},
    IndexSelectOpParams{{-1, -1, -1}, kFloat32, kValueAny, {4}, kInt64, {-1, -1, -1}, kFloat32},
    IndexSelectOpParams{{-2}, kFloat32, kValueAny, {4}, kInt64, {-2}, kFloat32},
    IndexSelectOpParams{{2, 2, 3}, kFloat32, kValueAny, {-1}, kInt64, {-1, -1, -1}, kFloat32},
    IndexSelectOpParams{{2, 2, 3}, kFloat32, kValueAny, {-2}, kInt64, {-1, -1, -1}, kFloat32}
  ));
}  // namespace ops
}  // namespace mindspore

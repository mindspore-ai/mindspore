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
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "common/common_test.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "ops/ops_func_impl/one_hot.h"
#include "ops/test_ops.h"
#include "test_value_utils.h"

namespace mindspore {
namespace ops {
struct OneHotOpParams {
  ShapeVector indices_shape;
  TypePtr indices_type;
  ValuePtr depth;
  ShapeVector on_value_shape;
  TypePtr on_value_type;
  ShapeVector off_value_shape;
  TypePtr off_value_type;
  ValuePtr axis;
  ShapeVector out_shape;
  TypePtr out_type;
};
class TestOneHot : public TestOps, public testing::WithParamInterface<OneHotOpParams> {};

TEST_P(TestOneHot, one_hot_dyn_shape) {
  auto primitive = std::make_shared<Primitive>("OneHot");
  ASSERT_NE(primitive, nullptr);
  const auto &param = GetParam();
  auto indices = std::make_shared<abstract::AbstractTensor>(param.indices_type, param.indices_shape);
  ASSERT_NE(indices, nullptr);
  std::vector<abstract::AbstractBasePtr> input_args{std::move(indices)};
  auto depth = param.depth->ToAbstract();
  ASSERT_NE(depth, nullptr);
  input_args.push_back(std::move(depth));
  auto on_value = std::make_shared<abstract::AbstractTensor>(param.on_value_type, param.on_value_shape);
  ASSERT_NE(on_value, nullptr);
  input_args.push_back(std::move(on_value));
  auto off_value = std::make_shared<abstract::AbstractTensor>(param.off_value_type, param.off_value_shape);
  ASSERT_NE(off_value, nullptr);
  input_args.push_back(std::move(off_value));
  auto axis = param.axis->ToAbstract();
  ASSERT_NE(axis, nullptr);
  input_args.push_back(std::move(axis));

  auto infer_impl = std::make_shared<OneHotFuncImpl>();
  ASSERT_NE(infer_impl, nullptr);
  auto infer_shape = infer_impl->InferShape(primitive, input_args);
  ASSERT_NE(infer_shape, nullptr);
  auto infer_type = infer_impl->InferType(primitive, input_args);
  ASSERT_NE(infer_type, nullptr);

  auto expect_shape = std::make_shared<abstract::Shape>(param.out_shape);
  ASSERT_NE(expect_shape, nullptr);
  auto expect_type = std::make_shared<TensorType>(param.out_type);
  ASSERT_NE(expect_type, nullptr);
  ASSERT_TRUE(*infer_shape == *expect_shape);
  ASSERT_TRUE(*infer_type == *expect_type);
}

INSTANTIATE_TEST_CASE_P(
  TestOneHotGroup, TestOneHot,
  testing::Values(
    OneHotOpParams{{2, 2, 3}, kInt32, CreateScalar<int64_t>(10), {1}, kInt32, {0}, kInt32,
                   CreateScalar<int64_t>(1), {2, 10, 2, 3}, kInt32},
    OneHotOpParams{{2, 2, 3}, kInt32, CreateScalar<int64_t>(10), {1}, kInt32, {0}, kInt32,
                   CreateScalar<int64_t>(-1), {2, 2, 3, 10}, kInt32},
    OneHotOpParams{{-2}, kInt32, CreateScalar(kValueAny), {1}, kInt32, {0}, kInt32,
                   CreateScalar(kValueAny), {-2}, kInt32},
    OneHotOpParams{{2, 2, 3}, kInt32, CreateScalar(kValueAny), {1}, kInt32, {0}, kInt32,
                   CreateScalar<int64_t>(-1), {2, 2, 3, -1}, kInt32},
    OneHotOpParams{{2, 2, 3}, kInt32, CreateScalar<int64_t>(4), {1}, kInt32, {0}, kInt32,
                   CreateScalar(kValueAny), {-1, -1, -1, -1}, kInt32},
    OneHotOpParams{{2, 2, 3}, kInt32, CreateScalar(kValueAny), {1}, kInt32, {0}, kInt32,
                   CreateScalar(kValueAny), {-1, -1, -1, -1}, kInt32},
    OneHotOpParams{{2, 2, -1}, kInt32, CreateScalar(kValueAny), {1}, kInt32, {0}, kInt32,
                   CreateScalar<int64_t>(-1), {2, 2, -1, -1}, kInt32},
    OneHotOpParams{{2, 2, -1}, kInt32, CreateScalar<int64_t>(4), {1}, kInt32, {0}, kInt32,
                   CreateScalar(kValueAny), {-1, -1, -1, -1}, kInt32},
    OneHotOpParams{{2, 2, -1}, kInt32, CreateScalar(kValueAny), {1}, kInt32, {0}, kInt32,
                   CreateScalar(kValueAny), {-1, -1, -1, -1}, kInt32}));
}  // namespace ops
}  // namespace mindspore

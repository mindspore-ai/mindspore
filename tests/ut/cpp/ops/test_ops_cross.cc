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
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"
#include "ops/ops_func_impl/cross.h"
#include "ops/test_value_utils.h"
#include "ops/test_ops_cmp_utils.h"

namespace mindspore {
namespace ops {
struct CrossShapeParams {
  ShapeVector input_shape;
  TypePtr input_type;
  ShapeVector other_shape;
  TypePtr other_type;
  ValuePtr dim;
};

class TestCross : public TestOps, public testing::WithParamInterface<CrossShapeParams> {};
class TestCrossException : public TestOps, public testing::WithParamInterface<CrossShapeParams> {};
class TestCrossSimpleInfer : public TestOps, public testing::WithParamInterface<CrossShapeParams> {};

TEST_P(TestCross, dyn_shape) {
  const auto &param = GetParam();
  auto input = std::make_shared<abstract::AbstractTensor>(param.input_type, param.input_shape);
  auto other = std::make_shared<abstract::AbstractTensor>(param.other_type, param.other_shape);
  auto expect_shape = input->GetShape();
  auto expect_type = input->GetType();
  AbstractBasePtr dim;
  if (param.dim == nullptr) {
    dim = std::make_shared<abstract::AbstractNone>();
  } else {
    dim = param.dim->ToAbstract();
  }
  CrossFuncImpl cross_func_impl;
  auto prim = std::make_shared<Primitive>("Cross");

  auto out_dtype = cross_func_impl.InferType(prim, {input, other, dim});
  ASSERT_TRUE(*out_dtype == *expect_type);
  auto out_shape = cross_func_impl.InferShape(prim, {input, other, dim});
  ASSERT_TRUE(*out_shape == *expect_shape);
}

TEST_P(TestCrossException, exception) {
  const auto &param = GetParam();
  auto input = std::make_shared<abstract::AbstractTensor>(param.input_type, param.input_shape);
  auto other = std::make_shared<abstract::AbstractTensor>(param.other_type, param.other_shape);
  auto expect_shape = input->GetShape();
  auto expect_type = input->GetType();
  AbstractBasePtr dim;
  if (param.dim == nullptr) {
    dim = std::make_shared<abstract::AbstractNone>();
  } else {
    dim = param.dim->ToAbstract();
  }
  CrossFuncImpl cross_func_impl;
  auto prim = std::make_shared<Primitive>("Cross");

  try {
    auto out_dtype = cross_func_impl.InferType(prim, {input, other, dim});
    auto out_shape = cross_func_impl.InferShape(prim, {input, other, dim});
  } catch (std::exception &e) {
    ASSERT_TRUE(true);
    return;
  }
  ASSERT_TRUE(false);
}

TEST_P(TestCrossSimpleInfer, simple_infer) {
  const auto &param = GetParam();
  auto input = std::make_shared<tensor::BaseTensor>(param.input_type->type_id(), param.input_shape);
  auto other = std::make_shared<tensor::BaseTensor>(param.other_type->type_id(), param.other_shape);
  auto expect_shape = ShapeArray{param.input_shape};
  auto expect_type = TypePtrList{param.input_type};
  CrossFuncImpl cross_func_impl;
  auto prim = std::make_shared<Primitive>("Cross");
  ValuePtrList input_values;
  input_values.push_back(std::move(input));
  input_values.push_back(std::move(other));
  input_values.push_back(std::move(param.dim));
  auto out_dtype = cross_func_impl.InferType(prim, input_values);
  TypeCompare(out_dtype, expect_type);
  auto out_shape = cross_func_impl.InferShape(prim, input_values);
  ShapeCompare(out_shape, expect_shape);
}

INSTANTIATE_TEST_CASE_P(
  TestCross, TestCross,
  testing::Values(CrossShapeParams{{4, 3}, kInt8, {4, 3}, kInt8, CreateScalar<int64_t>(1)},
                  CrossShapeParams{{2, 3}, kInt16, {2, 3}, kInt16, CreateScalar<int64_t>(1)},
                  CrossShapeParams{{3, 4, 5}, kInt32, {3, 4, 5}, kInt32, CreateScalar<int64_t>(2)},
                  CrossShapeParams{{2, 3}, kInt64, {2, 3}, kInt64, CreateScalar<int64_t>(-65530)},
                  CrossShapeParams{{3, 4, 5}, kUInt8, {3, 4, 5}, kUInt8, CreateScalar<int64_t>(-65530)},
                  CrossShapeParams{{3, 4, 5}, kFloat16, {3, 4, 5}, kFloat16, CreateScalar<int64_t>(-1)},
                  CrossShapeParams{{3, 4, 5}, kFloat32, {3, 4, 5}, kFloat32, CreateScalar<int64_t>(1)},
                  CrossShapeParams{{5, 4, 3}, kFloat64, {5, 4, 3}, kFloat64, CreateScalar<int64_t>(-1)},
                  CrossShapeParams{{-1, -1, -1}, kComplex64, {-1, -1, -1}, kComplex64, CreateScalar<int64_t>(0)},
                  CrossShapeParams{{-2}, kComplex128, {-2}, kComplex128, CreateScalar<int64_t>(0)}));

INSTANTIATE_TEST_CASE_P(
  TestCrossException, TestCrossException,
  testing::Values(CrossShapeParams{{4, 3}, kInt8, {3, 4}, kUInt8, CreateScalar<int64_t>(1)},
                  CrossShapeParams{{4, 3}, kInt8, {3, 4}, kInt16, CreateScalar<int64_t>(1)},
                  CrossShapeParams{{2, 3, 4}, kInt16, {2, 3}, kInt16, CreateScalar<int64_t>(1)},
                  CrossShapeParams{{3, 4, 5}, kInt32, {4, 3, 5}, kInt32, CreateScalar<int64_t>(2)},
                  CrossShapeParams{{2, 4}, kInt64, {2, 4}, kInt64, CreateScalar<int64_t>(-65530)},
                  CrossShapeParams{{3, 4, 5}, kUInt8, {3, 4, 5}, kUInt8, CreateScalar<int64_t>(3)},
                  CrossShapeParams{{2, 4, 5}, kUInt8, {2, 4, 5}, kUInt8, CreateScalar<int64_t>(-65530)},
                  CrossShapeParams{{3, 4, 5}, kUInt8, {3, 4, 5}, kUInt8, CreateScalar<int64_t>(-4)}));

INSTANTIATE_TEST_CASE_P(
  TestCrossSimpleInfer, TestCrossSimpleInfer,
  testing::Values(CrossShapeParams{{4, 3}, kInt8, {4, 3}, kInt8, CreateScalar<int64_t>(1)},
                  CrossShapeParams{{2, 3}, kInt16, {2, 3}, kInt16, CreateScalar<int64_t>(1)},
                  CrossShapeParams{{3, 4, 5}, kInt32, {3, 4, 5}, kInt32, CreateScalar<int64_t>(2)},
                  CrossShapeParams{{2, 3}, kInt64, {2, 3}, kInt64, CreateScalar<int64_t>(-65530)},
                  CrossShapeParams{{3, 4, 5}, kUInt8, {3, 4, 5}, kUInt8, CreateScalar<int64_t>(-65530)},
                  CrossShapeParams{{3, 4, 5}, kFloat16, {3, 4, 5}, kFloat16, CreateScalar<int64_t>(-1)},
                  CrossShapeParams{{3, 4, 5}, kFloat32, {3, 4, 5}, kFloat32, CreateScalar<int64_t>(1)},
                  CrossShapeParams{{5, 4, 3}, kFloat64, {5, 4, 3}, kFloat64, CreateScalar<int64_t>(-1)},
                  CrossShapeParams{{-1, -1, -1}, kComplex64, {-1, -1, -1}, kComplex64, CreateScalar<int64_t>(0)},
                  CrossShapeParams{{-2}, kComplex128, {-2}, kComplex128, CreateScalar<int64_t>(0)}));
}  // namespace ops
}  // namespace mindspore
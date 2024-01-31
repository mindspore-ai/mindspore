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

#include "ops/test_ops.h"
#include "common/common_test.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "ops/test_value_utils.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "ops/op_name.h"
#include "ops/ops_func_impl/cast.h"
#include "include/backend/optimizer/helper.h"

namespace mindspore {
namespace ops {
struct CastParams {
  ShapeVector x_shape;
  TypePtr x_dtype;
  ValuePtr dtype;
  ShapeVector output_shape;
  TypePtr output_type;
};

class TestCast : public TestOps, public testing::WithParamInterface<CastParams> {};

TEST_P(TestCast, dyn_shape) {
  const auto &param = GetParam();
  auto cast_func_impl = std::make_shared<CastFuncImpl>();
  auto prim = std::make_shared<Primitive>("Cast");

  auto x = std::make_shared<abstract::AbstractTensor>(param.x_dtype, param.x_shape);
  ASSERT_NE(x, nullptr);
  auto dtype = param.dtype->ToAbstract();
  auto expect = std::make_shared<abstract::AbstractTensor>(param.output_type, param.output_shape);

  auto out_dtype = cast_func_impl->InferType(prim, {x, dtype});
  ASSERT_TRUE(*out_dtype == *expect->GetType());
  auto out_shape = cast_func_impl->InferShape(prim, {x, dtype});
  ASSERT_TRUE(*out_shape == *expect->GetShape());
}

INSTANTIATE_TEST_CASE_P(
  TestCast, TestCast,
  testing::Values(
    CastParams{{6, 8}, kInt32, CreatePyInt(kNumberTypeFloat64), {6, 8}, kFloat64},
    CastParams{{-1, 8}, kFloat64,  CreatePyInt(kNumberTypeFloat16), {-1, 8}, kFloat16},
    CastParams{{6, -1}, kFloat32, CreatePyInt(kNumberTypeComplex128), {6, -1}, kComplex128},
    CastParams{{-1, -1}, kFloat64, CreatePyInt(kNumberTypeInt8), {-1, -1}, kInt8}));
}  // namespace ops
}  // namespace mindspore

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

#include "common/common_test.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"
#include "ops/test_value_utils.h"
#include "ops/ops_func_impl/isclose.h"

namespace mindspore {
namespace ops {
struct IsCloseShapeParams {
  ShapeVector input_shape;
  TypePtr input_type;
  ShapeVector other_shape;
  TypePtr other_type;
  ValuePtr rtol;
  ValuePtr atol;
  ValuePtr equal_nan;
  ShapeVector output_shape;
  TypePtr output_type;
};

class TestIsClose : public TestOps, public testing::WithParamInterface<IsCloseShapeParams> {};

TEST_P(TestIsClose, dyn_shape) {
  const auto &param = GetParam();
  auto input = std::make_shared<abstract::AbstractTensor>(param.input_type, param.input_shape);
  auto other = std::make_shared<abstract::AbstractTensor>(param.other_type, param.other_shape);
  auto rtol = param.rtol->ToAbstract();
  auto atol = param.atol->ToAbstract();
  auto equal_nan = param.equal_nan->ToAbstract();

  auto expect_shape = std::make_shared<abstract::TensorShape>(param.output_shape);
  auto expect_type = std::make_shared<TensorType>(param.output_type);

  IsCloseFuncImpl isclose_func_impl;
  auto prim = std::make_shared<Primitive>("IsClose");
  auto out_shape = isclose_func_impl.InferShape(prim, {input, other, rtol, atol, equal_nan});
  ASSERT_TRUE(*out_shape == *expect_shape);
  auto out_dtype = isclose_func_impl.InferType(prim, {input, other, rtol, atol, equal_nan});
  ASSERT_TRUE(*out_dtype == *expect_type);
}

INSTANTIATE_TEST_CASE_P(TestIsClose, TestIsClose,
                        testing::Values(IsCloseShapeParams{{3, 4},
                                                           kFloat32,
                                                           {},
                                                           kFloat32,
                                                           CreateScalar<double>(1e-5f),
                                                           CreateScalar<double>(1e-8f),
                                                           CreateScalar<bool>(false),
                                                           {3, 4},
                                                           kBool},
                                        IsCloseShapeParams{{3, 4, 1},
                                                           kFloat16,
                                                           {3, 4, 5},
                                                           kFloat16,
                                                           CreateScalar<double>(1e-5f),
                                                           CreateScalar<double>(1e-8f),
                                                           CreateScalar<bool>(false),
                                                           {3, 4, 5},
                                                           kBool},
                                        IsCloseShapeParams{{},
                                                           kInt32,
                                                           {-2},
                                                           kInt32,
                                                           CreateScalar<double>(1e-5f),
                                                           CreateScalar<double>(1e-8f),
                                                           CreateScalar<bool>(false),
                                                           {-2},
                                                           kBool}));

}  // namespace ops
}  // namespace mindspore

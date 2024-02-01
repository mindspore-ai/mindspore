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
#include "ops/ops_func_impl/ifftshift.h"
#include "ops/auto_generate/gen_ops_name.h"
#include "ops/op_name.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
struct IFFTShiftShape {
  ShapeVector x_shape;
  ValuePtr dim;
  ShapeVector out_shape;
};

struct IFFTShiftType {
  TypePtr x_type;
  TypePtr out_type;
};

class TestIFFTShift : public TestOps, public testing::WithParamInterface<std::tuple<IFFTShiftShape, IFFTShiftType>> {};

TEST_P(TestIFFTShift, dyn_shape) {
  // prepare
  const auto &shape_param = std::get<0>(GetParam());
  const auto &type_param = std::get<1>(GetParam());

  // input
  IFFTShiftFuncImpl ifftshift_func_impl;
  auto primitive = std::make_shared<Primitive>("IFFTShift");
  ASSERT_NE(primitive, nullptr);
  auto x = std::make_shared<abstract::AbstractTensor>(type_param.x_type, shape_param.x_shape);
  ASSERT_NE(x, nullptr);
  auto dim = shape_param.dim->ToAbstract();
  std::vector<AbstractBasePtr> input_args = {x, dim};

  // expect output
  auto expect_shape = std::make_shared<abstract::Shape>(shape_param.out_shape);
  ASSERT_NE(expect_shape, nullptr);
  auto expect_dtype = std::make_shared<TensorType>(type_param.out_type);
  ASSERT_NE(expect_dtype, nullptr);

  // execute
  auto out_shape = ifftshift_func_impl.InferShape(primitive, input_args);
  auto out_dtype = ifftshift_func_impl.InferType(primitive, input_args);

  // verify output
  ASSERT_NE(out_shape, nullptr);
  ASSERT_TRUE(*out_shape == *expect_shape);
  ASSERT_NE(out_dtype, nullptr);
  ASSERT_TRUE(*out_dtype == *expect_dtype);
}

auto ifftshift_shape_cases = testing::Values(IFFTShiftShape{{5, 5}, CreateTuple({0, 1}), {5, 5}},
                                             IFFTShiftShape{{-1, -1, -1}, CreateTuple({0, 1}), {-1, -1, -1}},
                                             IFFTShiftShape{{-2}, CreateTuple({0, 1}), {-2}});

auto ifftshift_type_cases = testing::ValuesIn({
  IFFTShiftType{kBool, kBool},
  IFFTShiftType{kUInt8, kUInt8},
  IFFTShiftType{kUInt16, kUInt16},
  IFFTShiftType{kUInt32, kUInt32},
  IFFTShiftType{kUInt64, kUInt64},
  IFFTShiftType{kInt8, kInt8},
  IFFTShiftType{kInt16, kInt16},
  IFFTShiftType{kInt32, kInt32},
  IFFTShiftType{kInt64, kInt64},
  IFFTShiftType{kFloat16, kFloat16},
  IFFTShiftType{kFloat32, kFloat32},
  IFFTShiftType{kFloat64, kFloat64},
  IFFTShiftType{kComplex64, kComplex64},
  IFFTShiftType{kComplex128, kComplex128},
});

INSTANTIATE_TEST_CASE_P(TestIFFTShiftGroup, TestIFFTShift,
                        testing::Combine(ifftshift_shape_cases, ifftshift_type_cases));
}  // namespace ops
}  // namespace mindspore

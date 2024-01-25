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
#include "ops/ops_func_impl/fftshift.h"
#include "ops/auto_generate/gen_ops_name.h"
#include "ops/op_name.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
struct FFTShiftShape {
  ShapeVector x_shape;
  ValuePtr axes;
  ValuePtr forward;
  ShapeVector out_shape;
};

struct FFTShiftType {
  TypePtr x_type;
  TypePtr out_type;
};

class TestFFTShift : public TestOps, public testing::WithParamInterface<std::tuple<FFTShiftShape, FFTShiftType>> {};

TEST_P(TestFFTShift, dyn_shape) {
  // prepare
  const auto &shape_param = std::get<0>(GetParam());
  const auto &type_param = std::get<1>(GetParam());

  // input
  FFTShiftFuncImpl fftshift_func_impl;
  auto primitive = std::make_shared<Primitive>("FFTShift");
  ASSERT_NE(primitive, nullptr);
  auto x = std::make_shared<abstract::AbstractTensor>(type_param.x_type, shape_param.x_shape);
  ASSERT_NE(x, nullptr);
  auto axes = shape_param.axes->ToAbstract();
  auto forward = shape_param.forward->ToAbstract();
  std::vector<AbstractBasePtr> input_args = {x, axes, forward};

  // expect output
  auto expect_shape = std::make_shared<abstract::Shape>(shape_param.out_shape);
  ASSERT_NE(expect_shape, nullptr);
  auto expect_dtype = std::make_shared<TensorType>(type_param.out_type);
  ASSERT_NE(expect_dtype, nullptr);

  // execute
  auto out_shape = fftshift_func_impl.InferShape(primitive, input_args);
  auto out_dtype = fftshift_func_impl.InferType(primitive, input_args);

  // verify output
  ASSERT_NE(out_shape, nullptr);
  ASSERT_TRUE(*out_shape == *expect_shape);
  ASSERT_NE(out_dtype, nullptr);
  ASSERT_TRUE(*out_dtype == *expect_dtype);
}

auto fftshift_shape_cases =
  testing::Values(FFTShiftShape{{5, 5}, CreateTuple({0, 1}), CreateScalar(true), {5, 5}},
                  FFTShiftShape{{-1, -1, -1}, CreateTuple({0, 1}), CreateScalar(true), {-1, -1, -1}},
                  FFTShiftShape{{-2}, CreateTuple({0, 1}), CreateScalar(true), {-2}},
                  FFTShiftShape{{5, 5}, CreateTuple({0, 1}), CreateScalar(false), {5, 5}},
                  FFTShiftShape{{-1, -1, -1}, CreateTuple({0, 1}), CreateScalar(false), {-1, -1, -1}},
                  FFTShiftShape{{-2}, CreateTuple({0, 1}), CreateScalar(false), {-2}});

auto fftshift_type_cases = testing::ValuesIn({
  FFTShiftType{kBool, kBool},
  FFTShiftType{kUInt8, kUInt8},
  FFTShiftType{kUInt16, kUInt16},
  FFTShiftType{kUInt32, kUInt32},
  FFTShiftType{kUInt64, kUInt64},
  FFTShiftType{kInt8, kInt8},
  FFTShiftType{kInt16, kInt16},
  FFTShiftType{kInt32, kInt32},
  FFTShiftType{kInt64, kInt64},
  FFTShiftType{kFloat16, kFloat16},
  FFTShiftType{kFloat32, kFloat32},
  FFTShiftType{kFloat64, kFloat64},
  FFTShiftType{kComplex64, kComplex64},
  FFTShiftType{kComplex128, kComplex128},
});

INSTANTIATE_TEST_CASE_P(TestFFTShiftGroup, TestFFTShift, testing::Combine(fftshift_shape_cases, fftshift_type_cases));
}  // namespace ops
}  // namespace mindspore

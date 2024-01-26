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
#include "ops/ops_func_impl/fftbase.h"
#include "ops/auto_generate/gen_ops_name.h"
#include "ops/op_name.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
#define I64(x) (static_cast<int64_t>((x)))
struct FFTBaseShape {
  ShapeVector x_shape;
  ValuePtr s;
  ValuePtr dims;
  ValuePtr norm;
  ValuePtr fft_mode;
  ValuePtr forward;
  ShapeVector out_shape;
};

struct FFTBaseType {
  TypePtr x_type;
  TypePtr out_type;
};

class TestFFTBase : public TestOps, public testing::WithParamInterface<std::tuple<FFTBaseShape, FFTBaseType>> {};

TEST_P(TestFFTBase, dyn_shape) {
  // prepare
  const auto &shape_param = std::get<0>(GetParam());
  const auto &type_param = std::get<1>(GetParam());

  // input
  FFTBaseFuncImpl fftbase_func_impl;
  auto primitive = std::make_shared<Primitive>("FFTBase");
  ASSERT_NE(primitive, nullptr);
  auto x = std::make_shared<abstract::AbstractTensor>(type_param.x_type, shape_param.x_shape);
  ASSERT_NE(x, nullptr);
  auto s = shape_param.s->ToAbstract();
  auto dims = shape_param.dims->ToAbstract();
  auto norm = shape_param.norm->ToAbstract();
  auto fft_mode = shape_param.fft_mode->ToAbstract();
  auto forward = shape_param.forward->ToAbstract();
  std::vector<AbstractBasePtr> input_args = {x, s, dims, norm, fft_mode, forward};

  // expect output
  auto expect_shape = std::make_shared<abstract::Shape>(shape_param.out_shape);
  ASSERT_NE(expect_shape, nullptr);
  auto expect_dtype = std::make_shared<TensorType>(type_param.out_type);
  ASSERT_NE(expect_dtype, nullptr);

  // execute
  auto out_shape = fftbase_func_impl.InferShape(primitive, input_args);
  auto out_dtype = fftbase_func_impl.InferType(primitive, input_args);

  // verify output
  ASSERT_NE(out_shape, nullptr);
  ASSERT_TRUE(*out_shape == *expect_shape);
  ASSERT_NE(out_dtype, nullptr);
  ASSERT_TRUE(*out_dtype == *expect_dtype);
}

auto fftbase_shape_cases = testing::Values(
  FFTBaseShape{
    {4, 4}, CreateTuple({}), CreateTuple({I64(-1)}), CreateScalar(0), CreateScalar(0), CreateScalar(true), {4, 4}},
  FFTBaseShape{
    {4, 4}, CreateTuple({I64(4)}), CreateTuple({I64(0)}), CreateScalar(0), CreateScalar(0), CreateScalar(true), {4, 4}},
  FFTBaseShape{
    {4, 4}, CreateTuple({I64(4)}), CreateTuple({I64(1)}), CreateScalar(0), CreateScalar(0), CreateScalar(true), {4, 4}},
  FFTBaseShape{
    {4, 4}, CreateTuple({I64(2)}), CreateTuple({I64(0)}), CreateScalar(0), CreateScalar(0), CreateScalar(true), {2, 4}},
  FFTBaseShape{
    {4, 4}, CreateTuple({I64(2)}), CreateTuple({I64(1)}), CreateScalar(0), CreateScalar(0), CreateScalar(true), {4, 2}},
  FFTBaseShape{
    {4, 4}, CreateTuple({}), CreateTuple({I64(1)}), CreateScalar(0), CreateScalar(0), CreateScalar(true), {4, 4}},
  FFTBaseShape{
    {4, 4}, CreateTuple({I64(2)}), CreateTuple({}), CreateScalar(0), CreateScalar(0), CreateScalar(true), {4, 2}},
  FFTBaseShape{
    {-1, -1}, CreateTuple({}), CreateTuple({I64(-1)}), CreateScalar(0), CreateScalar(0), CreateScalar(true), {-1, -1}},
  FFTBaseShape{
    {-2}, CreateTuple({}), CreateTuple({I64(-1)}), CreateScalar(0), CreateScalar(0), CreateScalar(true), {-2}},
  FFTBaseShape{
    {4, 4}, CreateTuple({}), CreateTuple({I64(-1)}), CreateScalar(0), CreateScalar(0), CreateScalar(false), {4, 4}},
  FFTBaseShape{{4, 4},
               CreateTuple({I64(4)}),
               CreateTuple({I64(0)}),
               CreateScalar(0),
               CreateScalar(0),
               CreateScalar(false),
               {4, 4}},
  FFTBaseShape{{4, 4},
               CreateTuple({I64(4)}),
               CreateTuple({I64(1)}),
               CreateScalar(0),
               CreateScalar(0),
               CreateScalar(false),
               {4, 4}},
  FFTBaseShape{{4, 4},
               CreateTuple({I64(2)}),
               CreateTuple({I64(0)}),
               CreateScalar(0),
               CreateScalar(0),
               CreateScalar(false),
               {2, 4}},
  FFTBaseShape{{4, 4},
               CreateTuple({I64(2)}),
               CreateTuple({I64(1)}),
               CreateScalar(0),
               CreateScalar(0),
               CreateScalar(false),
               {4, 2}},
  FFTBaseShape{
    {4, 4}, CreateTuple({}), CreateTuple({I64(1)}), CreateScalar(0), CreateScalar(0), CreateScalar(false), {4, 4}},
  FFTBaseShape{
    {4, 4}, CreateTuple({I64(2)}), CreateTuple({}), CreateScalar(0), CreateScalar(0), CreateScalar(false), {4, 2}},
  FFTBaseShape{
    {-1, -1}, CreateTuple({}), CreateTuple({I64(-1)}), CreateScalar(0), CreateScalar(0), CreateScalar(false), {-1, -1}},
  FFTBaseShape{
    {-2}, CreateTuple({}), CreateTuple({I64(-1)}), CreateScalar(0), CreateScalar(0), CreateScalar(false), {-2}});

auto fftbase_type_cases = testing::ValuesIn({
  FFTBaseType{kInt16, kComplex64},
  FFTBaseType{kInt32, kComplex64},
  FFTBaseType{kInt64, kComplex64},
  FFTBaseType{kFloat16, kComplex64},
  FFTBaseType{kFloat32, kComplex64},
  FFTBaseType{kFloat64, kComplex128},
  FFTBaseType{kComplex64, kComplex64},
  FFTBaseType{kComplex128, kComplex128},
});

INSTANTIATE_TEST_CASE_P(TestFFTBaseGroup, TestFFTBase, testing::Combine(fftbase_shape_cases, fftbase_type_cases));
}  // namespace ops
}  // namespace mindspore

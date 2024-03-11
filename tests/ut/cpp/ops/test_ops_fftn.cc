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
#include "ops/ops_func_impl/fftn.h"
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
struct FFTNShape {
  ShapeVector x_shape;
  ValuePtr s;
  ValuePtr dim;
  ValuePtr norm;
  ShapeVector out_shape;
};

struct FFTNType {
  TypePtr x_type;
  TypePtr out_type;
};

class TestFFTN : public TestOps, public testing::WithParamInterface<std::tuple<FFTNShape, FFTNType>> {};

TEST_P(TestFFTN, dyn_shape) {
  // prepare
  const auto &shape_param = std::get<0>(GetParam());
  const auto &type_param = std::get<1>(GetParam());

  // input
  FFTNFuncImpl fftn_func_impl;
  auto primitive = std::make_shared<Primitive>("FFTN");
  ASSERT_NE(primitive, nullptr);
  auto x = std::make_shared<abstract::AbstractTensor>(type_param.x_type, shape_param.x_shape);
  ASSERT_NE(x, nullptr);
  auto s = shape_param.s->ToAbstract();
  auto dim = shape_param.dim->ToAbstract();
  auto norm = shape_param.norm->ToAbstract();
  std::vector<AbstractBasePtr> input_args = {x, s, dim, norm};

  // expect output
  auto expect_shape = std::make_shared<abstract::Shape>(shape_param.out_shape);
  ASSERT_NE(expect_shape, nullptr);
  auto expect_dtype = std::make_shared<TensorType>(type_param.out_type);
  ASSERT_NE(expect_dtype, nullptr);

  // execute
  auto out_shape = fftn_func_impl.InferShape(primitive, input_args);
  auto out_dtype = fftn_func_impl.InferType(primitive, input_args);

  // verify output
  ASSERT_NE(out_shape, nullptr);
  ASSERT_TRUE(*out_shape == *expect_shape);
  ASSERT_NE(out_dtype, nullptr);
  ASSERT_TRUE(*out_dtype == *expect_dtype);
}

auto fftn_shape_cases = testing::Values(
  FFTNShape{{4, 4}, CreateTuple({I64(4), I64(4)}), CreateTuple({I64(0), I64(1)}), CreateScalar(I64(0)), {4, 4}},
  FFTNShape{{4, 4}, CreateTuple({I64(2)}), CreateTuple({I64(0)}), CreateScalar(I64(0)), {2, 4}},
  FFTNShape{{4, 4}, CreateTuple({I64(8)}), CreateTuple({I64(1)}), CreateScalar(I64(1)), {4, 8}},
  FFTNShape{{4, 4}, CreateTuple({I64(2), I64(4)}), CreateTuple({I64(0), I64(1)}), CreateScalar(I64(0)), {2, 4}},
  FFTNShape{{4, 4}, CreateTuple({I64(8), I64(4)}), CreateTuple({I64(0), I64(1)}), CreateScalar(I64(1)), {8, 4}});

auto fftn_type_cases = testing::ValuesIn({
  FFTNType{kInt16, kComplex64},
  FFTNType{kInt32, kComplex64},
  FFTNType{kInt64, kComplex64},
  FFTNType{kFloat16, kComplex64},
  FFTNType{kFloat32, kComplex64},
  FFTNType{kFloat64, kComplex128},
  FFTNType{kComplex64, kComplex64},
  FFTNType{kComplex128, kComplex128},
});

INSTANTIATE_TEST_CASE_P(TestFFTNGroup, TestFFTN, testing::Combine(fftn_shape_cases, fftn_type_cases));
}  // namespace ops
}  // namespace mindspore

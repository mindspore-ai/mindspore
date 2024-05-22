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
#include "ops/ops_func_impl/fftfreq.h"
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
#define F32(x) (static_cast<float>((x)))
struct FFTFreqShape {
  ValuePtr n;
  ValuePtr d;
  ShapeVector out_shape;
};

struct FFTFreqType {
  ValuePtr dtype;
  TypePtr out_type;
};

class TestFFTFreq : public TestOps, public testing::WithParamInterface<std::tuple<FFTFreqShape, FFTFreqType>> {};

TEST_P(TestFFTFreq, dyn_shape) {
  // prepare
  const auto &shape_param = std::get<0>(GetParam());
  const auto &type_param = std::get<1>(GetParam());

  // input
  FFTFreqFuncImpl fftfreq_func_impl;
  auto primitive = std::make_shared<Primitive>("FFTFreq");
  ASSERT_NE(primitive, nullptr);
  auto n = shape_param.n->ToAbstract();
  auto d = shape_param.d->ToAbstract();
  auto dtype = type_param.dtype->ToAbstract();
  std::vector<AbstractBasePtr> input_args = {n, d, dtype};

  // expect output
  auto expect_shape = std::make_shared<abstract::Shape>(shape_param.out_shape);
  ASSERT_NE(expect_shape, nullptr);
  auto expect_dtype = std::make_shared<TensorType>(type_param.out_type);
  ASSERT_NE(expect_dtype, nullptr);

  // execute
  auto out_shape = fftfreq_func_impl.InferShape(primitive, input_args);
  auto out_dtype = fftfreq_func_impl.InferType(primitive, input_args);

  // verify output
  ASSERT_NE(out_shape, nullptr);
  ASSERT_TRUE(*out_shape == *expect_shape);
  ASSERT_NE(out_dtype, nullptr);
  ASSERT_TRUE(*out_dtype == *expect_dtype);
}

auto fftfreq_shape_cases = testing::Values(FFTFreqShape{CreateScalar(I64(4)), CreateScalar(F32(1.0)), {4}},
                                           FFTFreqShape{CreateScalar(I64(7)), CreateScalar(F32(2.5)), {7}},
                                           FFTFreqShape{CreateScalar(I64(9)), CreateScalar(F32(3.7)), {9}},
                                           FFTFreqShape{CreateScalar(I64(1)), CreateScalar(F32(4.2)), {1}});

auto fftfreq_type_cases = testing::ValuesIn({FFTFreqType{CreateScalar<int64_t>(kNumberTypeBFloat16), kBFloat16},
                                             FFTFreqType{CreateScalar<int64_t>(kNumberTypeFloat16), kFloat16},
                                             FFTFreqType{CreateScalar<int64_t>(kNumberTypeFloat32), kFloat32},
                                             FFTFreqType{CreateScalar<int64_t>(kNumberTypeFloat64), kFloat64},
                                             FFTFreqType{CreateScalar<int64_t>(kNumberTypeComplex64), kComplex64},
                                             FFTFreqType{CreateScalar<int64_t>(kNumberTypeComplex128), kComplex128}});

INSTANTIATE_TEST_CASE_P(TestFFTFreqGroup, TestFFTFreq, testing::Combine(fftfreq_shape_cases, fftfreq_type_cases));
}  // namespace ops
}  // namespace mindspore

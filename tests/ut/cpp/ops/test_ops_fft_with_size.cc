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

#include "ops/test_value_utils.h"
#include "ops/auto_generate/gen_ops_name.h"
#include "ops/op_enum.h"
#include "ops/test_ops.h"
#include "ops/ops_func_impl/fft_with_size.h"

namespace mindspore::ops {
#define Any CreateScalar(kValueAny)
struct FFTParams {
  ShapeVector input_shape;
  int signal_ndim;       /* int */
  bool inverse;          /* bool */
  bool real;             /* bool */
  bool onesided;         /* bool */
  ValuePtr signal_sizes; /* tuple[int] */
  ShapeVector output_shape;
  TypePtr input_dtype;
  TypePtr output_dtype;
};

class TestFFT : public TestOps, public testing::WithParamInterface<FFTParams> {};

TEST_P(TestFFT, dyn_shape) {
  const auto &param = GetParam();
  auto input = std::make_shared<abstract::AbstractTensor>(param.input_dtype, param.input_shape);
  auto signal_ndim = param.signal_ndim == -1 ? Any->ToAbstract() : CreatePyInt(param.signal_ndim)->ToAbstract();
  auto inverse = CreateScalar(param.inverse)->ToAbstract();
  auto real = CreateScalar(param.real)->ToAbstract();
  auto norm = CreateScalar(static_cast<int64_t>(ops::NormMode::BACKWARD))->ToAbstract();
  auto onesided = CreateScalar(param.onesided)->ToAbstract();
  auto signal_sizes = param.signal_sizes->ToAbstract();

  FFTWithSizeFuncImpl fft_with_size_func_impl;
  auto prim = std::make_shared<Primitive>(kNameFFTWithSize);
  auto inputs = std::vector<AbstractBasePtr>{input, signal_ndim, inverse, real, norm, onesided, signal_sizes};
  auto expect = std::make_shared<abstract::AbstractTensor>(param.output_dtype, param.output_shape);

  auto out_dtype = fft_with_size_func_impl.InferType(prim, inputs);
  ASSERT_TRUE(*out_dtype == *expect->GetType());
  auto out_shape = fft_with_size_func_impl.InferShape(prim, inputs);
  ASSERT_TRUE(*out_shape == *expect->GetShape());
}

auto cases = testing::Values(
  /*
   * fft mode:
   *      condition:
   *          inverse:   false
   *          real:      false
   *      shape:
   *          output:    input
   *      dtype
   *          output:    complex -> complex
   * */
  FFTParams{{4, 5}, -1, false, false, true, CreateList({Any, CreateScalar(9)}), {4, 5}, kComplex128, kComplex128},
  FFTParams{{2, 3, 4, 5}, 2, false, false, true, CreateList({Any, Any}), {2, 3, 4, 5}, kComplex128, kComplex128},
  FFTParams{{2, 3, 4, 5}, 2, false, false, true, CreatePyIntList({4, 9}), {2, 3, 4, 5}, kComplex128, kComplex128},

  /*
   * ifft mode:
   *      condition:
   *          inverse:   true
   *          real:      false
   *      shape:
   *          output:    input
   *      dtype
   *          output:    complex -> complex
   * */
  FFTParams{{4, 5}, -1, true, false, true, CreateList({Any, CreateScalar(9)}), {4, 5}, kComplex128, kComplex128},
  FFTParams{{2, 3, 4, 5}, 2, true, false, true, CreateList({Any, Any}), {2, 3, 4, 5}, kComplex128, kComplex128},
  FFTParams{{2, 3, 4, 5}, 2, true, false, true, CreatePyIntList({4, 9}), {2, 3, 4, 5}, kComplex128, kComplex128},

  /*
   * rfft mode:
   *      condition:
   *          inverse:   false
   *          real:      true
   *      shape:
   *          output:    input.back = input_shape.back() / 2 + 1;
   *      dtype
   *          output:    common  -> complex
   * */
  FFTParams{{2, 3, 4, 5}, 2, false, true, true, CreateList({Any, Any}), {2, 3, 4, 3}, kFloat32, kComplex64},
  FFTParams{{2, 3, 4, 6}, 2, false, true, false, CreateList({}), {2, 3, 4, 6}, kFloat32, kComplex64},
  FFTParams{{2, 3, 4, 6}, 2, false, true, true, CreatePyIntList({4, 9}), {2, 3, 4, 4}, kUInt8, kComplex64},
  FFTParams{{-1, 6}, -1, false, true, true, CreatePyIntList({4, 9}), {-1, 4}, kFloat64, kComplex128},
  FFTParams{{-2}, -1, false, true, true, CreatePyIntList({4, 9}), {-2}, kInt8, kComplex64},

  /*
   * irfft mode:
   *      condition:
   *          inverse:   true
   *          real:      true
   *      shape:
   *          output:    input.back = signal_sizes.back()                  // when signal_sizes is valid
   *                     input.back = (input_shape.back() - 1) * kDimNum;  // when signal_sizes is empty
   *      dtype
   *          output:    complex -> common
   * */
  FFTParams{{2, 3, 4, 5}, 2, true, true, true, CreatePyIntList({4, 9}), {2, 3, 4, 9}, kComplex128, kFloat64},
  FFTParams{{2, 3, 4, 6}, -1, true, true, true, CreateList({}), {2, 3, 4, 10}, kComplex64, kFloat32},
  FFTParams{{2, 3, 4, -1}, 2, true, true, true, CreatePyIntList({4, 8}), {2, 3, 4, 8}, kComplex64, kFloat32},
  FFTParams{{2, -1}, 2, true, true, true, CreateList({CreatePyInt(4), Any}), {2, -1}, kComplex64, kFloat32});

INSTANTIATE_TEST_CASE_P(TestFFT, TestFFT, cases);
}  // namespace mindspore::ops

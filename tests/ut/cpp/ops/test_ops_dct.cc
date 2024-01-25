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
#include "ops/test_ops.h"
#include "ops/ops_func_impl/dct.h"
#include "mindspore/core/ops/op_enum.h"

namespace mindspore {
namespace ops {
#define Any CreateScalar(kValueAny)
struct DCTParams {
  ShapeVector input_shape;
  int type;     /* int */
  int n;        /* int */
  int axis;     /* int */
  bool forward; /* bool */
  bool grad;    /* bool */
  ShapeVector output_shape;
  TypePtr input_dtype;
  TypePtr output_dtype;
};

class TestDCT : public TestOps, public testing::WithParamInterface<DCTParams> {};

TEST_P(TestDCT, dyn_shape) {
  const auto &param = GetParam();
  auto input = std::make_shared<abstract::AbstractTensor>(param.input_dtype, param.input_shape);
  auto type = CreatePyInt(param.type)->ToAbstract();
  auto n = CreatePyInt(param.n)->ToAbstract();
  auto axis = CreatePyInt(param.axis)->ToAbstract();
  auto norm = CreateScalar(static_cast<int64_t>(ops::NormMode::BACKWARD))->ToAbstract();
  auto forward = CreateScalar(param.forward)->ToAbstract();
  auto grad = CreateScalar(param.grad)->ToAbstract();

  DCTFuncImpl dct_func_impl;
  auto prim = std::make_shared<Primitive>(kNameDCT);
  auto inputs = std::vector<AbstractBasePtr>{input, type, n, axis, norm, forward, grad};
  auto expect = std::make_shared<abstract::AbstractTensor>(param.output_dtype, param.output_shape);

  auto out_dtype = dct_func_impl.InferType(prim, inputs);
  ASSERT_TRUE(*out_dtype == *expect->GetType());
  auto out_shape = dct_func_impl.InferShape(prim, inputs);
  ASSERT_TRUE(*out_shape == *expect->GetShape());
}

auto dct_cases = testing::Values(
  /*
   * dct mode:
   *      condition:
   *          forward:   true
   *      shape:
   *          output:    shape[i] == x.shape[i] if i != axis else n
   *      dtype
   *          output:    float/double -> float/double
   * */
  DCTParams{{4, 5}, 2, 5, -1, true, false, {4, 5}, kFloat32, kFloat32},
  DCTParams{{4, 5}, 2, 6, 0, true, false, {6, 5}, kFloat64, kFloat64},
  DCTParams{{-1, -1, -1}, 2, 6, 0, true, false, {6, -1, -1}, kFloat64, kFloat64},
  DCTParams{{-2}, 2, 6, 0, true, false, {-2}, kFloat64, kFloat64},

  /*
   * idct mode:
   *      condition:
   *          forward:   false
   *      shape:
   *          output:    shape[i] == x.shape[i] if i != axis else n
   *      dtype
   *          output:    float/double -> float/double
   * */
  DCTParams{{4, 5}, 2, 5, -1, false, false, {4, 5}, kFloat32, kFloat32},
  DCTParams{{4, 5}, 2, 6, 0, false, false, {6, 5}, kFloat64, kFloat64},
  DCTParams{{-1, -1, -1}, 2, 6, 0, false, false, {6, -1, -1}, kFloat64, kFloat64},
  DCTParams{{-2}, 2, 6, 0, false, false, {-2}, kFloat64, kFloat64});
INSTANTIATE_TEST_CASE_P(TestDCT, TestDCT, dct_cases);
}  // namespace ops
}  // namespace mindspore

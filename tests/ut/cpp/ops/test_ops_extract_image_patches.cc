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

#include "ops/ops_func_impl/extract_image_patches.h"
#include "ops/test_ops.h"
#include "ops/test_value_utils.h"
#include "include/c_api/ms/base/types.h"

namespace mindspore {
namespace ops {
struct ExtractImagePatchesParams {
  ShapeVector input_shape;
  TypePtr dtype;
  ValuePtr ksizes;
  ValuePtr strides;
  ValuePtr rates;
  ValuePtr padding;
  ShapeVector output_shape;
};

class TestExtractImagePatches : public TestOps, public testing::WithParamInterface<ExtractImagePatchesParams> {};

TEST_P(TestExtractImagePatches, dyn_shape) {
  const auto &param = GetParam();
  auto input = std::make_shared<abstract::AbstractTensor>(param.dtype, param.input_shape);
  auto expect = std::make_shared<abstract::AbstractTensor>(param.dtype, param.output_shape);

  auto ksizes = param.ksizes->ToAbstract();
  auto strides = param.strides->ToAbstract();
  auto rates = param.rates->ToAbstract();
  auto padding = param.padding->ToAbstract();

  ExtractImagePatchesFuncImpl extract_image_patches_func_impl;
  auto prim = std::make_shared<Primitive>("ExtractImagePatches");

  auto out_dtype = extract_image_patches_func_impl.InferType(prim, {input, ksizes, strides, rates, padding});
  ASSERT_TRUE(*out_dtype == *expect->GetType());
  auto out_shape = extract_image_patches_func_impl.InferShape(prim, {input, ksizes, strides, rates, padding});
  ASSERT_TRUE(*out_shape == *expect->GetShape());
}
auto eip_cases = testing::Values(
  /* same */
  ExtractImagePatchesParams{{1, 32, 288, 288},
                            kFloat32,
                            CreatePyIntList({3, 3}),
                            CreatePyIntList({1, 1}),
                            CreatePyIntList({2, 2}),
                            CreatePyInt(PadMode::SAME),
                            {1, 288, 288, 288}},
  ExtractImagePatchesParams{{1, 32, 288, 288},
                            kFloat32,
                            CreateList({CreateScalar(kValueAny), CreatePyInt(3)}),
                            CreatePyIntList({1, 1}),
                            CreatePyIntList({2, 2}),
                            CreatePyInt(PadMode::SAME),
                            {1, -1, 288, 288}},
  ExtractImagePatchesParams{{1, 32, 288, 288},
                            kFloat32,
                            CreatePyIntList({3, 3}),
                            CreateList({CreateScalar(kValueAny), CreatePyInt(1)}),
                            CreatePyIntList({2, 2}),
                            CreatePyInt(PadMode::SAME),
                            {1, 288, -1, 288}},
  ExtractImagePatchesParams{{1, 32, 288, 288},
                            kFloat32,
                            CreatePyIntList({3, 3}),
                            CreatePyIntList({1, 1}),
                            CreateList({CreatePyInt(2), CreateScalar(kValueAny)}),
                            CreatePyInt(PadMode::SAME),
                            {1, 288, 288, 288}},

  /* valid */
  ExtractImagePatchesParams{{1, 32, 288, 288},
                            kFloat32,
                            CreatePyIntList({3, 3}),
                            CreatePyIntList({1, 1}),
                            CreatePyIntList({2, 2}),
                            CreatePyInt(PadMode::VALID),
                            {1, 288, 284, 284}},
  ExtractImagePatchesParams{{1, 32, 288, 288},
                            kFloat32,
                            CreateList({CreateScalar(kValueAny), CreatePyInt(3)}),
                            CreatePyIntList({1, 1}),
                            CreatePyIntList({2, 2}),
                            CreatePyInt(PadMode::VALID),
                            {1, -1, -1, 284}},
  ExtractImagePatchesParams{{1, 32, 288, 288},
                            kFloat32,
                            CreatePyIntList({3, 3}),
                            CreateList({CreateScalar(kValueAny), CreatePyInt(1)}),
                            CreatePyIntList({2, 2}),
                            CreatePyInt(PadMode::VALID),
                            {1, 288, -1, 284}},
  ExtractImagePatchesParams{{1, 32, 288, 288},
                            kFloat32,
                            CreatePyIntList({3, 3}),
                            CreatePyIntList({1, 1}),
                            CreateList({CreatePyInt(2), CreateScalar(kValueAny)}),
                            CreatePyInt(PadMode::VALID),
                            {1, 288, 284, -1}},

  /* pad */
  ExtractImagePatchesParams{{1, 32, 288, 288},
                            kFloat32,
                            CreatePyIntList({3, 3}),
                            CreatePyIntList({1, 1}),
                            CreatePyIntList({2, 2}),
                            CreateScalar(kValueAny),
                            {1, 288, -1, -1}},
  ExtractImagePatchesParams{{-2},
                            kFloat32,
                            CreatePyIntList({3, 3}),
                            CreatePyIntList({1, 1}),
                            CreatePyIntList({2, 2}),
                            CreatePyInt(PadMode::SAME),
                            {-1, -1, -1, -1}});

INSTANTIATE_TEST_CASE_P(TestExtractImagePatches, TestExtractImagePatches, eip_cases);
}  // namespace ops
}  // namespace mindspore

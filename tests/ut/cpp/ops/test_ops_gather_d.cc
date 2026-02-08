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

#include "ops/test_ops.h"
#include "infer/ops_func_impl/gather_d.h"
#include "ops/test_value_utils.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/utils/general_infer_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "primitive/auto_generate/gen_ops_primitive_g.h"

namespace mindspore {
namespace ops {
struct GatherDParams {
  ShapeVector x_shape;
  TypePtr x_dtype;
  ValuePtr dim;
  ShapeVector index_shape;
  TypePtr index_dtype;
};

class TestGatherD : public TestOps, public testing::WithParamInterface<GatherDParams> {};

TEST_P(TestGatherD, dyn_shape) {
  const auto &param = GetParam();
  auto x = std::make_shared<abstract::AbstractTensor>(param.x_dtype, param.x_shape);
  auto dim = param.dim->ToAbstract();
  auto index = std::make_shared<abstract::AbstractTensor>(param.index_dtype, param.index_shape);
  auto expect = std::make_shared<abstract::AbstractTensor>(param.x_dtype, param.index_shape);

  GatherDFuncImpl gather_d_func_impl;
  auto prim = std::make_shared<Primitive>("GatherD");

  auto out_dtype = gather_d_func_impl.InferType(prim, {x, dim, index});
  ASSERT_TRUE(*out_dtype == *expect->GetType());
  auto out_shape = gather_d_func_impl.InferShape(prim, {x, dim, index});
  ASSERT_TRUE(*out_shape == *expect->GetShape());
}

auto gather_d_cases = testing::Values(
  /* static */
  GatherDParams{{2, 3, 4, 5}, kFloat64, CreatePyInt(1), {2, 10, 4, 5}, kInt64},
  GatherDParams{{2, 3, 4, 5}, kFloat16, CreatePyInt(3), {2, 3, 4, 10}, kInt32},
  /* -1 */
  GatherDParams{{-1, 3, -1, 5}, kFloat32, CreatePyInt(0), {2, 6, 4, 5}, kInt64},
  GatherDParams{{-1, 3, -1, 5}, kFloat16, CreateScalar(kValueAny), {-1, 6, 4, 5}, kInt32},
  /* -2 */
  GatherDParams{{-2}, kFloat64, CreatePyInt(1), {2, 10, 4, 5}, kInt64},
  GatherDParams{{-2}, kFloat64, CreatePyInt(4), {-2}, kInt64},
  GatherDParams{{2, 3, 4, 5}, kFloat64, CreatePyInt(2), {-2}, kInt64});
INSTANTIATE_TEST_CASE_P(TestGatherD, TestGatherD, gather_d_cases);

struct GatherDInferValueParams {
  tensor::TensorPtr input;
  ValuePtr dim;
  tensor::TensorPtr index;
  tensor::TensorPtr out;
};

class TestGatherDInferValue : public TestOps, public testing::WithParamInterface<GatherDInferValueParams> {};

TEST_P(TestGatherDInferValue, dyn_shape_infer_value) {
  const auto &param = GetParam();
  ASSERT_NE(param.input, nullptr);
  auto input = param.input->ToAbstract();
  ASSERT_NE(input, nullptr);

  ASSERT_NE(param.dim, nullptr);
  auto dim = param.dim->ToAbstract();
  ASSERT_NE(dim, nullptr);

  ASSERT_NE(param.index, nullptr);
  auto index = param.index->ToAbstract();
  ASSERT_NE(index, nullptr);

  auto input_args = abstract::AbstractBasePtrList{input, dim, index};
  auto value_opt = abstract::InferValueByFuncImpl(prim::kPrimGatherD, input_args);
  if (!value_opt.has_value()) {
    MS_LOG(ERROR) << "GatherD have no infer value implement!";
    ASSERT_TRUE(false);
  }
  auto infer_out = value_opt.value();
  if (infer_out == nullptr) {
    MS_LOG(ERROR) << "GatherD can not infer value with inputs: " << input_args;
    ASSERT_TRUE(false);
  }
  auto infer_tensor = infer_out->cast<tensor::TensorPtr>();
  ASSERT_NE(infer_tensor, nullptr);
  ASSERT_TRUE(infer_tensor->ValueEqual(*param.out));
}

INSTANTIATE_TEST_CASE_P(
  TestGatherDInferValue, TestGatherDInferValue,
  testing::Values(
    GatherDInferValueParams{CreateTensor<float>(kNumberTypeFloat32, ShapeVector{2, 3}, std::vector<float>{-0.1, 0.3, 3.6, 0.4, 0.5, -3.2}),
                            CreateScalar<int64_t>(1),
                            CreateTensor<int32_t>(kNumberTypeInt32, ShapeVector{2, 2}, std::vector<int32_t>{0, 0, 1, 1}),
                            CreateTensor<float>(kNumberTypeFloat32, ShapeVector{2, 2}, std::vector<float>{-0.1, -0.1, 0.5, 0.5})},
    GatherDInferValueParams{CreateTensor<float>(kNumberTypeFloat32, ShapeVector{2, 3}, std::vector<float>{-0.1, 0.3, 3.6, 0.4, 0.5, -3.2}),
                            CreateScalar<int64_t>(0),
                            CreateTensor<int32_t>(kNumberTypeInt32, ShapeVector{2, 2}, std::vector<int32_t>{0, 0, 1, 1}),
                            CreateTensor<float>(kNumberTypeFloat32, ShapeVector{2, 2}, std::vector<float>{-0.1, 0.3, 0.4, 0.5})},
    GatherDInferValueParams{CreateTensor<float>(kNumberTypeFloat32, ShapeVector{2, 3}, std::vector<float>{-0.1, 0.3, 3.6, 0.4, 0.5, -3.2}),
                            CreateScalar<int64_t>(1),
                            CreateTensor<int32_t>(kNumberTypeInt32, ShapeVector{2, 2}, std::vector<int32_t>{1, 2, 2, 0}),
                            CreateTensor<float>(kNumberTypeFloat32, ShapeVector{2, 2}, std::vector<float>{0.3, 3.6, -3.2, 0.4})},
    GatherDInferValueParams{CreateTensor<float>(kNumberTypeFloat32, ShapeVector{3, 3}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9}),
                            CreateScalar<int64_t>(0),
                            CreateTensor<int32_t>(kNumberTypeInt32, ShapeVector{2, 3}, std::vector<int32_t>{0, 2, 1, 2, 1, 1}),
                            CreateTensor<float>(kNumberTypeFloat32, ShapeVector{2, 3}, std::vector<float>{1, 8, 6, 7, 5, 6})},
    GatherDInferValueParams{CreateTensor<float>(kNumberTypeFloat32, ShapeVector{3, 3}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9}),
                            CreateScalar<int64_t>(1),
                            CreateTensor<int32_t>(kNumberTypeInt32, ShapeVector{2, 3}, std::vector<int32_t>{0, 2, 1, 2, 1, 1}),
                            CreateTensor<float>(kNumberTypeFloat32, ShapeVector{2, 3}, std::vector<float>{1, 3, 2, 6, 5, 5})},
    GatherDInferValueParams{CreateTensor<float>(kNumberTypeFloat32, ShapeVector{3, 3}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9}),
                            CreateScalar<int64_t>(-1),
                            CreateTensor<int32_t>(kNumberTypeInt32, ShapeVector{2, 3}, std::vector<int32_t>{0, 2, 1, 2, 1, 1}),
                            CreateTensor<float>(kNumberTypeFloat32, ShapeVector{2, 3}, std::vector<float>{1, 3, 2, 6, 5, 5})},
    GatherDInferValueParams{CreateTensor<float>(kNumberTypeFloat32, ShapeVector{1, 3, 3}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9}),
                            CreateScalar<int64_t>(1),
                            CreateTensor<int32_t>(kNumberTypeInt32, ShapeVector{1, 2, 3}, std::vector<int32_t>{0, 2, 0, 1, 0, 0}),
                            CreateTensor<float>(kNumberTypeFloat32, ShapeVector{1, 2, 3}, std::vector<float>{1, 8, 3, 4, 2, 3})},
    GatherDInferValueParams{CreateTensor<float>(kNumberTypeFloat32, ShapeVector{1, 3, 3}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9}),
                            CreateScalar<int64_t>(2),
                            CreateTensor<int32_t>(kNumberTypeInt32, ShapeVector{1, 2, 3}, std::vector<int32_t>{0, 2, 0, 1, 0, 0}),
                            CreateTensor<float>(kNumberTypeFloat32, ShapeVector{1, 2, 3}, std::vector<float>{1, 3, 1, 5, 4, 4})}));
}  // namespace ops
}  // namespace mindspore

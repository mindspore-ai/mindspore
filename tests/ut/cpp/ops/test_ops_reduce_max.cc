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
#include "ir/tensor_new.h"
#include "ops/test_ops_cmp_utils.h"
#include "ir/dtype/number.h"
#include "infer/ops_func_impl/reduce_max.h"
#include "ops/test_value_utils.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"

namespace mindspore {
namespace ops {

struct ReduceMaxParams {
  ShapeVector input_shape;
  TypePtr     input_dtype;
  ValuePtr    dim;
  ValuePtr    keepdim;
  ShapeVector output_shape;
  TypePtr     output_dtype;
};

class TestReduceMax : public TestOps, public testing::WithParamInterface<ReduceMaxParams> {};

TEST_P(TestReduceMax, dyn_shape) {
  const auto &param = GetParam();
  auto x = std::make_shared<abstract::AbstractTensor>(param.input_dtype, param.input_shape);
  ASSERT_NE(x, nullptr);
  auto dim = param.dim->ToAbstract();
  ASSERT_NE(dim, nullptr);
  auto keepdim = param.keepdim->ToAbstract();
  ASSERT_NE(keepdim, nullptr);

  auto expect_shape = std::make_shared<abstract::Shape>(param.output_shape);
  auto expect_type = std::make_shared<TensorType>(param.output_dtype);
  DoFuncImplInferAndCompare<ReduceMaxFuncImpl>(kNameReduceMax, {x, dim, keepdim}, expect_shape, expect_type);
}

static std::vector<ReduceMaxParams> GetCases() {
  auto dyn_rank = abstract::TensorShape::kShapeRankAny;
  auto dyn_dim = abstract::TensorShape::kShapeDimAny;
  std::vector<ReduceMaxParams> cases = {
    ReduceMaxParams{{4, 2, 3}, kFloat32, CreatePyIntTuple({1}), CreateScalar(true), {4, 1, 3}, kFloat32},
    ReduceMaxParams{{4, 2, 3}, kFloat32, CreatePyIntTuple({1}), CreateScalar(false), {4, 3}, kFloat32},
    ReduceMaxParams{{dyn_rank}, kFloat32, CreatePyIntTuple({1}), CreateScalar(true), {dyn_rank}, kFloat32},
    ReduceMaxParams{{dyn_rank}, kFloat32, CreatePyIntTuple({1}), CreateScalar(false), {dyn_rank}, kFloat32},
    ReduceMaxParams{{4, 2, 3}, kFloat32, kValueAny, CreateScalar(true), {dyn_dim, dyn_dim, dyn_dim}, kFloat32},
    ReduceMaxParams{{4, 2, 3}, kFloat32, kValueAny, CreateScalar(false), {dyn_rank}, kFloat32},
    ReduceMaxParams{{4, dyn_dim, 3}, kFloat32, CreatePyIntTuple({1}), CreateScalar(true), {4, 1, 3}, kFloat32},
    ReduceMaxParams{{4, dyn_dim, 3}, kFloat32, CreatePyIntTuple({1}), CreateScalar(false), {4, 3}, kFloat32},
  };
  return cases;
}

class TestReduceMaxSimple : public TestOps, public testing::WithParamInterface<ReduceMaxParams> {};

TEST_P(TestReduceMaxSimple, simple_infer) {
  const auto &param = GetParam();
  auto x = tensor::from_spec(param.input_dtype->type_id(), param.input_shape, device::DeviceType::kCPU);
  auto dim = param.dim->ToAbstract();
  auto keepdim = param.keepdim->ToAbstract();

  auto expect_shape = std::make_shared<abstract::Shape>(param.output_shape);
  auto expect_type = std::make_shared<TensorType>(param.output_dtype);

  DoFuncImplInferAndCompare<ReduceMaxFuncImpl>(kNameReduceMax, {x->ToAbstract(), dim, keepdim}, expect_shape, expect_type);
}

INSTANTIATE_TEST_CASE_P(TestReduceMax, TestReduceMax, testing::ValuesIn(GetCases()));

INSTANTIATE_TEST_CASE_P(
  TestReduceMaxSimple, TestReduceMaxSimple,testing::ValuesIn(GetCases()));
}  // namespace ops
}  // namespace mindspore

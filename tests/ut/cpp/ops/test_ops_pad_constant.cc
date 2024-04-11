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
#include "ops/test_ops_cmp_utils.h"
#include "ir/dtype/number.h"
#include "ops/ops_func_impl/constant_pad_nd.h"
#include "ops/auto_generate/gen_ops_name.h"
#include "ops/test_value_utils.h"
#include "abstract/dshape.h"

namespace mindspore {
namespace ops {
struct ConstantPadNDParams  {
  ShapeVector input_shape;
  TypePtr input_dtype;
  ValuePtr    padding;
  ValuePtr    constant_value;
  ShapeVector output_shape;
  TypePtr output_dtype;
};

class TestConstantPadND : public TestOps, public testing::WithParamInterface<ConstantPadNDParams> {};

TEST_P(TestConstantPadND, dyn_shape) {
  const auto &param = GetParam();
  auto x = std::make_shared<abstract::AbstractTensor>(param.input_dtype, param.input_shape);
  ASSERT_NE(x, nullptr);
  auto padding = param.padding->ToAbstract();
  ASSERT_NE(padding, nullptr);
  auto constant_value = param.constant_value->ToAbstract();
  ASSERT_NE(constant_value, nullptr);

  auto expect_shape = std::make_shared<abstract::Shape>(param.output_shape);
  auto expect_type = std::make_shared<TensorType>(param.output_dtype);
  DoFuncImplInferAndCompare<ConstantPadNDFuncImpl>(kNameConstantPadND, {x, padding, constant_value}, expect_shape, expect_type);
}

static std::vector<ConstantPadNDParams> GetCases() {
  auto dyn_rank = abstract::TensorShape::kShapeRankAny;
  auto dyn_dim = abstract::TensorShape::kShapeDimAny;
  std::vector<ConstantPadNDParams> cases = {
    ConstantPadNDParams{{4, 2, 3}, kFloat32, CreatePyIntTuple({1, 1}),
                        CreateScalar(1), {4, 2, 5}, kFloat32},
    ConstantPadNDParams{{dyn_rank}, kFloat32, CreatePyIntTuple({dyn_dim, 1}),
                        CreateScalar(1), {dyn_rank}, kFloat32},
    ConstantPadNDParams{{4, dyn_dim, dyn_dim}, kFloat32, CreatePyIntTuple({1, 1}),
                        CreateScalar(1), {4, dyn_dim, dyn_dim}, kFloat32},
    ConstantPadNDParams{{4, dyn_dim, dyn_dim}, kFloat32, kValueAny,
                        CreateScalar(1), {dyn_dim, dyn_dim, dyn_dim}, kFloat32},
    ConstantPadNDParams{{4, dyn_dim, dyn_dim}, kFloat32, CreatePyIntTuple({dyn_dim, 1}),
                        CreateScalar(1), {4, dyn_dim, dyn_dim}, kFloat32}
  };
  return cases;
}

INSTANTIATE_TEST_CASE_P(TestConstantPadND, TestConstantPadND, testing::ValuesIn(GetCases()));
}  // namespace ops
}  // namespace mindspore

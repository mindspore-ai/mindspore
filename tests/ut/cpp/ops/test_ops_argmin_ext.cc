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
#include "ops/test_ops_cmp_utils.h"
#include "ir/dtype/number.h"
#include "ops/ops_func_impl/argmin_ext.h"
#include "ops/test_value_utils.h"
#include "abstract/dshape.h"
#include "ops/auto_generate/gen_ops_primitive.h"
#include "utils/tensor_construct_utils.h"

namespace mindspore {
namespace ops {

struct ArgMinExtParams {
  ShapeVector input_shape;
  TypePtr     input_dtype;
  ValuePtr    dim;
  ValuePtr    keepdim;
  ShapeVector output_shape;
  TypePtr     output_dtype;
};

class TestArgMinExt : public TestOps, public testing::WithParamInterface<ArgMinExtParams> {};

TEST_P(TestArgMinExt, dyn_shape) {
  const auto &param = GetParam();
  auto x = std::make_shared<abstract::AbstractTensor>(param.input_dtype, param.input_shape);
  ASSERT_NE(x, nullptr);
  auto dim = param.dim->ToAbstract();
  ASSERT_NE(dim, nullptr);
  auto keepdim = param.keepdim->ToAbstract();
  ASSERT_NE(keepdim, nullptr);

  auto expect_shape = std::make_shared<abstract::Shape>(param.output_shape);
  auto expect_type = std::make_shared<TensorType>(param.output_dtype);
  DoFuncImplInferAndCompare<ArgMinExtFuncImpl>(kNameArgMinExt, {x, dim, keepdim}, expect_shape, expect_type);
}

static std::vector<ArgMinExtParams> GetCases() {
  auto dyn_rank = abstract::TensorShape::kShapeRankAny;
  auto dyn_dim = abstract::TensorShape::kShapeDimAny;
  std::vector<ArgMinExtParams> cases = {
    ArgMinExtParams{{4, 2, 3}, kFloat16, CreateScalar<int64_t>(1), CreateScalar(false), {4, 3}, kInt64},
    ArgMinExtParams{{dyn_rank}, kFloat16, CreateScalar<int64_t>(1), CreateScalar(false), {dyn_rank}, kInt64},
    ArgMinExtParams{{4, 2, 3}, kFloat16, CreateScalar(kValueAny), CreateScalar(false), {dyn_dim, dyn_dim}, kInt64},
    ArgMinExtParams{{4, dyn_dim, 3}, kFloat16, CreateScalar<int64_t>(1), CreateScalar(false), {4, 3}, kInt64},
  };
  return cases;
}

class TestArgMinExtSimple : public TestOps, public testing::WithParamInterface<ArgMinExtParams> {};

TEST_P(TestArgMinExtSimple, simple_infer) {
  const auto &param = GetParam();
  auto x = std::make_shared<tensor::BaseTensor>(param.input_dtype->type_id(), param.input_shape);
  auto dim = param.dim->ToAbstract();
  auto keepdim = param.keepdim->ToAbstract();

  auto expect_shape = std::make_shared<abstract::Shape>(param.output_shape);
  auto expect_type = std::make_shared<TensorType>(param.output_dtype);

  DoFuncImplInferAndCompare<ArgMinExtFuncImpl>(kNameArgMinExt, {x->ToAbstract(), dim, keepdim}, expect_shape, expect_type);
}

INSTANTIATE_TEST_CASE_P(TestArgMinExt, TestArgMinExt, testing::ValuesIn(GetCases()));

INSTANTIATE_TEST_CASE_P(
  TestArgMinExtSimple, TestArgMinExtSimple,
  testing::Values(
    ArgMinExtParams{{4, 2, 3}, kFloat16, CreateScalar<int64_t>(1), CreateScalar(false), {4, 3}, kInt64},
    ArgMinExtParams{{4, 2, 3}, kFloat16, CreateScalar<int64_t>(0), CreateScalar(false), {2, 3}, kInt64},
    ArgMinExtParams{{4, 2, 3}, kFloat16, CreateScalar<int64_t>(0), CreateScalar(true), {1, 2, 3}, kInt64}
));
}  // namespace ops
}  // namespace mindspore

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
#include "infer/ops_func_impl/select_ext_view.h"
#include "ops/test_value_utils.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"

namespace mindspore {
namespace ops {

struct SelectExtParams {
  ShapeVector input_shape;
  TypePtr     input_dtype;
  ValuePtr    dim;
  ValuePtr    index;
  ShapeVector output_shape;
  TypePtr     output_dtype;
};

class TestSelectExt : public TestOps, public testing::WithParamInterface<SelectExtParams> {};

TEST_P(TestSelectExt, dyn_shape) {
  const auto &param = GetParam();
  auto x = std::make_shared<abstract::AbstractTensor>(param.input_dtype, param.input_shape);
  ASSERT_NE(x, nullptr);
  auto dim = param.dim->ToAbstract();
  ASSERT_NE(dim, nullptr);
  auto index = param.index->ToAbstract();
  ASSERT_NE(index, nullptr);

  auto expect_shape = std::make_shared<abstract::Shape>(param.output_shape);
  auto expect_type = std::make_shared<TensorType>(param.output_dtype);
  DoFuncImplInferAndCompare<SelectExtViewFuncImpl>(kNameSelectExtView, {x, dim, index}, expect_shape, expect_type);
}

static std::vector<SelectExtParams> GetCases() {
  auto dyn_rank = abstract::TensorShape::kShapeRankAny;
  auto dyn_dim = abstract::TensorShape::kShapeDimAny;
  std::vector<SelectExtParams> cases = {
    SelectExtParams{{4, 2, 3}, kFloat16, CreateScalar<int64_t>(1), CreateScalar<int64_t>(1), {4, 3}, kFloat16},
    SelectExtParams{{dyn_rank}, kFloat16, CreateScalar<int64_t>(1), CreateScalar<int64_t>(1), {dyn_rank}, kFloat16},
    SelectExtParams{{4, 2, 3}, kFloat16, CreateScalar(kValueAny), CreateScalar<int64_t>(1), {dyn_rank}, kFloat16},
    SelectExtParams{{4, dyn_dim, 3}, kFloat16, CreateScalar<int64_t>(1), CreateScalar<int64_t>(1), {4, 3}, kFloat16},
  };
  return cases;
}

class TestSelectExtSimple : public TestOps, public testing::WithParamInterface<SelectExtParams> {};

TEST_P(TestSelectExtSimple, simple_infer) {
  const auto &param = GetParam();
  auto x = tensor::from_spec(param.input_dtype->type_id(), param.input_shape, device::DeviceType::kCPU);
  auto dim = param.dim->ToAbstract();
  auto index = param.index->ToAbstract();

  auto expect_shape = std::make_shared<abstract::Shape>(param.output_shape);
  auto expect_type = std::make_shared<TensorType>(param.output_dtype);

  DoFuncImplInferAndCompare<SelectExtViewFuncImpl>(kNameSelectExtView, {x->ToAbstract(), dim, index}, expect_shape, expect_type);
}

INSTANTIATE_TEST_CASE_P(TestSelectExt, TestSelectExt, testing::ValuesIn(GetCases()));

INSTANTIATE_TEST_CASE_P(
  TestSelectExtSimple, TestSelectExtSimple,
  testing::Values(
    SelectExtParams{{4, 2, 3}, kFloat16, CreateScalar<int64_t>(1), CreateScalar<int64_t>(1), {4, 3}, kFloat16},
    SelectExtParams{{4, 2, 3}, kFloat16, CreateScalar<int64_t>(0), CreateScalar<int64_t>(1), {2, 3}, kFloat16}
));
}  // namespace ops
}  // namespace mindspore

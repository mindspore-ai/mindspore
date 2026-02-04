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
#include <memory>
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"
#include "infer/ops_func_impl/transpose_ext_view.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/test_value_utils.h"
#include "ir/dtype/tensor_type.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"

namespace mindspore::ops {
#define I64(x) (static_cast<int64_t>((x)))

struct TransExtParams {
  bool dynamic_len;
  ShapeVector x_shape;
  TypePtr x_type;
  ValuePtr dim0;
  ValuePtr dim1;
  ShapeVector out_shape;
};

class TestTransposeExt : public TestOps, public testing::WithParamInterface<TransExtParams> {};

TEST_P(TestTransposeExt, dyn_shape) {
  const auto &param = GetParam();

  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  ASSERT_NE(x, nullptr);

  auto dim0_abs = param.dim0->ToAbstract();
  auto dim1_abs = param.dim1->ToAbstract();
  ASSERT_NE(dim0_abs, nullptr);
  ASSERT_NE(dim1_abs, nullptr);

  auto expect = std::make_shared<abstract::AbstractTensor>(param.x_type, param.out_shape);
  ASSERT_NE(expect, nullptr);

  auto expect_shape = std::make_shared<abstract::TensorShape>(param.out_shape);
  auto expect_type = std::make_shared<TensorType>(param.x_type);
  DoFuncImplInferAndCompare<TransposeExtViewFuncImpl>(kNameTransposeExtView, {x, dim0_abs, dim1_abs}, expect_shape,
                                                      expect_type);
}

INSTANTIATE_TEST_CASE_P(
  TestTransposeExt, TestTransposeExt,
  testing::Values(
    TransExtParams{false, {2, 3, 4, 5}, kFloat32, CreateScalar<int64_t>(0), CreateScalar<int64_t>(1), {3, 2, 4, 5}},
    TransExtParams{false, {2, 3, -1, 5}, kFloat32, CreateScalar<int64_t>(0), CreateScalar<int64_t>(2), {-1, 3, 2, 5}},
    TransExtParams{true, {-2}, kFloat32, CreateScalar(kValueAny), CreateScalar(kValueAny), {-2}},
    TransExtParams{false, {2, -1, 4, -1}, kFloat32, CreateScalar<int64_t>(2), CreateScalar<int64_t>(3), {2, -1, -1, 4}},
    TransExtParams{false, {2, 3, 4, 5}, kFloat32, CreateScalar<int64_t>(1), CreateScalar<int64_t>(3), {2, 5, 4, 3}},
    TransExtParams{false, {2, 3, -1, 5}, kFloat32, CreateScalar<int64_t>(1), CreateScalar<int64_t>(3), {2, 5, -1, 3}}));
}  // namespace mindspore::ops

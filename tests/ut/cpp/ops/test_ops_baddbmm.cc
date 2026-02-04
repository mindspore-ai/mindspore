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

#include <memory>
#include <vector>
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "common/common_test.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "primitive/nn_op_name.h"
#include "primitive/op_name.h"
#include "infer/ops_func_impl/baddbmm.h"
#include "ops/test_ops.h"
#include "ops/test_value_utils.h"
#include "utils/ms_context.h"
#include "utils/tensor_construct_utils.h"

namespace mindspore {
namespace ops {
struct BaddbmmParams {
  ShapeVector x_shape;
  TypePtr x_type;
  ShapeVector batch1_shape;
  TypePtr batch1_type;
  ShapeVector batch2_shape;
  TypePtr batch2_type;
  ValuePtr beta;
  ValuePtr alpha;
  ShapeVector out_shape;
};

class TestBaddbmm : public TestOps, public testing::WithParamInterface<BaddbmmParams> {};

TEST_P(TestBaddbmm, dyn_shape) {
  const auto &param = GetParam();
  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  auto batch1 = std::make_shared<abstract::AbstractTensor>(param.batch1_type, param.batch1_shape);
  auto batch2 = std::make_shared<abstract::AbstractTensor>(param.batch2_type, param.batch2_shape);
  auto beta = param.beta->ToAbstract();
  auto alpha = param.alpha->ToAbstract();
  ASSERT_NE(x, nullptr);
  ASSERT_NE(batch1, nullptr);
  ASSERT_NE(batch2, nullptr);
  ASSERT_NE(beta, nullptr);
  ASSERT_NE(alpha, nullptr);

  auto expect = std::make_shared<abstract::TensorShape>(param.out_shape);
  auto prim = std::make_shared<Primitive>("Baddbmm");
  auto infer_impl = std::make_shared<BaddbmmFuncImpl>();
  auto out_shape = infer_impl->InferShape(prim, {x, batch1, batch2, beta, alpha});
  ASSERT_NE(out_shape, nullptr);
  ASSERT_TRUE(*out_shape == *expect);
}

INSTANTIATE_TEST_CASE_P(
  TestBaddbmm, TestBaddbmm,
  testing::Values(
    BaddbmmParams{ShapeVector{3, 4}, kFloat32, ShapeVector{2, 3, 6}, kFloat32, ShapeVector{2, 6, 4}, kFloat32,
                  CreateScalar<int64_t>(2), CreateScalar<int64_t>(8), ShapeVector{2, 3, 4}},
    BaddbmmParams{ShapeVector{2, 3, 4}, kFloat16, ShapeVector{2, 3, 5}, kFloat16, ShapeVector{2, 5, 4}, kFloat16,
                  CreateScalar(kValueAny), CreateScalar<int64_t>(8), ShapeVector{2, 3, 4}},
    BaddbmmParams{ShapeVector{4}, kBFloat16, ShapeVector{2, 3, 2}, kBFloat16, ShapeVector{2, 2, 4}, kBFloat16,
                  CreateScalar<int64_t>(2), CreateScalar(kValueAny), ShapeVector{2, 3, 4}},
    BaddbmmParams{ShapeVector{2, 3, 4}, kBFloat16, ShapeVector{2, 3, 4}, kBFloat16, ShapeVector{2, 4, 4}, kBFloat16,
                  CreateScalar(kValueAny), CreateScalar(kValueAny), ShapeVector{2, 3, 4}},
    BaddbmmParams{ShapeVector{-1, 2, -1}, kFloat32, ShapeVector{-1, 2, 4}, kFloat32, ShapeVector{-1, 4, -1}, kFloat32,
                  CreateScalar<int64_t>(2), CreateScalar<int64_t>(8), ShapeVector{-1, 2, -1}},
    BaddbmmParams{ShapeVector{-1, 2, -1}, kFloat32, ShapeVector{-1, 2, 5}, kFloat32, ShapeVector{-1, 5, -1}, kFloat32,
                  CreateScalar<int64_t>(2), CreateScalar(kValueAny), ShapeVector{-1, 2, -1}},
    BaddbmmParams{ShapeVector{-1, 2, -1}, kFloat32, ShapeVector{-1, 2, -1}, kFloat32, ShapeVector{-1, 2, -1}, kFloat32,
                  CreateScalar(kValueAny), CreateScalar<int64_t>(8), ShapeVector{-1, 2, -1}},
    BaddbmmParams{ShapeVector{-1, 2, -1}, kFloat32, ShapeVector{-1, 2, 1}, kFloat32, ShapeVector{-1, 1, -1}, kFloat32,
                  CreateScalar(kValueAny), CreateScalar(kValueAny), ShapeVector{-1, 2, -1}}));
}  // namespace ops
}  // namespace mindspore

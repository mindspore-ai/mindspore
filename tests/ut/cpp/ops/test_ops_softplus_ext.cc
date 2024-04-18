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
#include <vector>
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "common/common_test.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "ops/nn_op_name.h"
#include "ops/op_name.h"
#include "ops/ops_func_impl/softplus_ext.h"
#include "ops/test_ops.h"
#include "ops/test_value_utils.h"
#include "utils/ms_context.h"
#include "utils/tensor_construct_utils.h"

namespace mindspore {
namespace ops {
struct SoftplusExtParams {
  ShapeVector x_shape;
  TypePtr x_type;
  ValuePtr beta;
  ValuePtr threshold;
  ShapeVector out_shape;
};

class TestSoftplusExt : public TestOps, public testing::WithParamInterface<SoftplusExtParams> {};

TEST_P(TestSoftplusExt, dyn_shape) {
  const auto &param = GetParam();
  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  auto beta = param.beta->ToAbstract();
  auto threshold = param.threshold->ToAbstract();
  ASSERT_NE(x, nullptr);
  ASSERT_NE(beta, nullptr);
  ASSERT_NE(threshold, nullptr);

  auto expect = std::make_shared<abstract::TensorShape>(param.out_shape);
  auto prim = std::make_shared<Primitive>("SoftplusExt");
  auto infer_impl = std::make_shared<SoftplusExtFuncImpl>();
  auto out_shape = infer_impl->InferShape(prim, {x, beta, threshold});
  ASSERT_NE(out_shape, nullptr);
  ASSERT_TRUE(*out_shape == *expect);
}

INSTANTIATE_TEST_CASE_P(
  TestSoftplusExt, TestSoftplusExt,
  testing::Values(SoftplusExtParams{ShapeVector{2, 3, 4}, kFloat32, CreateScalar<int64_t>(2), CreateScalar<int64_t>(8), ShapeVector{2, 3, 4}},
                  SoftplusExtParams{ShapeVector{2, 3, 4}, kFloat16, CreateScalar(kValueAny), CreateScalar<int64_t>(8), ShapeVector{2, 3, 4}},
                  SoftplusExtParams{ShapeVector{2, 3, 4}, kBFloat16, CreateScalar<int64_t>(2), CreateScalar(kValueAny), ShapeVector{2, 3, 4}},
                  SoftplusExtParams{ShapeVector{2, 3, 4}, kBFloat16, CreateScalar(kValueAny), CreateScalar(kValueAny), ShapeVector{2, 3, 4}},
                  SoftplusExtParams{ShapeVector{-1, 2, -1}, kFloat32, CreateScalar<int64_t>(2), CreateScalar<int64_t>(8), ShapeVector{-1, 2, -1}},
                  SoftplusExtParams{ShapeVector{-1, 2, -1}, kFloat32, CreateScalar<int64_t>(2), CreateScalar(kValueAny), ShapeVector{-1, 2, -1}},
                  SoftplusExtParams{ShapeVector{-1, 2, -1}, kFloat32, CreateScalar(kValueAny), CreateScalar<int64_t>(8), ShapeVector{-1, 2, -1}},
                  SoftplusExtParams{ShapeVector{-1, 2, -1}, kFloat32, CreateScalar(kValueAny), CreateScalar(kValueAny), ShapeVector{-1, 2, -1}},
                  SoftplusExtParams{ShapeVector{-2}, kFloat32, CreateScalar<int64_t>(2), CreateScalar<int64_t>(10), ShapeVector{-2}},
                  SoftplusExtParams{ShapeVector{-2}, kFloat32, CreateScalar(kValueAny), CreateScalar<int64_t>(10), ShapeVector{-2}},
                  SoftplusExtParams{ShapeVector{-2}, kFloat32, CreateScalar<int64_t>(2), CreateScalar(kValueAny), ShapeVector{-2}},
                  SoftplusExtParams{ShapeVector{-2}, kFloat32, CreateScalar(kValueAny), CreateScalar(kValueAny), ShapeVector{-2}}));
}  // namespace ops
}  // namespace mindspore

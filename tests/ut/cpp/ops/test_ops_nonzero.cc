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
#include <memory>
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "common/common_test.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "ops/ops_func_impl/non_zero.h"
#include "ops/ops_frontend_func_impl.h"
#include "ops/test_ops.h"

namespace mindspore {
namespace ops {
struct NonZeroOpParams {
  ShapeVector x_shape;
  TypePtr x_type;
  ShapeVector out_shape;
  TypePtr out_type;
  bool is_compile_only;
};
class TestNonZero : public TestOps, public testing::WithParamInterface<NonZeroOpParams> {};

TEST_P(TestNonZero, non_zero_dyn_shape) {
  auto primitive = std::make_shared<Primitive>("Nonzero");
  ASSERT_NE(primitive, nullptr);
  const auto &param = GetParam();
  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  ASSERT_NE(x, nullptr);
  std::vector<abstract::AbstractBasePtr> input_args{std::move(x)};
  auto expect_shape = std::make_shared<abstract::Shape>(param.out_shape);
  ASSERT_NE(expect_shape, nullptr);
  auto expect_type = std::make_shared<TensorType>(param.out_type);
  ASSERT_NE(expect_type, nullptr);

  if (param.is_compile_only) {
    auto infer_impl = GetOpFrontendFuncImplPtr("NonZero");
    ASSERT_NE(infer_impl, nullptr);
    auto infer_shape_type = infer_impl->InferAbstract(primitive, input_args);
    ASSERT_NE(infer_shape_type, nullptr);
    auto infer_shape = infer_shape_type->GetShape();
    ASSERT_NE(infer_shape, nullptr);
    auto infer_type = infer_shape_type->GetType();
    ASSERT_NE(infer_type, nullptr);
    ASSERT_TRUE(*infer_shape == *expect_shape);
    ASSERT_TRUE(*infer_type == *expect_type);
  } else {
    auto infer_impl = std::make_shared<NonZeroFuncImpl>();
    ASSERT_NE(infer_impl, nullptr);
    auto infer_shape = infer_impl->InferShape(primitive, input_args);
    ASSERT_NE(infer_shape, nullptr);
    auto infer_type = infer_impl->InferType(primitive, input_args);
    ASSERT_NE(infer_type, nullptr);
    ASSERT_TRUE(*infer_shape == *expect_shape);
    ASSERT_TRUE(*infer_type == *expect_type);
  }
}

INSTANTIATE_TEST_CASE_P(TestNonZeroGroup, TestNonZero,
                        testing::Values(NonZeroOpParams{{2, 3}, kFloat, {6, 2}, kInt64, false},
                                        NonZeroOpParams{{2, 2, 3}, kFloat, {12, 3}, kInt64, false},
                                        NonZeroOpParams{{3, 4}, kFloat, {-1, 2}, kInt64, true},
                                        NonZeroOpParams{{-1, -1, -1}, kFloat, {-1, 3}, kInt64, true},
                                        NonZeroOpParams{{-2}, kFloat, {-1, -1}, kInt64, true}));
}  // namespace ops
}  // namespace mindspore

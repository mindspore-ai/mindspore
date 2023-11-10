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

#include "ops/test_ops.h"
#include "common/common_test.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "ops/op_name.h"
#include "ops/ops_func_impl/sigmoid_grad.h"
#include "include/backend/optimizer/helper.h"

namespace mindspore {
namespace ops {
struct SigmoidGradShape {
  ShapeVector y_shape;
  ShapeVector out_shape;
};

struct SigmoidGradDType {
  TypePtr y_dtype;
  TypePtr out_dtype;
};

class TestSigmoidGrad : public TestOps, public testing::WithParamInterface<std::tuple<SigmoidGradShape, SigmoidGradDType>> {};

TEST_P(TestSigmoidGrad, dyn_shape) {
  const auto &shape_param = std::get<0>(GetParam());
  const auto &dtype_param = std::get<1>(GetParam());
  auto sigmoid_grad_func_impl = std::make_shared<SigmoidGradFuncImpl>();
  auto prim = std::make_shared<Primitive>("SigmoidGrad");

  auto y = std::make_shared<abstract::AbstractTensor>(dtype_param.y_dtype, shape_param.y_shape);
  ASSERT_NE(y, nullptr);

  auto expect_shape = std::make_shared<abstract::Shape>(shape_param.out_shape);
  auto expect_dtype = std::make_shared<TensorType>(dtype_param.out_dtype);

  auto infer_shape = sigmoid_grad_func_impl->InferShape(prim, {y});
  ASSERT_NE(infer_shape, nullptr);
  ASSERT_TRUE(*infer_shape == *expect_shape);
  auto infer_dtype = sigmoid_grad_func_impl->InferType(prim, {y});
  ASSERT_NE(infer_dtype, nullptr);
  ASSERT_TRUE(*infer_dtype == *expect_dtype);
}

auto SigmoidGradDynTestCase = testing::ValuesIn({
  SigmoidGradShape{{1}, {1}},
  SigmoidGradShape{{-1}, {-1}},
  SigmoidGradShape{{-2}, {-2}},
});

auto SigmoidGradDTypeTestCase = testing::ValuesIn({
  SigmoidGradDType{kFloat16, kFloat16},
  SigmoidGradDType{kFloat32, kFloat32},
  SigmoidGradDType{kFloat64, kFloat64},
});

INSTANTIATE_TEST_CASE_P(TestSigmoidGradGroup, TestSigmoidGrad, testing::Combine(SigmoidGradDynTestCase, SigmoidGradDTypeTestCase));
}  // namespace ops
}  // namespace mindspore

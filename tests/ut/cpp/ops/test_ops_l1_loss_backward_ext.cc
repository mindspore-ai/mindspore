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
#include <memory>
#include "common/common_test.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "include/backend/optimizer/helper.h"
#include "ops/test_ops.h"
#include "ops/ops_func_impl/l1_loss_backward_ext.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
struct L1LossBackwardExtParams {
  ShapeVector grad_output_shape;
  TypePtr grad_output_type;
  ShapeVector input_shape;
  TypePtr input_type;
  ShapeVector target_shape;
  TypePtr target_type;
  ValuePtr reduction;
  ShapeVector output_shape;
};

class TestL1LossBackwardExt : public TestOps, public testing::WithParamInterface<L1LossBackwardExtParams> {};

TEST_P(TestL1LossBackwardExt, dyn_shape) {
  const auto &param = GetParam();
  auto grad_output = std::make_shared<abstract::AbstractTensor>(param.grad_output_type, param.grad_output_shape);
  auto input = std::make_shared<abstract::AbstractTensor>(param.input_type, param.input_shape);
  auto target = std::make_shared<abstract::AbstractTensor>(param.target_type, param.target_shape);
  auto reduction = param.reduction->ToAbstract();

  auto expect_shape = std::make_shared<abstract::Shape>(param.output_shape);
  auto expect_dtype = std::make_shared<TensorType>(param.input_type);

  L1LossBackwardExtFuncImpl l1_loss_backward_ext_func_impl;
  auto prim = std::make_shared<Primitive>("L1LossBackwardExt");
  auto out_dtype = l1_loss_backward_ext_func_impl.InferType(
    prim, {grad_output, input, target, reduction});
  ASSERT_TRUE(*out_dtype == *expect_dtype);
  auto out_shape = l1_loss_backward_ext_func_impl.InferShape(
    prim, {grad_output, input, target, reduction});
  ASSERT_TRUE(*out_shape == *expect_shape);
}
// enum Reduction : int64_t {REDUCTION_SUM = 0,MEAN = 1,NONE = 2,};
INSTANTIATE_TEST_CASE_P(TestL1LossBackwardExt, TestL1LossBackwardExt,
                        testing::Values(L1LossBackwardExtParams{{3, 4, 5},
                                                                kFloat32,
                                                                {3, 4, 5},
                                                                kFloat32,
                                                                {3, 4, 5},
                                                                kFloat32,
                                                                CreateScalar<int64_t>(2),
                                                                {3, 4, 5}},
                                        L1LossBackwardExtParams{{3, 4, 5},
                                                                kFloat32,
                                                                {3, 4, 5},
                                                                kFloat32,
                                                                {3, 4, 5},
                                                                kFloat16,
                                                                CreateScalar<int64_t>(1),
                                                                {3, 4, 5}},
                                        L1LossBackwardExtParams{{3, 4, 5},
                                                                kFloat32,
                                                                {3, 4, 5},
                                                                kFloat32,
                                                                {3, 4, 5},
                                                                kFloat32,
                                                                CreateScalar<int64_t>(0),
                                                                {3, 4, 5}}));
}  // namespace ops
}  // namespace mindspore

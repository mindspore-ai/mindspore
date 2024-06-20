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
#include "ops/ops_func_impl/binary_cross_entropy_with_logits_backward.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
struct BCEWithLogitsBackwardParams {
  ShapeVector grad_output_shape;
  TypePtr grad_output_type;
  ShapeVector input_shape;
  TypePtr input_type;
  ShapeVector target_shape;
  TypePtr target_type;
  ShapeVector weight_shape;
  TypePtr weight_type;
  ShapeVector posWeight_shape;
  TypePtr posWeight_type;
  ValuePtr reduction;
  ShapeVector output_shape;
  TypePtr output_type;
};

class TestBCEWithLogitsBackward : public TestOps, public testing::WithParamInterface<BCEWithLogitsBackwardParams> {};

TEST_P(TestBCEWithLogitsBackward, dyn_shape) {
  const auto &param = GetParam();
  auto grad_output = std::make_shared<abstract::AbstractTensor>(param.grad_output_type, param.grad_output_shape);
  auto input = std::make_shared<abstract::AbstractTensor>(param.input_type, param.input_shape);
  auto target = std::make_shared<abstract::AbstractTensor>(param.target_type, param.target_shape);
  auto weight = std::make_shared<abstract::AbstractTensor>(param.weight_type, param.weight_shape);
  auto posWight = std::make_shared<abstract::AbstractTensor>(param.posWeight_type, param.posWeight_shape);
  auto reduction = param.reduction->ToAbstract();

  auto expect_shape = std::make_shared<abstract::Shape>(param.output_shape);
  auto expect_dtype = std::make_shared<TensorType>(param.output_type);

  BinaryCrossEntropyWithLogitsBackwardFuncImpl binary_cross_entropy_with_logits_func_impl;
  auto prim = std::make_shared<Primitive>("BinaryCrossEntropyWithLogitsBackward");
  auto out_dtype = binary_cross_entropy_with_logits_func_impl.InferType(
    prim, {grad_output, input, target, weight, posWight, reduction});
  ASSERT_TRUE(*out_dtype == *expect_dtype);
  auto out_shape = binary_cross_entropy_with_logits_func_impl.InferShape(
    prim, {grad_output, input, target, weight, posWight, reduction});
  ASSERT_TRUE(*out_shape == *expect_shape);
}
// enum Reduction : int64_t {REDUCTION_SUM = 0,MEAN = 1,NONE = 2,};
INSTANTIATE_TEST_CASE_P(TestBCEWithLogitsBackward, TestBCEWithLogitsBackward,
                        testing::Values(BCEWithLogitsBackwardParams{{3, 4, 5},
                                                                    kFloat32,
                                                                    {3, 4, 5},
                                                                    kFloat32,
                                                                    {3, 4, 5},
                                                                    kFloat32,
                                                                    {3, 4, 5},
                                                                    kFloat32,
                                                                    {3, 4, 5},
                                                                    kFloat32,
                                                                    CreateScalar<int64_t>(2),
                                                                    {3, 4, 5},
                                                                    kFloat32},
                                        BCEWithLogitsBackwardParams{{-1, -1, -1},
                                                                    kFloat32,
                                                                    {-1, -1, -1},
                                                                    kFloat32,
                                                                    {-1, -1, -1},
                                                                    kFloat32,
                                                                    {-1, -1, -1},
                                                                    kFloat32,
                                                                    {-1, -1, -1},
                                                                    kFloat32,
                                                                    CreateScalar<int64_t>(2),
                                                                    {-1, -1, -1},
                                                                    kFloat32},
                                        BCEWithLogitsBackwardParams{{-2},
                                                                    kFloat32,
                                                                    {-2},
                                                                    kFloat32,
                                                                    {-2},
                                                                    kFloat32,
                                                                    {-2},
                                                                    kFloat32,
                                                                    {-2},
                                                                    kFloat32,
                                                                    CreateScalar<int64_t>(2),
                                                                    {-2},
                                                                    kFloat32},
                                        BCEWithLogitsBackwardParams{{3, 4, 5},
                                                                    kFloat32,
                                                                    {3, 4, 5},
                                                                    kFloat32,
                                                                    {3, 4, 5},
                                                                    kFloat16,
                                                                    {3, 4, 5},
                                                                    kFloat32,
                                                                    {3, 4, 5},
                                                                    kFloat32,
                                                                    CreateScalar<int64_t>(1),
                                                                    {3, 4, 5},
                                                                    kFloat16},
                                        BCEWithLogitsBackwardParams{{-1, -1, -1},
                                                                    kFloat32,
                                                                    {-1, -1, -1},
                                                                    kFloat32,
                                                                    {-1, -1, -1},
                                                                    kFloat32,
                                                                    {-1, -1, -1},
                                                                    kFloat32,
                                                                    {-1, -1, -1},
                                                                    kFloat32,
                                                                    CreateScalar<int64_t>(1),
                                                                    {-1, -1, -1},
                                                                    kFloat32},
                                        BCEWithLogitsBackwardParams{{-2},
                                                                    kFloat32,
                                                                    {-2},
                                                                    kFloat32,
                                                                    {-2},
                                                                    kFloat32,
                                                                    {-2},
                                                                    kFloat32,
                                                                    {-2},
                                                                    kFloat32,
                                                                    CreateScalar<int64_t>(1),
                                                                    {-2},
                                                                    kFloat32},
                                        BCEWithLogitsBackwardParams{{3, 4, 5},
                                                                    kFloat32,
                                                                    {3, 4, 5},
                                                                    kFloat32,
                                                                    {3, 4, 5},
                                                                    kFloat32,
                                                                    {3, 4, 5},
                                                                    kFloat32,
                                                                    {3, 4, 5},
                                                                    kFloat32,
                                                                    CreateScalar<int64_t>(0),
                                                                    {3, 4, 5},
                                                                    kFloat32},
                                        BCEWithLogitsBackwardParams{{-1, -1, -1},
                                                                    kFloat32,
                                                                    {-1, -1, -1},
                                                                    kFloat32,
                                                                    {-1, -1, -1},
                                                                    kFloat32,
                                                                    {-1, -1, -1},
                                                                    kFloat32,
                                                                    {-1, -1, -1},
                                                                    kFloat32,
                                                                    CreateScalar<int64_t>(0),
                                                                    {-1, -1, -1},
                                                                    kFloat32},
                                        BCEWithLogitsBackwardParams{{-2},
                                                                    kFloat32,
                                                                    {-2},
                                                                    kFloat32,
                                                                    {-2},
                                                                    kFloat32,
                                                                    {-2},
                                                                    kFloat32,
                                                                    {-2},
                                                                    kFloat32,
                                                                    CreateScalar<int64_t>(0),
                                                                    {-2},
                                                                    kFloat32}));
}  // namespace ops
}  // namespace mindspore

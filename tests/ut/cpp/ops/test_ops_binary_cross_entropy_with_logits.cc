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
#include "ops/ops_func_impl/binary_cross_entropy_with_logits.h"
#include "ops/test_value_utils.h"
#include "ops/test_ops_cmp_utils.h"

namespace mindspore {
namespace ops {
struct BCEWithLogitsParams {
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

class TestBCEWithLogits : public TestOps, public testing::WithParamInterface<BCEWithLogitsParams> {};

TEST_P(TestBCEWithLogits, dyn_shape) {
  const auto &param = GetParam();
  auto input = std::make_shared<abstract::AbstractTensor>(param.input_type, param.input_shape);
  auto target = std::make_shared<abstract::AbstractTensor>(param.target_type, param.target_shape);
  auto weight = std::make_shared<abstract::AbstractTensor>(param.weight_type, param.weight_shape);
  auto posWight = std::make_shared<abstract::AbstractTensor>(param.posWeight_type, param.posWeight_shape);
  auto reduction = param.reduction->ToAbstract();

  auto expect_shape = std::make_shared<abstract::Shape>(param.output_shape);
  auto expect_dtype = std::make_shared<TensorType>(param.output_type);

  BCEWithLogitsLossFuncImpl binary_cross_entropy_with_logits_func_impl;
  auto prim = std::make_shared<Primitive>("BinaryCrossEntropyWithLogits");
  auto out_dtype =
    binary_cross_entropy_with_logits_func_impl.InferType(prim, {input, target, weight, posWight, reduction});
  ASSERT_TRUE(*out_dtype == *expect_dtype);
  auto out_shape =
    binary_cross_entropy_with_logits_func_impl.InferShape(prim, {input, target, weight, posWight, reduction});
  ASSERT_TRUE(*out_shape == *expect_shape);
}

class TestBCEWithLogitsSimpleInfer : public TestOps, public testing::WithParamInterface<BCEWithLogitsParams> {};

TEST_P(TestBCEWithLogitsSimpleInfer, dyn_shape) {
  const auto &param = GetParam();
  auto input = std::make_shared<tensor::BaseTensor>(param.input_type->type_id(), param.input_shape);
  auto target = std::make_shared<tensor::BaseTensor>(param.target_type->type_id(), param.target_shape);
  auto weight = std::make_shared<tensor::BaseTensor>(param.weight_type->type_id(), param.weight_shape);
  auto posWight = std::make_shared<tensor::BaseTensor>(param.posWeight_type->type_id(), param.posWeight_shape);
  ValuePtrList input_values;
  input_values.push_back(std::move(input));
  input_values.push_back(std::move(target));
  input_values.push_back(std::move(weight));
  input_values.push_back(std::move(posWight));
  input_values.push_back(std::move(param.reduction));

  BCEWithLogitsLossFuncImpl binary_cross_entropy_with_logits_func_impl;
  auto prim = std::make_shared<Primitive>("BinaryCrossEntropyWithLogits");

  auto expect_shape = ShapeArray{param.output_shape};
  auto expect_type = TypePtrList{param.output_type};

  auto output_shape = binary_cross_entropy_with_logits_func_impl.InferShape(prim, input_values);
  auto output_type = binary_cross_entropy_with_logits_func_impl.InferType(prim, input_values);

  ShapeCompare(output_shape, expect_shape);
  TypeCompare(output_type, expect_type);
}

// enum Reduction : int64_t {REDUCTION_SUM = 0,MEAN = 1,NONE = 2,};
INSTANTIATE_TEST_CASE_P(
  TestBCEWithLogits, TestBCEWithLogits,
  testing::Values(
    BCEWithLogitsParams{{3, 4, 5},
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
    BCEWithLogitsParams{{-1, -1, -1},
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
    BCEWithLogitsParams{
      {-2}, kFloat32, {-2}, kFloat32, {-2}, kFloat32, {-2}, kFloat32, CreateScalar<int64_t>(2), {-2}, kFloat32},
    BCEWithLogitsParams{{3, 4, 5},
                        kFloat32,
                        {3, 4, 5},
                        kFloat16,
                        {3, 4, 5},
                        kFloat32,
                        {3, 4, 5},
                        kFloat32,
                        CreateScalar<int64_t>(1),
                        {},
                        kFloat16},
    BCEWithLogitsParams{{-1, -1, -1},
                        kFloat32,
                        {-1, -1, -1},
                        kFloat16,
                        {-1, -1, -1},
                        kFloat32,
                        {-1, -1, -1},
                        kFloat32,
                        CreateScalar<int64_t>(1),
                        {},
                        kFloat16},
    BCEWithLogitsParams{
      {-2}, kFloat32, {-2}, kFloat16, {-2}, kFloat32, {-2}, kFloat32, CreateScalar<int64_t>(1), {}, kFloat16},
    BCEWithLogitsParams{{3, 4, 5},
                        kFloat32,
                        {3, 4, 5},
                        kFloat32,
                        {3, 4, 5},
                        kFloat32,
                        {3, 4, 5},
                        kFloat32,
                        CreateScalar<int64_t>(0),
                        {},
                        kFloat32},
    BCEWithLogitsParams{{-1, -1, -1},
                        kFloat32,
                        {-1, -1, -1},
                        kFloat16,
                        {-1, -1, -1},
                        kFloat32,
                        {-1, -1, -1},
                        kFloat32,
                        CreateScalar<int64_t>(0),
                        {},
                        kFloat16},
    BCEWithLogitsParams{
      {-2}, kFloat32, {-2}, kFloat16, {-2}, kFloat32, {-2}, kFloat32, CreateScalar<int64_t>(0), {}, kFloat16}));

INSTANTIATE_TEST_CASE_P(TestBCEWithLogitsSimpleInfer, TestBCEWithLogitsSimpleInfer,
                        testing::Values(BCEWithLogitsParams{{3, 4, 5},
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
                                        BCEWithLogitsParams{{3, 4, 5},
                                                            kFloat32,
                                                            {3, 4, 5},
                                                            kFloat16,
                                                            {3, 4, 5},
                                                            kFloat32,
                                                            {3, 4, 5},
                                                            kFloat32,
                                                            CreateScalar<int64_t>(1),
                                                            {},
                                                            kFloat16},
                                        BCEWithLogitsParams{{3, 4, 5},
                                                            kFloat32,
                                                            {3, 4, 5},
                                                            kFloat32,
                                                            {3, 4, 5},
                                                            kFloat32,
                                                            {3, 4, 5},
                                                            kFloat32,
                                                            CreateScalar<int64_t>(0),
                                                            {},
                                                            kFloat32}));
}  // namespace ops
}  // namespace mindspore

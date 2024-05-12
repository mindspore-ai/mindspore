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
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"
#include "ops/ops_frontend_func_impl.h"
#include "ops/ops_func_impl/masked_select.h"
#include "ops/test_value_utils.h"
#include "ops/test_ops_cmp_utils.h"

namespace mindspore {
namespace ops {
struct MaskedSelectShapeParams {
  ShapeVector input_shape;
  TypePtr input_type;
  ShapeVector mask_shape;
  TypePtr mask_type;
};

class TestMaskedSelect : public TestOps, public testing::WithParamInterface<MaskedSelectShapeParams> {};
class TestMaskedSelectException : public TestOps, public testing::WithParamInterface<MaskedSelectShapeParams> {};
class TestMaskedSelectFrontend : public TestOps, public testing::WithParamInterface<MaskedSelectShapeParams> {};
class TestMaskedSelectSimpleInfer : public TestOps, public testing::WithParamInterface<MaskedSelectShapeParams> {};

TEST_P(TestMaskedSelect, dyn_shape) {
  const auto &param = GetParam();
  auto input = std::make_shared<abstract::AbstractTensor>(param.input_type, param.input_shape);
  auto mask = std::make_shared<abstract::AbstractTensor>(param.mask_type, param.mask_shape);
  int64_t num = std::accumulate(param.input_shape.begin(), param.input_shape.end(), 1, std::multiplies<int64_t>());
  auto expect_shape = std::make_shared<abstract::TensorShape>(ShapeVector({num}));;
  auto expect_type = input->GetType();
  MaskedSelectFuncImpl masked_select_func_impl;
  auto prim = std::make_shared<Primitive>("MaskedSelect");

  auto out_dtype = masked_select_func_impl.InferType(prim, {input, mask});
  ASSERT_TRUE(*out_dtype == *expect_type);
  auto out_shape = masked_select_func_impl.InferShape(prim, {input, mask});
  ASSERT_TRUE(*out_shape == *expect_shape);
}

TEST_P(TestMaskedSelectSimpleInfer, simple_infer) {
  const auto &param = GetParam();
  auto input = std::make_shared<tensor::BaseTensor>(param.input_type->type_id(), param.input_shape);
  auto mask = std::make_shared<tensor::BaseTensor>(param.mask_type->type_id(), param.mask_shape);
  int64_t num = std::accumulate(param.input_shape.begin(), param.input_shape.end(), 1, std::multiplies<int64_t>());
  auto expect_shape = ShapeArray{{num}};
  auto expect_type = TypePtrList{param.input_type};
  MaskedSelectFuncImpl masked_select_func_impl;
  auto prim = std::make_shared<Primitive>("MaskedSelect");
  ValuePtrList input_values;
  input_values.push_back(std::move(input));
  input_values.push_back(std::move(mask));
  auto out_dtype = masked_select_func_impl.InferType(prim, input_values);
  TypeCompare(out_dtype, expect_type);
  auto out_shape = masked_select_func_impl.InferShape(prim, input_values);
  ShapeCompare(out_shape, expect_shape);
}

TEST_P(TestMaskedSelectFrontend, dyn_shape) {
  const auto &param = GetParam();
  auto input = std::make_shared<abstract::AbstractTensor>(param.input_type, param.input_shape);
  auto mask = std::make_shared<abstract::AbstractTensor>(param.mask_type, param.mask_shape);
  auto masked_select_frontend_impl = GetOpFrontendFuncImplPtr("MaskedSelect");
  ASSERT_NE(masked_select_frontend_impl, nullptr);
  auto expect_shape = std::make_shared<abstract::TensorShape>(ShapeVector({abstract::Shape::kShapeDimAny}));
  auto expect_type = input->GetType();
  auto prim = std::make_shared<Primitive>("MaskedSelect");

  auto infer_shape_type = masked_select_frontend_impl->InferAbstract(prim, {input, mask});
  ASSERT_NE(infer_shape_type, nullptr);
  auto out_shape = infer_shape_type->GetShape();
  ASSERT_NE(out_shape, nullptr);
  ASSERT_TRUE(*out_shape == *expect_shape);
  auto out_dtype = infer_shape_type->GetType();
  ASSERT_NE(out_dtype, nullptr);
  ASSERT_TRUE(*out_dtype == *expect_type);  
}

TEST_P(TestMaskedSelectException, exception) {
  const auto &param = GetParam();
  auto input = std::make_shared<abstract::AbstractTensor>(param.input_type, param.input_shape);
  auto mask = std::make_shared<abstract::AbstractTensor>(param.mask_type, param.mask_shape);
  auto expect_shape = input->GetShape();
  auto expect_type = input->GetType();

  MaskedSelectFuncImpl masked_select_func_impl;
  auto prim = std::make_shared<Primitive>("MaskedSelect");

  try {
    auto out_dtype = masked_select_func_impl.InferType(prim, {input, mask});
    auto out_shape = masked_select_func_impl.InferShape(prim, {input, mask});
  } catch (std::exception &e) {
    ASSERT_TRUE(true);
    return;
  }
  ASSERT_TRUE(false);
}

INSTANTIATE_TEST_CASE_P(
  TestMaskedSelect, TestMaskedSelect,
  testing::Values(MaskedSelectShapeParams{{4, 3}, kInt8, {4, 3}, kBool},
                  MaskedSelectShapeParams{{2, 3}, kInt16, {2, 3}, kBool},
                  MaskedSelectShapeParams{{3, 4, 5}, kInt32, {4, 5}, kBool},
                  MaskedSelectShapeParams{{3, 3}, kInt64, {3, 1}, kBool},
                  MaskedSelectShapeParams{{3, 4, 5}, kUInt8, {3, 4, 5}, kBool},
                  MaskedSelectShapeParams{{3, 4, 5}, kFloat16, {3, 4, 5}, kBool},
                  MaskedSelectShapeParams{{3, 4, 5}, kFloat32, {3, 4, 5}, kBool}));

INSTANTIATE_TEST_CASE_P(
  TestMaskedSelectFrontend, TestMaskedSelectFrontend,
  testing::Values(MaskedSelectShapeParams{{-2}, kFloat64, {-2}, kBool},
                  MaskedSelectShapeParams{{-1, -1, -1}, kBFloat16, {-1, -1, -1}, kBool}));

INSTANTIATE_TEST_CASE_P(
  TestMaskedSelectException, TestMaskedSelectException,
  testing::Values(MaskedSelectShapeParams{{4, 3}, kInt8, {3, 3}, kInt32},
                  MaskedSelectShapeParams{{3, 4, 5}, kUInt32, {6}, kBool},
                  MaskedSelectShapeParams{{2, 3, 4}, kUInt16, {3}, kBool}));

INSTANTIATE_TEST_CASE_P(
  TestMaskedSelectSimpleInfer, TestMaskedSelectSimpleInfer,
  testing::Values(MaskedSelectShapeParams{{4, 3}, kInt8, {4, 3}, kBool},
                  MaskedSelectShapeParams{{2, 3}, kInt16, {2, 3}, kBool},
                  MaskedSelectShapeParams{{3, 4, 5}, kInt32, {4, 5}, kBool},
                  MaskedSelectShapeParams{{3, 3}, kInt64, {3, 1}, kBool},
                  MaskedSelectShapeParams{{3, 4, 5}, kUInt8, {3, 4, 5}, kBool},
                  MaskedSelectShapeParams{{3, 4, 5}, kFloat16, {3, 4, 5}, kBool},
                  MaskedSelectShapeParams{{3, 4, 5}, kFloat32, {3, 4, 5}, kBool}));

}  // namespace ops
}  // namespace mindspore

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
#include "ops/ops_func_impl/masked_select_ext.h"
#include "ops/ops_frontend_func_impl.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
struct MaskedSelectExtShapeParams {
  ShapeVector input_shape;
  TypePtr input_type;
  ShapeVector mask_shape;
  TypePtr mask_type;
};

class TestMaskedSelectExt : public TestOps, public testing::WithParamInterface<MaskedSelectExtShapeParams> {};
class TestMaskedSelectExtException : public TestOps, public testing::WithParamInterface<MaskedSelectExtShapeParams> {};
class TestMaskedSelectExtFrontend : public TestOps, public testing::WithParamInterface<MaskedSelectExtShapeParams> {};

TEST_P(TestMaskedSelectExt, dyn_shape) {
  const auto &param = GetParam();
  auto input = std::make_shared<abstract::AbstractTensor>(param.input_type, param.input_shape);
  auto mask = std::make_shared<abstract::AbstractTensor>(param.mask_type, param.mask_shape);
  int64_t num = std::accumulate(param.input_shape.begin(), param.input_shape.end(), 1, std::multiplies<int64_t>());
  auto expect_shape = std::make_shared<abstract::TensorShape>(ShapeVector({num}));;
  auto expect_type = input->GetType();
  MaskedSelectExtFuncImpl masked_select_ext_func_impl;
  auto prim = std::make_shared<Primitive>("MaskedSelectExt");

  auto out_dtype = masked_select_ext_func_impl.InferType(prim, {input, mask});
  ASSERT_TRUE(*out_dtype == *expect_type);
  auto out_shape = masked_select_ext_func_impl.InferShape(prim, {input, mask});
  ASSERT_TRUE(*out_shape == *expect_shape);
}

TEST_P(TestMaskedSelectExtFrontend, dyn_shape) {
  const auto &param = GetParam();
  auto input = std::make_shared<abstract::AbstractTensor>(param.input_type, param.input_shape);
  auto mask = std::make_shared<abstract::AbstractTensor>(param.mask_type, param.mask_shape);
  auto masked_select_ext_frontend_impl = GetOpFrontendFuncImplPtr("MaskedSelectExt");
  ASSERT_NE(masked_select_ext_frontend_impl, nullptr);
  auto expect_shape = std::make_shared<abstract::TensorShape>(ShapeVector({abstract::Shape::kShapeDimAny}));
  auto expect_type = input->GetType();
  auto prim = std::make_shared<Primitive>("MaskedSelectExt");

  auto infer_shape_type = masked_select_ext_frontend_impl->InferAbstract(prim, {input, mask});
  ASSERT_NE(infer_shape_type, nullptr);
  auto out_shape = infer_shape_type->GetShape();
  ASSERT_NE(out_shape, nullptr);
  ASSERT_TRUE(*out_shape == *expect_shape);
  auto out_dtype = infer_shape_type->GetType();
  ASSERT_NE(out_dtype, nullptr);
  ASSERT_TRUE(*out_dtype == *expect_type);  
}

TEST_P(TestMaskedSelectExtException, exception) {
  const auto &param = GetParam();
  auto input = std::make_shared<abstract::AbstractTensor>(param.input_type, param.input_shape);
  auto mask = std::make_shared<abstract::AbstractTensor>(param.mask_type, param.mask_shape);
  auto expect_shape = input->GetShape();
  auto expect_type = input->GetType();

  MaskedSelectExtFuncImpl masked_select_ext_func_impl;
  auto prim = std::make_shared<Primitive>("MaskedSelectExt");

  try {
    auto out_dtype = masked_select_ext_func_impl.InferType(prim, {input, mask});
    auto out_shape = masked_select_ext_func_impl.InferShape(prim, {input, mask});
  } catch (std::exception &e) {
    ASSERT_TRUE(true);
    return;
  }
  ASSERT_TRUE(false);
}

INSTANTIATE_TEST_CASE_P(
  TestMaskedSelectExt, TestMaskedSelectExt,
  testing::Values(MaskedSelectExtShapeParams{{4, 3}, kInt8, {4, 3}, kBool},
                  MaskedSelectExtShapeParams{{2, 3}, kInt16, {2, 3}, kBool},
                  MaskedSelectExtShapeParams{{3, 4, 5}, kInt32, {4, 5}, kBool},
                  MaskedSelectExtShapeParams{{3, 3}, kInt64, {3, 1}, kBool},
                  MaskedSelectExtShapeParams{{3, 4, 5}, kUInt8, {3, 4, 5}, kBool},
                  MaskedSelectExtShapeParams{{3, 4, 5}, kFloat16, {3, 4, 5}, kBool},
                  MaskedSelectExtShapeParams{{3, 4, 5}, kFloat32, {3, 4, 5}, kBool}));

INSTANTIATE_TEST_CASE_P(
  TestMaskedSelectExtFrontend, TestMaskedSelectExtFrontend,
  testing::Values(MaskedSelectExtShapeParams{{-2}, kFloat64, {-2}, kBool},
                  MaskedSelectExtShapeParams{{-1, -1, -1}, kBFloat16, {-1, -1, -1}, kBool}));

INSTANTIATE_TEST_CASE_P(
  TestMaskedSelectExtException, TestMaskedSelectExtException,
  testing::Values(MaskedSelectExtShapeParams{{4, 3}, kInt8, {4, 3}, kInt32},
                  MaskedSelectExtShapeParams{{3, 4, 5}, kUInt32, {3, 4, 5}, kBool},
                  MaskedSelectExtShapeParams{{2, 3, 4}, kUInt16, {2, 3, 4}, kBool}));
}  // namespace ops
}  // namespace mindspore

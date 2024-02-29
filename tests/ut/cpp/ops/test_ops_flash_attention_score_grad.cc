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
#include "ops/auto_generate/gen_ops_name.h"
#include "ops/ops_func_impl/flash_attention_score_grad.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
struct FlashAttentionScoreGradParams {
  ShapeVector qkv_dy_shape;
  TypePtr qkv_dy_dtype;
  ShapeVector pse_shape;
  TypePtr pse_dtype;
  ShapeVector out1_shape;
  TypePtr out1_type;
  ShapeVector out2_shape;
  TypePtr out2_type;
  ShapeVector out3_shape;
  TypePtr out3_type;
  ShapeVector out4_shape;
  TypePtr out4_type;
};

class TestFlashAttentionScoreGrad : public TestOps,
                                    public testing::WithParamInterface<FlashAttentionScoreGradParams> {};

TEST_P(TestFlashAttentionScoreGrad, dyn_shape) {
  const auto &param = GetParam();
  auto flash_attention_score_grad_func_impl = std::make_shared<FlashAttentionScoreGradFuncImpl>();
  auto prim = std::make_shared<Primitive>("FlashAttentionScoreGrad");

  auto qkv_dy = std::make_shared<abstract::AbstractTensor>(param.qkv_dy_dtype, param.qkv_dy_shape);
  ASSERT_NE(qkv_dy, nullptr);
  auto pse = std::make_shared<abstract::AbstractTensor>(param.pse_dtype, param.pse_shape);
  ASSERT_NE(pse, nullptr);

  auto expect_out1_shape = std::make_shared<abstract::Shape>(param.out1_shape);
  auto expect_out2_shape = std::make_shared<abstract::Shape>(param.out2_shape);
  auto expect_out3_shape = std::make_shared<abstract::Shape>(param.out3_shape);
  auto expect_out4_shape = std::make_shared<abstract::Shape>(param.out4_shape);
  ASSERT_NE(expect_out1_shape, nullptr);
  ASSERT_NE(expect_out2_shape, nullptr);
  ASSERT_NE(expect_out3_shape, nullptr);
  ASSERT_NE(expect_out4_shape, nullptr);
  auto expect_shape = std::make_shared<abstract::TupleShape>(
    abstract::BaseShapePtrList({expect_out1_shape, expect_out2_shape, expect_out3_shape, expect_out4_shape}));
  auto expect_out1_dtype = std::make_shared<TensorType>(param.out1_type)->cast<TypePtr>();
  auto expect_out2_dtype = std::make_shared<TensorType>(param.out2_type)->cast<TypePtr>();
  auto expect_out3_dtype = std::make_shared<TensorType>(param.out3_type)->cast<TypePtr>();
  auto expect_out4_dtype = std::make_shared<TensorType>(param.out4_type)->cast<TypePtr>();
  ASSERT_NE(expect_out1_dtype, nullptr);
  ASSERT_NE(expect_out2_dtype, nullptr);
  ASSERT_NE(expect_out3_dtype, nullptr);
  ASSERT_NE(expect_out4_dtype, nullptr);
  auto expect_dtype =
    std::make_shared<Tuple>(TypePtrList({expect_out1_dtype, expect_out2_dtype, expect_out3_dtype, expect_out4_dtype}));

  // execute
  auto input_none = std::make_shared<abstract::AbstractNone>();
  auto input_scalar = std::make_shared<abstract::AbstractScalar>();
  std::vector<AbstractBasePtr> input_args = {
    qkv_dy,       qkv_dy,       qkv_dy,       qkv_dy,       pse,          input_none,   input_none,  input_none,
    input_none,   input_none,   input_none,   input_none,   input_none,   input_none,   input_none,  input_scalar,
    input_scalar, input_scalar, input_scalar, input_scalar, input_scalar, input_scalar, input_scalar};
  auto out_shape = flash_attention_score_grad_func_impl->InferShape(prim, input_args);
  auto out_dtype = flash_attention_score_grad_func_impl->InferType(prim, input_args);

  // verify output
  ASSERT_NE(out_shape, nullptr);
  ASSERT_TRUE(*out_shape == *expect_shape);
  ASSERT_NE(out_dtype, nullptr);
  ASSERT_TRUE(*out_dtype == *expect_dtype);
}

INSTANTIATE_TEST_CASE_P(
  TestFlashAttentionScoreGradGroup, TestFlashAttentionScoreGrad,
  testing::Values(
    FlashAttentionScoreGradParams{
      {-2}, kFloat16, {-2}, kFloat16, {-2}, kFloat16, {-2}, kFloat16, {-2}, kFloat16, {-2}, kFloat16},
    FlashAttentionScoreGradParams{
      {-2}, kBFloat16, {-2}, kBFloat16, {-2}, kBFloat16, {-2}, kBFloat16, {-2}, kBFloat16, {-2}, kBFloat16},
    FlashAttentionScoreGradParams{{-1, -1, -1},
                                  kFloat16,
                                  {-1, -1, -1},
                                  kFloat16,
                                  {-1, -1, -1},
                                  kFloat16,
                                  {-1, -1, -1},
                                  kFloat16,
                                  {-1, -1, -1},
                                  kFloat16,
                                  {-1, -1, -1},
                                  kFloat16},
    FlashAttentionScoreGradParams{{-1, -1, -1},
                                  kBFloat16,
                                  {-1, -1, -1},
                                  kBFloat16,
                                  {-1, -1, -1},
                                  kBFloat16,
                                  {-1, -1, -1},
                                  kBFloat16,
                                  {-1, -1, -1},
                                  kBFloat16,
                                  {-1, -1, -1},
                                  kBFloat16},
    FlashAttentionScoreGradParams{{4, 6, 8},
                                  kFloat16,
                                  {4, 6, 8},
                                  kFloat16,
                                  {4, 6, 8},
                                  kFloat16,
                                  {4, 6, 8},
                                  kFloat16,
                                  {4, 6, 8},
                                  kFloat16,
                                  {4, 6, 8},
                                  kFloat16},
    FlashAttentionScoreGradParams{{4, 6, 8},
                                  kBFloat16,
                                  {4, 6, 8},
                                  kBFloat16,
                                  {4, 6, 8},
                                  kBFloat16,
                                  {4, 6, 8},
                                  kBFloat16,
                                  {4, 6, 8},
                                  kBFloat16,
                                  {4, 6, 8},
                                  kBFloat16}));

}  // namespace ops
}  // namespace mindspore

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
#include "ops/ops_func_impl/flash_attention_score.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
#define I64(x) (static_cast<int64_t>((x)))
struct FlashAttentionScoreParams {
  ShapeVector qkv_shape;
  TypePtr qkv_dtype;
  ValuePtr input_layout_value;
  ValuePtr actual_seq_qlen_value;
  ValuePtr actual_seq_kvlen_value;
  ValuePtr head_num_value;
  ShapeVector out1_shape;
  TypePtr out1_type;
  ShapeVector out2_shape;
  TypePtr out2_type;
  ShapeVector out3_shape;
  TypePtr out3_type;
  ShapeVector out4_shape;
  TypePtr out4_type;
};

class TestFlashAttentionScore : public TestOps, public testing::WithParamInterface<FlashAttentionScoreParams> {};

TEST_P(TestFlashAttentionScore, dyn_shape) {
  const auto &param = GetParam();
  auto flash_attention_score_func_impl = std::make_shared<FlashAttentionScoreFuncImpl>();
  auto prim = std::make_shared<Primitive>("FlashAttentionScore");
  auto none = std::make_shared<abstract::AbstractNone>();

  auto qkv = std::make_shared<abstract::AbstractTensor>(param.qkv_dtype, param.qkv_shape);
  ASSERT_NE(qkv, nullptr);
  abstract::AbstractBasePtr input_layout = nullptr;
  if (param.input_layout_value == nullptr) {
    input_layout = none;
  } else {
    input_layout = param.input_layout_value->ToAbstract();
  }
  abstract::AbstractBasePtr head_num = nullptr;
  if (param.head_num_value == nullptr) {
    head_num = none;
  } else {
    head_num = param.head_num_value->ToAbstract();
  }
  abstract::AbstractBasePtr actual_seq_qlen = nullptr;
  if (param.actual_seq_qlen_value == nullptr) {
    actual_seq_qlen = none;
  } else {
    actual_seq_qlen = param.actual_seq_qlen_value->ToAbstract();
  }
  abstract::AbstractBasePtr actual_seq_kvlen = nullptr;
  if (param.actual_seq_kvlen_value == nullptr) {
    actual_seq_kvlen = none;
  } else {
    actual_seq_kvlen = param.actual_seq_kvlen_value->ToAbstract();
  }
  MS_EXCEPTION_IF_NULL(input_layout);
  MS_EXCEPTION_IF_NULL(head_num);
  MS_EXCEPTION_IF_NULL(actual_seq_qlen);
  MS_EXCEPTION_IF_NULL(actual_seq_kvlen);

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
  auto expect_out1_dtype = std::make_shared<TensorType>(param.out1_type);
  auto expect_out2_dtype = std::make_shared<TensorType>(param.out2_type);
  auto expect_out3_dtype = std::make_shared<TensorType>(param.out3_type);
  auto expect_out4_dtype = std::make_shared<TensorType>(param.out4_type);
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
    qkv,          qkv,          qkv,          input_none,      input_none,
    input_none,   input_none,   input_none,   actual_seq_qlen, actual_seq_kvlen,
    head_num,     input_scalar, input_scalar, input_scalar,    input_scalar,
    input_scalar, input_layout, input_scalar};
  auto out_shape = flash_attention_score_func_impl->InferShape(prim, input_args);
  auto out_dtype = flash_attention_score_func_impl->InferType(prim, input_args);

  // verify output
  ASSERT_NE(out_shape, nullptr);
  ASSERT_TRUE(*out_shape == *expect_shape);
  ASSERT_NE(out_dtype, nullptr);
  ASSERT_TRUE(*out_dtype == *expect_dtype);
}

INSTANTIATE_TEST_CASE_P(TestFlashAttentionScoreGroup, TestFlashAttentionScore,
                        testing::Values(FlashAttentionScoreParams{{-2},
                                                                  kFloat16,
                                                                  CreateScalar<int64_t>(0),
                                                                  nullptr,
                                                                  nullptr,
                                                                  CreateScalar<int64_t>(1),
                                                                  {-1, -1, -1, 8},
                                                                  kFloat32,
                                                                  {-1, -1, -1, 8},
                                                                  kFloat32,
                                                                  {1},
                                                                  kFloat16,
                                                                  {-2},
                                                                  kFloat16},
                                        FlashAttentionScoreParams{{-2},
                                                                  kBFloat16,
                                                                  CreateScalar<int64_t>(0),
                                                                  nullptr,
                                                                  nullptr,
                                                                  CreateScalar<int64_t>(1),
                                                                  {-1, -1, -1, 8},
                                                                  kFloat32,
                                                                  {-1, -1, -1, 8},
                                                                  kFloat32,
                                                                  {1},
                                                                  kBFloat16,
                                                                  {-2},
                                                                  kBFloat16},
                                        FlashAttentionScoreParams{{-1, -1, -1},
                                                                  kFloat16,
                                                                  CreateScalar<int64_t>(0),
                                                                  nullptr,
                                                                  nullptr,
                                                                  CreateScalar<int64_t>(1),
                                                                  {-1, 1, -1, 8},
                                                                  kFloat32,
                                                                  {-1, 1, -1, 8},
                                                                  kFloat32,
                                                                  {1},
                                                                  kFloat16,
                                                                  {-1, -1, -1},
                                                                  kFloat16},
                                        FlashAttentionScoreParams{{4, 6, 8},
                                                                  kFloat16,
                                                                  nullptr,
                                                                  nullptr,
                                                                  nullptr,
                                                                  CreateScalar<int64_t>(1),
                                                                  {-2},
                                                                  kFloat32,
                                                                  {-2},
                                                                  kFloat32,
                                                                  {1},
                                                                  kFloat16,
                                                                  {4, 6, 8},
                                                                  kFloat16},
                                        FlashAttentionScoreParams{{4, 6, 8},
                                                                  kFloat16,
                                                                  CreateScalar<int64_t>(0),
                                                                  nullptr,
                                                                  nullptr,
                                                                  nullptr,
                                                                  {4, -1, 6, 8},
                                                                  kFloat32,
                                                                  {4, -1, 6, 8},
                                                                  kFloat32,
                                                                  {1},
                                                                  kFloat16,
                                                                  {4, 6, 8},
                                                                  kFloat16},
                                        FlashAttentionScoreParams{{4, 6, 8},
                                                                  kFloat16,
                                                                  CreateScalar<int64_t>(0),
                                                                  nullptr,
                                                                  nullptr,
                                                                  CreateScalar<int64_t>(2),
                                                                  {4, 2, 6, 8},
                                                                  kFloat32,
                                                                  {4, 2, 6, 8},
                                                                  kFloat32,
                                                                  {1},
                                                                  kFloat16,
                                                                  {4, 6, 8},
                                                                  kFloat16},
                                        FlashAttentionScoreParams{{4, 2, 8, 10},
                                                                  kFloat16,
                                                                  CreateScalar<int64_t>(1),
                                                                  nullptr,
                                                                  nullptr,
                                                                  CreateScalar<int64_t>(2),
                                                                  {4, 2, 8, 8},
                                                                  kFloat32,
                                                                  {4, 2, 8, 8},
                                                                  kFloat32,
                                                                  {1},
                                                                  kFloat16,
                                                                  {4, 2, 8, 10},
                                                                  kFloat16},
                                        FlashAttentionScoreParams{{4, 2, 8},
                                                                  kFloat16,
                                                                  CreateScalar<int64_t>(2),
                                                                  nullptr,
                                                                  nullptr,
                                                                  CreateScalar<int64_t>(2),
                                                                  {2, 2, 4, 8},
                                                                  kFloat32,
                                                                  {2, 2, 4, 8},
                                                                  kFloat32,
                                                                  {1},
                                                                  kFloat16,
                                                                  {4, 2, 8},
                                                                  kFloat16},
                                        FlashAttentionScoreParams{{4, 6, 2, 10},
                                                                  kFloat16,
                                                                  CreateScalar<int64_t>(3),
                                                                  nullptr,
                                                                  nullptr,
                                                                  CreateScalar<int64_t>(2),
                                                                  {4, 2, 6, 8},
                                                                  kFloat32,
                                                                  {4, 2, 6, 8},
                                                                  kFloat32,
                                                                  {1},
                                                                  kFloat16,
                                                                  {4, 6, 2, 10},
                                                                  kFloat16},
                                        FlashAttentionScoreParams{{4, 6, 10},
                                                                  kFloat16,
                                                                  CreateScalar<int64_t>(4),
                                                                  CreateTuple({I64(4)}),
                                                                  CreateTuple({I64(4)}),
                                                                  CreateScalar<int64_t>(2),
                                                                  {4, 6, 8},
                                                                  kFloat32,
                                                                  {4, 6, 8},
                                                                  kFloat32,
                                                                  {1},
                                                                  kFloat16,
                                                                  {4, 6, 10},
                                                                  kFloat16}));

}  // namespace ops
}  // namespace mindspore

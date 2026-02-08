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

#include "ops/test_ops.h"
#include "common/common_test.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "primitive/op_name.h"
#include "infer/ops_func_impl/prompt_flash_attention.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
#define I64(x) (static_cast<int64_t>((x)))
struct PromptFlashAttentionParams {
  ShapeVector qkv_shape;
  TypePtr qkv_dtype;
  ValuePtr input_layout_value;
  ValuePtr actual_seq_qlen_value;
  ValuePtr actual_seq_kvlen_value;
  ValuePtr num_heads_value;
  ShapeVector out_shape;
  TypePtr out_type;
};

class TestPromptFlashAttention : public TestOps, public testing::WithParamInterface<PromptFlashAttentionParams> {};

TEST_P(TestPromptFlashAttention, dyn_shape) {
  const auto &param = GetParam();
  auto prompt_flash_attention_func_impl = std::make_shared<PromptFlashAttentionFuncImpl>();
  auto prim = std::make_shared<Primitive>("PromptFlashAttention");
  auto none = std::make_shared<abstract::AbstractNone>();

  auto qkv = std::make_shared<abstract::AbstractTensor>(param.qkv_dtype, param.qkv_shape);
  ASSERT_NE(qkv, nullptr);
  abstract::AbstractBasePtr input_layout = nullptr;
  if (param.input_layout_value == nullptr) {
    input_layout = none;
  } else {
    input_layout = param.input_layout_value->ToAbstract();
  }
  abstract::AbstractBasePtr num_heads = nullptr;
  if (param.num_heads_value == nullptr) {
    num_heads = none;
  } else {
    num_heads = param.num_heads_value->ToAbstract();
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
  MS_EXCEPTION_IF_NULL(num_heads);
  MS_EXCEPTION_IF_NULL(actual_seq_qlen);
  MS_EXCEPTION_IF_NULL(actual_seq_kvlen);

  auto expect_out_shape = std::make_shared<abstract::Shape>(param.out_shape);
  auto expect_out_dtype = std::make_shared<TensorType>(param.out_type);
  ASSERT_NE(expect_out_shape, nullptr);
  ASSERT_NE(expect_out_dtype, nullptr);

  // execute
  auto input_none = std::make_shared<abstract::AbstractNone>();
  auto input_scalar = std::make_shared<abstract::AbstractScalar>();
  std::vector<AbstractBasePtr> input_args = {
    qkv,          qkv,          qkv,          input_none,   actual_seq_qlen, actual_seq_kvlen, input_none,
    input_none,   input_none,   input_none,   input_none,   input_none,      num_heads,        input_scalar,
    input_scalar, input_scalar, input_layout, input_scalar, input_scalar,    input_scalar};
  auto out_shape = prompt_flash_attention_func_impl->InferShape(prim, input_args);
  auto out_dtype = prompt_flash_attention_func_impl->InferType(prim, input_args);

  // verify output
  ASSERT_NE(out_shape, nullptr);
  ASSERT_TRUE(*out_shape == *expect_out_shape);
  ASSERT_NE(out_dtype, nullptr);
  ASSERT_TRUE(*out_dtype == *expect_out_dtype);
}

INSTANTIATE_TEST_CASE_P(TestPromptFlashAttentionGroup, TestPromptFlashAttention,
                        testing::Values(PromptFlashAttentionParams{{-2},
                                                                  kFloat16,
                                                                  CreateScalar<int64_t>(0),
                                                                  nullptr,
                                                                  nullptr,
                                                                  CreateScalar<int64_t>(1),
                                                                  {-2},
                                                                  kFloat16},
                                        PromptFlashAttentionParams{{-2},
                                                                  kBFloat16,
                                                                  CreateScalar<int64_t>(0),
                                                                  nullptr,
                                                                  nullptr,
                                                                  CreateScalar<int64_t>(1),
                                                                  {-2},
                                                                  kBFloat16},
                                        PromptFlashAttentionParams{{-1, -1, -1},
                                                                  kFloat16,
                                                                  CreateScalar<int64_t>(0),
                                                                  nullptr,
                                                                  nullptr,
                                                                  CreateScalar<int64_t>(1),
                                                                  {-2},
                                                                  kFloat16},
                                        PromptFlashAttentionParams{{4, 6, 8},
                                                                  kBFloat16,
                                                                  CreateScalar<int64_t>(0),
                                                                  nullptr,
                                                                  nullptr,
                                                                  CreateScalar<int64_t>(2),
                                                                  {-2},
                                                                  kBFloat16},
                                        PromptFlashAttentionParams{{4, 2, 8, 10},
                                                                  kBFloat16,
                                                                  CreateScalar<int64_t>(1),
                                                                  nullptr,
                                                                  nullptr,
                                                                  CreateScalar<int64_t>(2),
                                                                  {-2},
                                                                  kBFloat16},
                                        PromptFlashAttentionParams{{4, 2, 8},
                                                                  kFloat16,
                                                                  CreateScalar<int64_t>(2),
                                                                  nullptr,
                                                                  nullptr,
                                                                  CreateScalar<int64_t>(2),
                                                                  {-2},
                                                                  kFloat16},
                                        PromptFlashAttentionParams{{4, 6, 2, 10},
                                                                  kFloat16,
                                                                  CreateScalar<int64_t>(3),
                                                                  nullptr,
                                                                  nullptr,
                                                                  CreateScalar<int64_t>(2),
                                                                  {-2},
                                                                  kFloat16},
                                        PromptFlashAttentionParams{{4, 6, 10},
                                                                  kFloat16,
                                                                  CreateScalar<int64_t>(4),
                                                                  CreateTuple({I64(4)}),
                                                                  CreateTuple({I64(4)}),
                                                                  CreateScalar<int64_t>(2),
                                                                  {-2},
                                                                  kFloat16}));

}  // namespace ops
}  // namespace mindspore

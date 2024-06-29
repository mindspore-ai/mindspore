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

#include "ops/ops_func_impl/paged_attention_mask.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr PagedAttentionMaskFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                    const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kPagedAttentionMaskInputsNum, primitive->name());

  auto query_shape_ptr = input_args[kPagedAttentionMaskInputQueryIndex]->GetShape();
  auto shape_element = query_shape_ptr->cast<abstract::ShapePtr>();
  return shape_element;
}

bool IsOptionalInputNone(const AbstractBasePtr &input) {
  MS_EXCEPTION_IF_NULL(input);
  return input->GetType()->type_id() == kMetaTypeNone;
}

TypePtr PagedAttentionMaskFuncImpl::InferType(const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args) const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  bool enable_infer_boost = ms_context->IsEnableInferBoost();
  std::set valid_types = {kFloat16};
  //  infer_boost support BF16 inputs
  if (enable_infer_boost) {
    (void)valid_types.emplace(kBFloat16);
  }
  auto op_name = primitive->name();
  std::map<std::string, TypePtr> types;

  (void)types.emplace("query", input_args[kPagedAttentionMaskInputQueryIndex]->GetType());
  (void)types.emplace("key_cache", input_args[kPagedAttentionMaskInputKeyCacheIndex]->GetType());
  (void)types.emplace("value_cache", input_args[kPagedAttentionMaskInputValueCacheIndex]->GetType());
  //  check alibi_mask dtype when alibi_mask is NOT None
  if (!IsOptionalInputNone(input_args[kPagedAttentionMaskInputAlibiMaskIndex])) {
    (void)types.emplace("alibi_mask", input_args[kPagedAttentionMaskInputAlibiMaskIndex]->GetType());
  }
  auto type = CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, op_name);
  // check attn_mask dtype when enabling infer_boost and attn_mask is NOT None
  if (enable_infer_boost && !IsOptionalInputNone(input_args[kPagedAttentionMaskInputAttnMaskIndex])) {
    (void)types.emplace("attn_mask", input_args[kPagedAttentionMaskInputAttnMaskIndex]->GetType());
    CheckAndConvertUtils::CheckTensorTypeValid(
      "attn_mask", input_args[kPagedAttentionMaskInputAttnMaskIndex]->GetType(), {kFloat16}, op_name);
  }

  auto block_tables_type = input_args[kPagedAttentionMaskInputBlockTablesIndex]->GetType();
  CheckAndConvertUtils::CheckTensorTypeValid("block_tables", block_tables_type, {kInt32, kInt64, kUInt64}, op_name);
  auto context_lens_type = input_args[kPagedAttentionMaskInputContextLensIndex]->GetType();
  CheckAndConvertUtils::CheckTensorTypeValid("context_lens", context_lens_type, {kInt32, kInt64, kUInt64}, op_name);

  return type;  // attention_out dtype
}
}  // namespace ops
}  // namespace mindspore

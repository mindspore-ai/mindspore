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

#include "ops/ops_func_impl/paged_attention.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr PagedAttentionFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kPagedAttentionInputsNum, primitive->name());

  auto query_shape_ptr = input_args[kPagedAttentionInputQueryIndex]->GetShape();
  auto shape_element = query_shape_ptr->cast<abstract::ShapePtr>();
  return shape_element;
}

TypePtr PagedAttentionFuncImpl::InferType(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) const {
  const std::set valid_types = {kFloat16};
  auto op_name = primitive->name();
  std::map<std::string, TypePtr> types;

  (void)types.emplace("query", input_args[kPagedAttentionInputQueryIndex]->GetType());
  (void)types.emplace("key_cache", input_args[kPagedAttentionInputKeyCacheIndex]->GetType());
  (void)types.emplace("value_cache", input_args[kPagedAttentionInputValueCacheIndex]->GetType());
  auto type = CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, op_name);

  auto block_tables_type = input_args[kPagedAttentionInputBlockTablesIndex]->GetType();
  CheckAndConvertUtils::CheckTensorTypeValid("block_tables", block_tables_type, {kInt32}, op_name);
  auto context_lens_type = input_args[kPagedAttentionInputContextLensIndex]->GetType();
  CheckAndConvertUtils::CheckTensorTypeValid("context_lens", context_lens_type, {kInt32}, op_name);

  return type;  // attention_out dtype
}
}  // namespace ops
}  // namespace mindspore

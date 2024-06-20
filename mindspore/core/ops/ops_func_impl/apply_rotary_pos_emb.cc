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

#include "ops/ops_func_impl/apply_rotary_pos_emb.h"

#include <set>
#include <string>
#include "abstract/ops/primitive_infer_map.h"
#include "ops/nn_ops.h"
#include "utils/check_convert_utils.h"
#include "ops/primitive_c.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
BaseShapePtr ApplyRotaryPosEmbFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                   const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kApplyRotaryPosEmbInputsNum, op_name);
  auto query_shape_vector = input_args[kApplyRotaryPosEmbQueryIndex]->GetShape()->GetShapeVector();
  auto key_shape_vector = input_args[kApplyRotaryPosEmbKeyIndex]->GetShape()->GetShapeVector();
  auto query_shape = std::make_shared<abstract::Shape>(query_shape_vector);
  auto key_shape = std::make_shared<abstract::Shape>(key_shape_vector);
  return std::make_shared<abstract::TupleShape>(abstract::BaseShapePtrList{query_shape, key_shape});
}

TypePtr ApplyRotaryPosEmbFuncImpl::InferType(const PrimitivePtr &prim,
                                             const std::vector<AbstractBasePtr> &input_args) const {
  auto op_name = prim->name();
  auto query_type = input_args[kApplyRotaryPosEmbQueryIndex]->GetType();
  auto key_type = input_args[kApplyRotaryPosEmbKeyIndex]->GetType();
  auto cos_type = input_args[kApplyRotaryPosEmbCosIndex]->GetType();
  auto sin_type = input_args[kApplyRotaryPosEmbSinIndex]->GetType();

  auto cos_element_type = kNumberTypeBegin;
  if (cos_type->isa<TensorType>()) {
    auto tensor_type = cos_type->cast<TensorTypePtr>();
    auto element = tensor_type->element();
    cos_element_type = element->type_id();
  }
  if (cos_element_type == kNumberTypeFloat16 || cos_element_type == kNumberTypeBFloat16) {
    const std::set qk_valid_types = {kFloat16, kBFloat16};
    std::map<std::string, TypePtr> input_types;
    (void)input_types.emplace("query", query_type);
    (void)input_types.emplace("key", key_type);
    (void)input_types.emplace("cos", cos_type);
    (void)input_types.emplace("sin", sin_type);
    (void)CheckAndConvertUtils::CheckTensorTypeSame(input_types, qk_valid_types, op_name);
  } else if (cos_element_type == kNumberTypeFloat32) {
    const std::set cs_valid_types = {kFloat32};
    const std::set qk_valid_types = {kFloat16, kBFloat16};
    std::map<std::string, TypePtr> cs_types;
    std::map<std::string, TypePtr> qk_types;
    (void)qk_types.emplace("query", query_type);
    (void)qk_types.emplace("key", key_type);
    (void)cs_types.emplace("cos", cos_type);
    (void)cs_types.emplace("sin", sin_type);
    (void)CheckAndConvertUtils::CheckTensorTypeSame(qk_types, qk_valid_types, op_name);
    (void)CheckAndConvertUtils::CheckTensorTypeSame(cs_types, cs_valid_types, op_name);
  } else {
    MS_EXCEPTION(TypeError) << "The primitive[" << op_name
                            << "]'s input arguments[query, key, cos, sin], invalid type list: {"
                            << query_type->ToString() << "," << key_type->ToString() << "," << cos_type->ToString()
                            << "," << sin_type->ToString() << "}";
  }
  auto seq_len_type = input_args[kApplyRotaryPosEmbSeqLenIndex]->GetType();
  const std::set<TypePtr> valid_position_ids_types = {kInt32};
  (void)CheckAndConvertUtils::CheckTypeValid("seq_len", seq_len_type, valid_position_ids_types, op_name);

  TypePtrList output_type_ptr_list(kFApplyRotaryPosEmbOutputsNum);
  output_type_ptr_list[kApplyRotaryPosEmbQueryEmbedIndex] = query_type;
  output_type_ptr_list[kApplyRotaryPosEmbKeyEmbedIndex] = key_type;
  return std::make_shared<Tuple>(output_type_ptr_list);
}
}  // namespace ops
}  // namespace mindspore

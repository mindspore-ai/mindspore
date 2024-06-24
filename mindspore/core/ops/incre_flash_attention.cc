/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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
#include <set>
#include <string>
#include "abstract/ops/primitive_infer_map.h"
#include "ops/nn_ops.h"
#include "utils/check_convert_utils.h"
#include "ops/primitive_c.h"
#include "mindapi/src/helper.h"
#include "ops/incre_flash_attention.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t kInputQueryBSHRank = 3;
constexpr size_t kInputQueryBNSDRank = 4;

ShapeValueDType GetDimension(const std::vector<ShapeValueDType> &dimensions, const std::string &op_name,
                             const std::string &input_name) {
  if (dimensions.empty()) {
    return abstract::Shape::kShapeDimAny;
  }
  ShapeValueDType baseValue = abstract::Shape::kShapeDimAny;
  for (const auto &item : dimensions) {
    if (item == abstract::Shape::kShapeDimAny || item == baseValue) {
      continue;
    }
    if (baseValue == abstract::Shape::kShapeDimAny && item > 0) {
      baseValue = item;
    } else {
      std::ostringstream buffer;
      for (const auto &dim : dimensions) {
        buffer << dim << ", ";
      }
      MS_LOG(EXCEPTION) << "For primitive[" << op_name << "], the " << input_name << " should not be equal -1 or equal "
                        << baseValue << " but got " << buffer.str();
    }
  }
  return baseValue;
}

bool IsOptionalInputNone(const AbstractBasePtr &input) {
  MS_EXCEPTION_IF_NULL(input);
  return input->GetType()->type_id() == kMetaTypeNone;
}

void CheckInputsShape(const AbstractBasePtr &input, const std::vector<ShapeValueDType> &expect_shape,
                      const std::string &op_name, const std::string &input_name, bool optional = false) {
  MS_EXCEPTION_IF_NULL(input);
  auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input->GetShape())[kShape];
  if (optional && input_shape.empty()) {
    return;
  }
  if (input_shape != expect_shape) {
    MS_LOG(EXCEPTION) << op_name << ": The shape of input `" << input_name << "' must be -- " << expect_shape
                      << ", but got shape is " << input_shape;
  }
}

void ParamsValidCheck(const PrimitivePtr &primitive, const std::vector<int64_t> &query_shape,
                      const std::vector<int64_t> &key_shape) {
  auto op_name = primitive->name();
  auto Q_H = query_shape[2];
  auto KV_H = key_shape[2];
  auto N = GetValue<int64_t>(primitive->GetAttr("num_heads"));
  auto KV_N = GetValue<int64_t>(primitive->GetAttr("num_key_value_heads"));
  if (Q_H % N != 0) {
    MS_LOG(EXCEPTION) << op_name << ": 'hidden_size` must be divisible by `head_num`, but got " << Q_H << " and " << N;
  }
  if (KV_N != 0) {
    if (KV_H % KV_N != 0) {
      MS_LOG(EXCEPTION) << op_name << ": 'kv_hidden_size` must be divisible by `num_key_value_heads`, but got " << KV_H
                        << " and " << KV_N;
    }
    if (N % KV_N != 0) {
      MS_LOG(EXCEPTION) << op_name << ": 'num_heads` must be divisible by `num_key_value_heads`, but got " << N
                        << " and " << KV_N;
    }
  } else {
    if (KV_H % N != 0) {
      MS_LOG(EXCEPTION) << op_name << ": 'kv_hidden_size` must be divisible by `head_num`, but got " << KV_H << " and "
                        << N;
    }
  }
}

void CheckShapeSizeRight(const PrimitivePtr &primitive, size_t shape_size) {
  auto op_name = primitive->name();
  if (shape_size != kInputQueryBSHRank && shape_size != kInputQueryBNSDRank) {
    MS_LOG(EXCEPTION) << op_name << ": key or value's shape must be 3 or 4, but got " << shape_size;
  }
}

bool CheckIsFrontend(const std::vector<AbstractBasePtr> &input_args) {
  if (input_args[kIncreFlashAttentionInputKeyIndex]->isa<abstract::AbstractSequence>()) {
    return true;
  }
  return false;
}

std::vector<int64_t> ObtainCorrShape(const std::vector<AbstractBasePtr> &input_args, size_t index) {
  std::vector<int64_t> out_shape;
  if (CheckIsFrontend(input_args)) {
    out_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[index]->GetShape())[kShape];
  } else {
    AbstractBasePtrList elements = input_args;
    out_shape = elements[index]->GetShape()->GetShapeVector();
  }
  return out_shape;
}

std::vector<int64_t> GetIFADynInputShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args,
                                         size_t index) {
  const auto &prim_name = primitive->name();
  std::vector<int64_t> query_shape = ObtainCorrShape(input_args, kIncreFlashAttentionInputQueryIndex);
  auto input_layout = GetValue<std::string>(primitive->GetAttr("input_layout"));

  if (!CheckIsFrontend(input_args)) {
    std::vector<int64_t> shape_vec = ObtainCorrShape(input_args, index);
    if (IsDynamicRank(shape_vec)) {
      if (input_layout == "BSH") {
        return std::vector(kInputQueryBSHRank, abstract::Shape::kShapeDimAny);
      } else {
        return std::vector(kInputQueryBNSDRank, abstract::Shape::kShapeDimAny);
      }
    }
    return shape_vec;
  }

  AbstractBasePtrList kv_elements = input_args[index]->cast<abstract::AbstractSequencePtr>()->elements();

  // if dyn rank
  auto ele_first = kv_elements[0]->cast<abstract::AbstractTensorPtr>();
  std::vector<int64_t> ele_first_sp = CheckAndConvertUtils::ConvertShapePtrToShapeMap(ele_first->GetShape())[kShape];
  if (IsDynamicRank(ele_first_sp)) {
    if (input_layout == "BSH") {
      return std::vector(kInputQueryBSHRank, abstract::Shape::kShapeDimAny);
    } else {
      return std::vector(kInputQueryBNSDRank, abstract::Shape::kShapeDimAny);
    }
  }

  if (kv_elements.size() == 1) {  // [B S H]
    auto element0 = kv_elements[0]->cast<abstract::AbstractTensorPtr>();
    std::vector<int64_t> element0_sp = CheckAndConvertUtils::ConvertShapePtrToShapeMap(element0->GetShape())[kShape];
    CheckShapeSizeRight(primitive, element0_sp.size());
    return element0_sp;
  }
  if (!IsDynamicRank(query_shape) && !IsDynamicShape(query_shape) && (int64_t)kv_elements.size() != query_shape[0]) {
    MS_EXCEPTION(TypeError) << "For '" << prim_name
                            << "', the key or value's list length must be B. But got:" << kv_elements.size();
  }
  std::vector<int64_t> element_first_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(
    kv_elements[0]->cast<abstract::AbstractTensorPtr>()->GetShape())[kShape];
  CheckShapeSizeRight(primitive, element_first_shape.size());

  std::vector<int64_t> element_each_shape;
  for (size_t i = 0; i < kv_elements.size(); ++i) {
    auto element_each = kv_elements[i]->cast<abstract::AbstractTensorPtr>();
    element_each_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(element_each->GetShape())[kShape];
    if (element_each_shape != element_first_shape) {
      MS_LOG(EXCEPTION) << prim_name << ": each element of key or value should be the same shape";
    }
  }
  element_first_shape[0] = (int64_t)kv_elements.size();
  return element_first_shape;
}

void CheckPaddingAttenMaskShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args,
                                int64_t B, int64_t S) {
  auto op_name = primitive->name();
  if (!IsOptionalInputNone(input_args[kIncreFlashAttentionInputPseShiftIndex])) {
    std::vector<int64_t> pse_shift_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(
      input_args[kIncreFlashAttentionInputPseShiftIndex]->GetShape())[kShape];
    size_t len_pa = pse_shift_shape.size();
    if (len_pa > 0 && !pse_shift_shape.empty() && !IsDynamicShape(pse_shift_shape)) {
      if ((pse_shift_shape[0] != B && pse_shift_shape[0] != 1) || pse_shift_shape[len_pa - 1] != S) {
        MS_LOG(EXCEPTION) << op_name << ": The shape of pse_shift must be: "
                          << "(B ... S) or (1 ... S)"
                          << ", but got shape (" << pse_shift_shape[0] << " ... " << pse_shift_shape[len_pa - 1] << ")";
      }
    }
  }

  if (!IsOptionalInputNone(input_args[kIncreFlashAttentionInputAttnMaskIndex]) &&
      IsOptionalInputNone(input_args[kIncreFlashAttentionInputBlockTable])) {
    std::vector<int64_t> atten_mask_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(
      input_args[kIncreFlashAttentionInputAttnMaskIndex]->GetShape())[kShape];
    size_t len_pa = atten_mask_shape.size();
    if (len_pa > 0 && !atten_mask_shape.empty() && !IsDynamicShape(atten_mask_shape)) {
      if (atten_mask_shape[0] != B || atten_mask_shape[len_pa - 1] != S) {
        MS_LOG(EXCEPTION) << op_name << ": The shape of atten_mask must be: "
                          << "(B ... S)"
                          << ", but got shape is (" << atten_mask_shape[0] << " ... " << atten_mask_shape[len_pa - 1]
                          << ")";
      }
    }
  }
}

void CheckActualSeqLengthsShapeValue(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args,
                                     int64_t B, int64_t S) {
  if (IsOptionalInputNone(input_args[kIncreFlashAttentionInputActualSeqLengths])) {
    return;
  }
  auto op_name = primitive->name();
  auto asl_type = input_args[kIncreFlashAttentionInputActualSeqLengths]->GetType();
  MS_EXCEPTION_IF_NULL(asl_type);
  if (!asl_type->isa<TensorType>()) {
    return;
  }
  std::vector<int64_t> asl_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(
    input_args[kIncreFlashAttentionInputActualSeqLengths]->GetShape())[kShape];
  if (IsDynamic(asl_shape)) {
    return;
  }
  if (asl_shape.size() != 1 || (asl_shape[0] != 1 && asl_shape[0] != B)) {
    MS_LOG(EXCEPTION) << op_name << ": The size of actual_seq_lengths's shape must be: 1 or " << B << ", but got "
                      << asl_shape[0];
  }
}

abstract::ShapePtr IncreFlashAttentionInferShapeBSH(const PrimitivePtr &primitive,
                                                    const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  CheckAndConvertUtils::CheckInputArgs(input_args, kLessEqual, kIncreFlashAttentionInputsNum, op_name);
  std::vector<int64_t> query_shape = ObtainCorrShape(input_args, kIncreFlashAttentionInputQueryIndex);
  std::vector<int64_t> key_shape = GetIFADynInputShape(primitive, input_args, kIncreFlashAttentionInputKeyIndex);
  std::vector<int64_t> value_shape = GetIFADynInputShape(primitive, input_args, kIncreFlashAttentionInputValueIndex);

  if (IsDynamicRank(query_shape)) {
    query_shape = std::vector(kInputQueryBSHRank, abstract::Shape::kShapeDimAny);
  }

  if (CheckIsFrontend(input_args) && IsOptionalInputNone(input_args[kIncreFlashAttentionInputBlockTable])) {
    if (!IsDynamicShape(query_shape) && !IsDynamicShape(key_shape) && !IsDynamicShape(value_shape)) {
      int64_t B = query_shape[0];
      int64_t Q_H = query_shape[2];
      ParamsValidCheck(primitive, query_shape, key_shape);
      CheckInputsShape(input_args[kIncreFlashAttentionInputQueryIndex], {B, 1, Q_H}, op_name, "query");
      int64_t S = key_shape[1];
      CheckPaddingAttenMaskShape(primitive, input_args, B, S);
      CheckActualSeqLengthsShapeValue(primitive, input_args, B, S);
      if (key_shape != value_shape) {
        MS_LOG(EXCEPTION) << op_name << ": The shape of key and value must be same, but got: " << key_shape << " and "
                          << value_shape;
      }
    }
  }

  ShapeVector attention_out_shape(kInputQueryBSHRank, abstract::Shape::kShapeDimAny);
  if (!IsOptionalInputNone(input_args[kIncreFlashAttentionInputBlockTable])) {
    // kv: [num_blocks,block_size,hidden_size], q: [batch,seq_length,hidden_size]
    attention_out_shape[0] = query_shape[0];
  } else {
    attention_out_shape[0] = GetDimension({query_shape[0]}, op_name, "B");
  }
  attention_out_shape[1] = 1;
  attention_out_shape[2] = GetDimension({query_shape[2]}, op_name, "H");  // 2: h_index
  return std::make_shared<abstract::Shape>(attention_out_shape);
}

abstract::ShapePtr IncreFlashAttentionInferShapeBNSD(const PrimitivePtr &primitive,
                                                     const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  CheckAndConvertUtils::CheckInputArgs(input_args, kLessEqual, kIncreFlashAttentionInputsNum, op_name);
  std::vector<int64_t> query_shape = ObtainCorrShape(input_args, kIncreFlashAttentionInputQueryIndex);
  std::vector<int64_t> key_shape = GetIFADynInputShape(primitive, input_args, kIncreFlashAttentionInputKeyIndex);
  std::vector<int64_t> value_shape = GetIFADynInputShape(primitive, input_args, kIncreFlashAttentionInputValueIndex);
  if (IsDynamicRank(query_shape)) {
    query_shape = std::vector(kInputQueryBNSDRank, abstract::Shape::kShapeDimAny);
  }
  if (CheckIsFrontend(input_args) && IsOptionalInputNone(input_args[kIncreFlashAttentionInputBlockTable])) {
    if (!IsDynamicShape(query_shape) && !IsDynamicShape(key_shape) && !IsDynamicShape(value_shape)) {
      int64_t B = query_shape[0];
      int64_t N_Q = query_shape[1];
      int64_t D = query_shape[3];
      int64_t KV_N = key_shape[1];
      int64_t N = GetValue<int64_t>(primitive->GetAttr("num_heads"));
      int64_t KV_N_ATTR = GetValue<int64_t>(primitive->GetAttr("num_key_value_heads"));
      if (N_Q != N) {
        MS_LOG(EXCEPTION) << op_name << ": query 's shape[1] should be num_heads, but got: " << N_Q << " and " << N;
      }
      if (KV_N_ATTR != 0 && KV_N != KV_N_ATTR) {
        MS_LOG(EXCEPTION) << op_name << ": key and value 's shape[1] should be num_key_value_heads, but got: " << KV_N
                          << " and " << KV_N_ATTR;
      }
      if (N % KV_N != 0) {
        MS_LOG(EXCEPTION) << op_name << ": 'num_heads` must be divisible by `num_key_value_heads`, but got " << N
                          << " and " << KV_N;
      }
      CheckInputsShape(input_args[kIncreFlashAttentionInputQueryIndex], {B, N, 1, D}, op_name, "query");

      int64_t S = key_shape[2];
      CheckPaddingAttenMaskShape(primitive, input_args, B, S);
      CheckActualSeqLengthsShapeValue(primitive, input_args, B, S);
      if (key_shape != value_shape) {
        MS_LOG(EXCEPTION) << op_name << ": The shape of key and value must be same, but got: " << key_shape << " and "
                          << value_shape;
      }
    }
  }

  ShapeVector attention_out_shape(kInputQueryBNSDRank, abstract::Shape::kShapeDimAny);
  attention_out_shape[0] = GetDimension({query_shape[0]}, op_name, "B");
  attention_out_shape[1] = GetDimension({query_shape[1]}, op_name, "N");
  attention_out_shape[2] = 1;                                             // 2: s_index
  attention_out_shape[3] = GetDimension({query_shape[3]}, op_name, "D");  // 3: d_index
  return std::make_shared<abstract::Shape>(attention_out_shape);
}

abstract::ShapePtr IncreFlashAttentionInferShape(const PrimitivePtr &primitive,
                                                 const std::vector<AbstractBasePtr> &input_args) {
  auto input_layout = GetValue<std::string>(primitive->GetAttr("input_layout"));
  if (input_layout == "BSH") {
    return IncreFlashAttentionInferShapeBSH(primitive, input_args);
  } else {
    return IncreFlashAttentionInferShapeBNSD(primitive, input_args);
  }
}

void CheckQuantParamType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = prim->name();
  std::map<std::string, TypePtr> dequant_types;
  const std::set<TypePtr> dequant_valid_types = {kUInt64};
  if (!IsOptionalInputNone(input_args[kIncreFlashAttentionInputDequantScale1])) {
    (void)dequant_types.emplace("dequant_scale1", input_args[kIncreFlashAttentionInputDequantScale1]->GetType());
    (void)CheckAndConvertUtils::CheckTensorTypeSame(dequant_types, dequant_valid_types, op_name);
  }
  if (!IsOptionalInputNone(input_args[kIncreFlashAttentionInputDequantScale2])) {
    (void)dequant_types.emplace("dequant_scale2", input_args[kIncreFlashAttentionInputDequantScale2]->GetType());
    (void)CheckAndConvertUtils::CheckTensorTypeSame(dequant_types, dequant_valid_types, op_name);
  }
  std::map<std::string, TypePtr> quant_types;
  const std::set<TypePtr> quant_valid_types = {kFloat};
  if (!IsOptionalInputNone(input_args[kIncreFlashAttentionInputQuantScale1])) {
    (void)quant_types.emplace("quant_scale1", input_args[kIncreFlashAttentionInputQuantScale1]->GetType());
    (void)CheckAndConvertUtils::CheckTensorTypeSame(quant_types, quant_valid_types, op_name);
  }
  if (!IsOptionalInputNone(input_args[kIncreFlashAttentionInputQuantScale2])) {
    (void)quant_types.emplace("quant_scale2", input_args[kIncreFlashAttentionInputQuantScale2]->GetType());
    (void)CheckAndConvertUtils::CheckTensorTypeSame(quant_types, quant_valid_types, op_name);
  }
  if (!IsOptionalInputNone(input_args[kIncreFlashAttentionInputQuantOffset2])) {
    (void)quant_types.emplace("quant_offset2", input_args[kIncreFlashAttentionInputQuantOffset2]->GetType());
    (void)CheckAndConvertUtils::CheckTensorTypeSame(quant_types, quant_valid_types, op_name);
  }
  if (!IsOptionalInputNone(input_args[kIncreFlashAttentionInputAntiquantScale])) {
    (void)quant_types.emplace("antiquant_scale", input_args[kIncreFlashAttentionInputAntiquantScale]->GetType());
    (void)CheckAndConvertUtils::CheckTensorTypeSame(quant_types, quant_valid_types, op_name);
  }
  if (!IsOptionalInputNone(input_args[kIncreFlashAttentionInputAntiquantOffset])) {
    (void)quant_types.emplace("antiquant_offset", input_args[kIncreFlashAttentionInputAntiquantOffset]->GetType());
    (void)CheckAndConvertUtils::CheckTensorTypeSame(quant_types, quant_valid_types, op_name);
  }
  std::map<std::string, TypePtr> block_table_types;
  const std::set<TypePtr> block_valid_types = {kInt32};
  if (!IsOptionalInputNone(input_args[kIncreFlashAttentionInputBlockTable])) {
    (void)block_table_types.emplace("block_table", input_args[kIncreFlashAttentionInputBlockTable]->GetType());
    (void)CheckAndConvertUtils::CheckTensorTypeSame(block_table_types, block_valid_types, op_name);
  }
}

TypePtr IncreFlashAttentionInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = prim->name();
  if (CheckIsFrontend(input_args)) {
    CheckQuantParamType(prim, input_args);
    std::map<std::string, TypePtr> pse_shift_types;
    const std::set<TypePtr> pse_shift_valid_types = {kFloat16, kBFloat16};
    if (!IsOptionalInputNone(input_args[kIncreFlashAttentionInputPseShiftIndex])) {
      (void)pse_shift_types.emplace("pse_shift", input_args[kIncreFlashAttentionInputPseShiftIndex]->GetType());
      (void)CheckAndConvertUtils::CheckTensorTypeSame(pse_shift_types, pse_shift_valid_types, op_name);
    }
    std::map<std::string, TypePtr> atten_mask_types;
    const std::set<TypePtr> atten_mask_valid_types = {kBool, kInt8, kUInt8};
    if (!IsOptionalInputNone(input_args[kIncreFlashAttentionInputAttnMaskIndex])) {
      (void)atten_mask_types.emplace("attn_mask", input_args[kIncreFlashAttentionInputAttnMaskIndex]->GetType());
      (void)CheckAndConvertUtils::CheckTensorTypeSame(atten_mask_types, atten_mask_valid_types, op_name);
    }
    auto asl_type = input_args[kIncreFlashAttentionInputActualSeqLengths]->GetType();
    MS_EXCEPTION_IF_NULL(asl_type);
    if (asl_type->isa<TensorType>()) {
      std::map<std::string, TypePtr> asl_types;
      const std::set<TypePtr> acl_valid_types = {kInt64};
      if (!IsOptionalInputNone(input_args[kIncreFlashAttentionInputActualSeqLengths])) {
        (void)asl_types.emplace("actual_seq_lengths", input_args[kIncreFlashAttentionInputActualSeqLengths]->GetType());
        (void)CheckAndConvertUtils::CheckTensorTypeSame(asl_types, acl_valid_types, op_name);
      }
    }
  }

  std::map<std::string, TypePtr> q_types;
  std::map<std::string, TypePtr> kv_types;
  const std::set<TypePtr> q_valid_types = {kFloat16, kBFloat16};
  const std::set<TypePtr> kv_valid_types = {kFloat16, kBFloat16, kInt8};
  TypePtr type;
  (void)q_types.emplace("query", input_args[kIncreFlashAttentionInputQueryIndex]->GetType());
  if (CheckIsFrontend(input_args)) {
    AbstractBasePtrList elements =
      input_args[kIncreFlashAttentionInputKeyIndex]->cast<abstract::AbstractSequencePtr>()->elements();
    (void)kv_types.emplace("key", elements[0]->GetType());
    elements = input_args[kIncreFlashAttentionInputValueIndex]->cast<abstract::AbstractSequencePtr>()->elements();
    (void)kv_types.emplace("value", elements[0]->GetType());
    type = CheckAndConvertUtils::CheckTensorTypeSame(kv_types, kv_valid_types, op_name);
    MS_EXCEPTION_IF_NULL(type);
  }
  type = CheckAndConvertUtils::CheckTensorTypeSame(q_types, q_valid_types, op_name);
  return type;
}
}  // namespace

AbstractBasePtr IncreFlashAttentionInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kIncreFlashAttentionInputsNum, primitive->name());
  auto infer_shape = IncreFlashAttentionInferShape(primitive, input_args);
  auto infer_type = IncreFlashAttentionInferType(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(IncreFlashAttention, BaseOperator);
// AG means auto generated
class MIND_API AGIncreFlashAttentionInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return IncreFlashAttentionInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return IncreFlashAttentionInferType(primitive, input_args);
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    auto op_name = primitive->name();
    AbstractBasePtr key_arg = input_args[kIncreFlashAttentionInputKeyIndex];
    AbstractBasePtr value_arg = input_args[kIncreFlashAttentionInputValueIndex];
    size_t valid_seq_length = 1;
    if (!(key_arg->isa<abstract::AbstractSequence>() &&
          key_arg->cast<abstract::AbstractSequencePtr>()->elements().size() == valid_seq_length &&
          key_arg->cast<abstract::AbstractSequencePtr>()->elements()[0]->isa<abstract::AbstractTensor>())) {
      MS_LOG(EXCEPTION) << op_name << ": parameter key should be a sequence containing exactly 1 tensor. ";
    }

    if (!(value_arg->isa<abstract::AbstractSequence>() &&
          value_arg->cast<abstract::AbstractSequencePtr>()->elements().size() == valid_seq_length &&
          value_arg->cast<abstract::AbstractSequencePtr>()->elements()[0]->isa<abstract::AbstractTensor>())) {
      MS_LOG(EXCEPTION) << op_name << ": parameter value should be a sequence containing exactly 1 tensor. ";
    }

    return IncreFlashAttentionInfer(engine, primitive, input_args);
  }

  std::set<int64_t> GetValueDependArgIndices() const { return {4}; }  // 4: pos of valuedepend
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(IncreFlashAttention, prim::kPrimIncreFlashAttention, AGIncreFlashAttentionInfer,
                                 false);
}  // namespace ops
}  // namespace mindspore

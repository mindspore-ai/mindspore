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

#include "infer/ops_func_impl/prompt_flash_attention.h"
#include <set>
#include <memory>
#include <map>
#include <string>
#include <sstream>
#include "abstract/ops/primitive_infer_map.h"
#include "primitive/nn_ops.h"
#include "utils/check_convert_utils.h"
#include "utils/shape_utils.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "mindapi/helper.h"
#include "ops_utils/op_constants.h"
#include "include/op_enum.h"

namespace mindspore {
namespace ops {
constexpr size_t kInputQueryBSHRankLayout = 3;
constexpr size_t kInputQueryBNSDRankLayout = 4;
constexpr int64_t SPARSE_LEFTUP_ATTENTION_SIZE = 2048;
enum class SparseMode : int64_t { SPARSE_MODE_0, SPARSE_MODE_1, SPARSE_MODE_2, SPARSE_MODE_3, SPARSE_MODE_4 };
constexpr int64_t ALIGN_BFLOAT_16 = 16;

ShapeValueDType GetDimensionPFA(const std::vector<ShapeValueDType> &dimensions, const std::string &op_name,
                                const std::string &input_name) {
  if (dimensions.empty()) {
    MS_LOG(EXCEPTION) << "For primitive[" << op_name << "], the " << input_name << " should not be empty";
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
      MS_LOG(EXCEPTION) << "For primitive[" << op_name << "], the " << input_name << " should be equal " << baseValue
                        << ", but got " << buffer.str();
    }
  }
  return baseValue;
}

bool CheckTenSorShape(const ShapeVector &tensor_shape, const std::vector<ShapeVector> &expect_shapes) {
  for (size_t i = 0; i < expect_shapes.size(); i++) {
    const auto &expect_shape = expect_shapes[i];
    if (tensor_shape.size() != expect_shape.size()) {
      continue;
    }

    bool is_match = true;
    for (size_t j = 0; j < expect_shape.size(); j++) {
      if (expect_shape[j] == abstract::Shape::kShapeDimAny || tensor_shape[j] == abstract::Shape::kShapeDimAny) {
        continue;
      }
      if (expect_shape[j] != tensor_shape[j]) {
        is_match = false;
        break;
      }
    }
    if (is_match) {
      return true;
    }
  }
  return false;
}

void CheckActuaSeqLength(AbstractBasePtr input_arg, int64_t input_s, int64_t dim_b, const std::string &op_name,
                         const std::string &input_name) {
  if (input_arg->GetType()->type_id() != kMetaTypeNone) {
    auto val_type = input_arg->GetType();
    if (val_type->isa<Tuple>()) {
      auto val_opt = GetArrayValue<int64_t>(input_arg);
      if (!val_opt.has_value()) {
        return;
      }
      auto seq_length_vec = val_opt.value();
      if (seq_length_vec.HasUnknownValue()) {
        return;
      }
      if (dim_b != abstract::Shape::kShapeDimAny) {
        CheckAndConvertUtils::CheckInteger("size of " + input_name, seq_length_vec.size(), kEqual, dim_b, op_name);
      }
      if (input_s < 0) {
        return;
      }
      for (size_t i = 0; i < seq_length_vec.size(); ++i) {
        CheckAndConvertUtils::CheckInteger(input_name, seq_length_vec[i], kLessEqual, input_s, op_name);
      }
    } else {
      auto actual_seq_length_shape = input_arg->GetShape()->GetShapeVector();
      CheckAndConvertUtils::CheckInteger("dim of " + input_name, actual_seq_length_shape.size(), kEqual, 1, op_name);
      if (!IsDynamic(actual_seq_length_shape) && dim_b != abstract::Shape::kShapeDimAny) {
        CheckAndConvertUtils::CheckInteger("size of " + input_name, actual_seq_length_shape[0], kEqual, dim_b, op_name);
      }
    }
  }
}

void CheckOptinalInputShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args,
                            ShapeValueDType b, ShapeValueDType q_s, ShapeValueDType kv_s) {
  auto op_name = primitive->name();
  auto sparse_mode = GetScalarValue<int64_t>(input_args[kPromptFlashAttentionInputSparseModeIndex]->GetValue()).value();
  if (sparse_mode != static_cast<int64_t>(SparseMode::SPARSE_MODE_0) &&
      sparse_mode != static_cast<int64_t>(SparseMode::SPARSE_MODE_1) &&
      sparse_mode != static_cast<int64_t>(SparseMode::SPARSE_MODE_2) &&
      sparse_mode != static_cast<int64_t>(SparseMode::SPARSE_MODE_3) &&
      sparse_mode != static_cast<int64_t>(SparseMode::SPARSE_MODE_4)) {
    MS_LOG(EXCEPTION) << "For primitive[" << op_name << "], sparse_mode must be 0, 1, 2, 3 or 4 but got "
                      << sparse_mode;
  }
  std::vector<ShapeVector> expect_mask_shapes = {
    {q_s, kv_s}, {1, q_s, kv_s}, {b, q_s, kv_s}, {b, 1, q_s, kv_s}, {1, 1, q_s, kv_s}};
  if (sparse_mode == static_cast<int64_t>(SparseMode::SPARSE_MODE_2) ||
      sparse_mode == static_cast<int64_t>(SparseMode::SPARSE_MODE_3) ||
      sparse_mode == static_cast<int64_t>(SparseMode::SPARSE_MODE_4)) {
    expect_mask_shapes = {{SPARSE_LEFTUP_ATTENTION_SIZE, SPARSE_LEFTUP_ATTENTION_SIZE},
                          {1, SPARSE_LEFTUP_ATTENTION_SIZE, SPARSE_LEFTUP_ATTENTION_SIZE},
                          {1, 1, SPARSE_LEFTUP_ATTENTION_SIZE, SPARSE_LEFTUP_ATTENTION_SIZE}};
  }
  auto atten_mask_ptr = input_args[kPromptFlashAttentionInputAttenMaskIndex];
  if (atten_mask_ptr->GetType()->type_id() != kMetaTypeNone) {
    auto atten_mask_shape = atten_mask_ptr->GetShape()->GetShapeVector();
    if (!atten_mask_shape.empty() && !IsDynamicRank(atten_mask_shape)) {
      if (!CheckTenSorShape(atten_mask_shape, expect_mask_shapes)) {
        MS_LOG(EXCEPTION) << "For primitive[" << op_name << "], atten_mask shape:  " << atten_mask_shape
                          << " dont match any of expect shape: " << expect_mask_shapes;
      }
    }
  }

  CheckActuaSeqLength(input_args[kPromptFlashAttentionInputActualSeqLengthsIndex], q_s, b, op_name,
                      "actual_seq_lengths");
  CheckActuaSeqLength(input_args[kPromptFlashAttentionInputActualSeqLengthsKvIndex], kv_s, b, op_name,
                      "actual_seq_lengths_kv");
}

abstract::TupleShapePtr GetInputsShape(const std::vector<AbstractBasePtr> &input_args,
                                       const FASInputLayoutMode &input_layout, const std::string &op_name) {
  auto query_shape = input_args[kPromptFlashAttentionInputQueryIndex]->GetShape()->GetShapeVector();
  auto key_shape = input_args[kPromptFlashAttentionInputKeyIndex]->GetShape()->GetShapeVector();
  auto value_shape = input_args[kPromptFlashAttentionInputValueIndex]->GetShape()->GetShapeVector();

  bool qeury_rank_is_dyn = IsDynamicRank(query_shape);
  bool key_rank_is_dyn = IsDynamicRank(key_shape);
  bool value_rank_is_dyn = IsDynamicRank(value_shape);
  size_t temp_rank = input_layout == FASInputLayoutMode::BSH ? kInputQueryBSHRankLayout : kInputQueryBNSDRankLayout;
  if (qeury_rank_is_dyn) {
    query_shape = std::vector(temp_rank, abstract::Shape::kShapeDimAny);
  }
  if (key_rank_is_dyn) {
    key_shape = std::vector(temp_rank, abstract::Shape::kShapeDimAny);
  }
  if (value_rank_is_dyn) {
    value_shape = std::vector(temp_rank, abstract::Shape::kShapeDimAny);
  }
  abstract::BaseShapePtrList input_shape_ptr_list(3);
  input_shape_ptr_list[0] = std::make_shared<abstract::Shape>(query_shape);
  input_shape_ptr_list[1] = std::make_shared<abstract::Shape>(key_shape);
  input_shape_ptr_list[2] = std::make_shared<abstract::Shape>(value_shape);

  CheckAndConvertUtils::CheckInteger("rank of query", query_shape.size(), kEqual, temp_rank, op_name);
  CheckAndConvertUtils::CheckInteger("rank of key", key_shape.size(), kEqual, temp_rank, op_name);
  CheckAndConvertUtils::CheckInteger("rank of value", value_shape.size(), kEqual, temp_rank, op_name);
  return std::make_shared<abstract::TupleShape>(input_shape_ptr_list);
}

void CheckShapeAlign(TypePtr query_dtype, int64_t dim_d, const std::string &op_name) {
  bool is_query_bf16 = IsIdentidityOrSubclass(query_dtype, kTensorTypeBF16);
  if (is_query_bf16 && dim_d != abstract::Shape::kShapeDimAny && dim_d % ALIGN_BFLOAT_16 != 0) {
    MS_LOG(EXCEPTION) << "For primitive[" << op_name
                      << "], dtype of query is bfloat16, dimension D must align with 16! but got " << dim_d;
  }
}

ShapeVector InferShapeBSH(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args,
                          int64_t num_heads, int64_t num_heads_kv) {
  auto op_name = primitive->name();
  auto input_shapes = GetInputsShape(input_args, FASInputLayoutMode::BSH, op_name);
  auto input_shape_ptrs = input_shapes->cast_ptr<abstract::TupleShape>();
  ShapeVector query_shape = ((*input_shape_ptrs)[0]->cast<abstract::ShapePtr>())->shape();
  ShapeVector key_shape = ((*input_shape_ptrs)[1]->cast<abstract::ShapePtr>())->shape();
  ShapeVector value_shape = ((*input_shape_ptrs)[2]->cast<abstract::ShapePtr>())->shape();
  if (key_shape.size() == 1 && value_shape.size() == 1 && key_shape[0] == 0 && value_shape[0] == 0) {
    return query_shape;
  }
  auto b_index = 0;
  auto s_index = 1;
  auto h_index = 2;
  ShapeVector attention_out_shape(3);
  auto dim_b = GetDimensionPFA({query_shape[b_index], key_shape[b_index], value_shape[b_index]}, op_name, "B");
  auto dim_q_s = query_shape[s_index];
  auto dim_kv_s = GetDimensionPFA({key_shape[s_index], value_shape[s_index]}, op_name, "KV_S");
  int64_t q_h = abstract::Shape::kShapeDimAny;
  int64_t kv_h = abstract::Shape::kShapeDimAny;
  if (num_heads_kv == 0) {
    q_h = GetDimensionPFA({query_shape[h_index], key_shape[h_index], value_shape[h_index]}, op_name, "H");
    kv_h = q_h;
  } else {
    q_h = query_shape[h_index];
    kv_h = GetDimensionPFA({key_shape[h_index], value_shape[h_index]}, op_name, "KV_H");
  }
  int64_t q_d = abstract::Shape::kShapeDimAny;
  int64_t kv_d = abstract::Shape::kShapeDimAny;
  if (q_h != abstract::Shape::kShapeDimAny) {
    if (q_h % num_heads != 0) {
      MS_LOG(EXCEPTION) << "For primitive[" << op_name << "], H must be divisible by `num_heads`, but got " << q_h
                        << " and " << num_heads;
    }
    q_d = q_h / num_heads;
  }
  if (num_heads_kv != 0 && kv_h != abstract::Shape::kShapeDimAny) {
    if (kv_h % num_heads_kv != 0) {
      MS_LOG(EXCEPTION) << "For primitive[" << op_name << "], KV_H must be divisible by `num_key_value_heads`, but got "
                        << kv_h << " and " << num_heads_kv;
    }
    kv_d = kv_h / num_heads_kv;
  }
  if (q_d != abstract::Shape::kShapeDimAny && kv_d != abstract::Shape::kShapeDimAny && q_d != kv_d) {
    MS_LOG(EXCEPTION) << "For primitive[" << op_name << "], Q_D must be equal KV_D, but got " << q_d << " and " << kv_d;
  }
  auto dim_d = q_d > kv_d ? q_d : kv_d;
  auto query_dtype = input_args[kPromptFlashAttentionInputQueryIndex]->GetType();
  CheckShapeAlign(query_dtype, dim_d, op_name);
  auto sparse_mode = GetScalarValue<int64_t>(input_args[kPromptFlashAttentionInputSparseModeIndex]->GetValue()).value();
  if (sparse_mode == static_cast<int64_t>(SparseMode::SPARSE_MODE_0) ||
      sparse_mode == static_cast<int64_t>(SparseMode::SPARSE_MODE_1)) {
    CheckOptinalInputShape(primitive, input_args, dim_b, dim_q_s, dim_kv_s);
  }
  return ShapeVector{dim_b, dim_q_s, q_h};
}

ShapeVector InferShapeBNSD(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args,
                           int64_t num_heads, int64_t num_heads_kv) {
  auto op_name = primitive->name();
  auto input_shapes = GetInputsShape(input_args, FASInputLayoutMode::BNSD, op_name);
  auto input_shape_ptrs = input_shapes->cast_ptr<abstract::TupleShape>();
  ShapeVector query_shape = ((*input_shape_ptrs)[0]->cast<abstract::ShapePtr>())->shape();
  ShapeVector key_shape = ((*input_shape_ptrs)[1]->cast<abstract::ShapePtr>())->shape();
  ShapeVector value_shape = ((*input_shape_ptrs)[2]->cast<abstract::ShapePtr>())->shape();
  if (key_shape.size() == 1 && value_shape.size() == 1 && key_shape[0] == 0 && value_shape[0] == 0) {
    return query_shape;
  }
  auto b_index = 0;
  auto n_index = 1;
  auto s_index = 2;
  auto d_index = 3;

  auto dim_b = GetDimensionPFA({query_shape[b_index], key_shape[b_index], value_shape[b_index]}, op_name, "B");
  int64_t kv_n = abstract::Shape::kShapeDimAny;
  int64_t q_n = abstract::Shape::kShapeDimAny;
  if (num_heads_kv == 0) {
    q_n = GetDimensionPFA({query_shape[n_index], key_shape[n_index], value_shape[n_index]}, op_name, "N");
    kv_n = q_n;
  } else {
    q_n = query_shape[n_index];
    kv_n = GetDimensionPFA({key_shape[n_index], value_shape[n_index]}, op_name, "KV_N");
  }
  auto dim_q_s = query_shape[s_index];
  auto dim_kv_s = GetDimensionPFA({key_shape[s_index], value_shape[s_index]}, op_name, "KV_S");
  auto dim_d = GetDimensionPFA({query_shape[d_index], key_shape[d_index], value_shape[d_index]}, op_name, "D");
  if (q_n != abstract::Shape::kShapeDimAny && q_n != num_heads) {
    MS_LOG(EXCEPTION) << "For primitive[" << op_name << "], N must be equal num_heads, but got " << q_n << " and "
                      << num_heads;
  }
  if (num_heads_kv != 0 && kv_n != abstract::Shape::kShapeDimAny && num_heads_kv != kv_n) {
    MS_LOG(EXCEPTION) << "For primitive[" << op_name << "], KV_N must equal num_key_value_heads, but got " << kv_n
                      << " and " << num_heads_kv;
  }
  auto query_dtype = input_args[kPromptFlashAttentionInputQueryIndex]->GetType();
  CheckShapeAlign(query_dtype, dim_d, op_name);
  auto sparse_mode_opt = GetScalarValue<int64_t>(input_args[kPromptFlashAttentionInputSparseModeIndex]->GetValue());
  if (sparse_mode_opt.has_value()) {
    auto sparse_mode = sparse_mode_opt.value();
    if (sparse_mode == static_cast<int64_t>(SparseMode::SPARSE_MODE_0) ||
        sparse_mode == static_cast<int64_t>(SparseMode::SPARSE_MODE_1)) {
      CheckOptinalInputShape(primitive, input_args, dim_b, dim_q_s, dim_kv_s);
    }
  } else {
    MS_LOG(ERROR) << "sparse_mode has no value.";
  }

  return ShapeVector{dim_b, q_n, dim_q_s, dim_d};
}

BaseShapePtr PromptFlashAttentionFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                      const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kPromptFlashAttentionInputsNum, op_name);
  auto input_layout_value = input_args[kPromptFlashAttentionInputLayoutIndex]->GetValue();
  MS_EXCEPTION_IF_NULL(input_layout_value);
  auto input_layout_opt = GetScalarValue<int64_t>(input_layout_value);
  auto num_heads_value = input_args[kPromptFlashAttentionInputNumHeadsIndex]->GetValue();
  MS_EXCEPTION_IF_NULL(num_heads_value);
  auto num_heads_opt = GetScalarValue<int64_t>(num_heads_value);
  auto num_heads_kv_value = input_args[kPromptFlashAttentionInputNumKeyValueHeadsIndex]->GetValue();
  MS_EXCEPTION_IF_NULL(num_heads_kv_value);
  auto num_heads_kv_opt = GetScalarValue<int64_t>(num_heads_kv_value);

  if (!input_layout_opt.has_value() || !num_heads_opt.has_value() || !num_heads_kv_opt.has_value()) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{-2});
  }
  auto input_layout = input_layout_opt.value();
  auto num_heads = num_heads_opt.value();
  auto num_heads_kv = num_heads_kv_opt.value();

  if (num_heads <= 0) {
    MS_LOG(EXCEPTION) << "For primitive[" << op_name << "], the num_heads should greater than zero.";
  }
  if (num_heads_kv < 0) {
    MS_LOG(EXCEPTION) << "For primitive[" << op_name << "], the num_key_value_heads should not less than zero.";
  }
  if (num_heads_kv > 0 && num_heads % num_heads_kv != 0) {
    MS_LOG(EXCEPTION) << "For primitive[" << op_name
                      << "], the num_heads should be divide by num_key_value_heads, but got num_heads: " << num_heads
                      << ", and num_key_value_heads: " << num_heads_kv;
  }

  ShapeVector attention_out_shape;
  if (input_layout == FASInputLayoutMode::BSH) {
    attention_out_shape = InferShapeBSH(primitive, input_args, num_heads, num_heads_kv);
  } else if (input_layout == FASInputLayoutMode::BNSD) {
    attention_out_shape = InferShapeBNSD(primitive, input_args, num_heads, num_heads_kv);
  } else {
    attention_out_shape = {abstract::Shape::kShapeRankAny};
  }
  return std::make_shared<abstract::Shape>(attention_out_shape);
}

bool CheckOptinalNone(const AbstractBasePtr &input) {
  MS_EXCEPTION_IF_NULL(input);
  return input->GetType()->type_id() == kMetaTypeNone;
}

void CheckQuantParams(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = prim->name();
  std::map<std::string, TypePtr> quant_scale1_type;
  const std::set<TypePtr> quant_scale1_type_valid_type = {kFloat32};
  if (!CheckOptinalNone(input_args[kPromptFlashAttentionInputQuantScale1Index])) {
    (void)quant_scale1_type.emplace("quant_scale1", input_args[kPromptFlashAttentionInputQuantScale1Index]->GetType());
    (void)CheckAndConvertUtils::CheckTensorTypeSame(quant_scale1_type, quant_scale1_type_valid_type, op_name);
  }
  std::map<std::string, TypePtr> quant_scale2_offset2_types;
  const std::set<TypePtr> quant_scale2_offset2_valid_type = {kFloat32, kBFloat16};
  if (!CheckOptinalNone(input_args[kPromptFlashAttentionInputQuantScale2Index])) {
    (void)quant_scale2_offset2_types.emplace("quant_scale2",
                                             input_args[kPromptFlashAttentionInputQuantScale2Index]->GetType());
    (void)CheckAndConvertUtils::CheckTensorTypeSame(quant_scale2_offset2_types, quant_scale2_offset2_valid_type,
                                                    op_name);
  }
  if (!CheckOptinalNone(input_args[kPromptFlashAttentionInputQuantOffset2Index])) {
    (void)quant_scale2_offset2_types.emplace("quant_offset2",
                                             input_args[kPromptFlashAttentionInputQuantOffset2Index]->GetType());
    (void)CheckAndConvertUtils::CheckTensorTypeSame(quant_scale2_offset2_types, quant_scale2_offset2_valid_type,
                                                    op_name);
  }

  std::map<std::string, TypePtr> dequant_types;
  const std::set<TypePtr> dequant_valid_types = {kFloat32, kUInt64};
  if (!CheckOptinalNone(input_args[kPromptFlashAttentionInputDeqScale1Index])) {
    (void)dequant_types.emplace("deq_scale1", input_args[kPromptFlashAttentionInputDeqScale1Index]->GetType());
    (void)CheckAndConvertUtils::CheckTensorTypeSame(dequant_types, dequant_valid_types, op_name);
  }
  if (!CheckOptinalNone(input_args[kPromptFlashAttentionInputDeqScale2Index])) {
    (void)dequant_types.emplace("deq_scale2", input_args[kPromptFlashAttentionInputDeqScale2Index]->GetType());
    (void)CheckAndConvertUtils::CheckTensorTypeSame(dequant_types, dequant_valid_types, op_name);
  }
}

TypePtr PromptFlashAttentionInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  CheckQuantParams(primitive, input_args);
  std::map<std::string, TypePtr> pse_shift_types;
  const std::set<TypePtr> pse_shift_valid_types = {kFloat16, kBFloat16};
  if (!CheckOptinalNone(input_args[kPromptFlashAttentionInputPseShiftIndex])) {
    (void)pse_shift_types.emplace("pse_shift", input_args[kPromptFlashAttentionInputPseShiftIndex]->GetType());
    (void)CheckAndConvertUtils::CheckTensorTypeSame(pse_shift_types, pse_shift_valid_types, op_name);
  }
  std::map<std::string, TypePtr> atten_mask_types;
  const std::set<TypePtr> atten_mask_valid_types = {kBool, kInt8, kUInt8, kFloat16};
  if (!CheckOptinalNone(input_args[kPromptFlashAttentionInputAttenMaskIndex])) {
    (void)atten_mask_types.emplace("atten_mask", input_args[kPromptFlashAttentionInputAttenMaskIndex]->GetType());
    (void)CheckAndConvertUtils::CheckTensorTypeSame(atten_mask_types, atten_mask_valid_types, op_name);
  }

  if (!CheckOptinalNone(input_args[kPromptFlashAttentionInputQuantScale2Index])) {
    return kInt8;
  }
  std::map<std::string, TypePtr> types;
  (void)types.emplace("query", input_args[kPromptFlashAttentionInputQueryIndex]->GetType());
  (void)types.emplace("key", input_args[kPromptFlashAttentionInputKeyIndex]->GetType());
  (void)types.emplace("value", input_args[kPromptFlashAttentionInputValueIndex]->GetType());
  const std::set<TypePtr> valid_types = {kFloat16, kBFloat16, kInt8};
  auto type = CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, op_name);
  if ((*type == *kInt8) && !CheckOptinalNone(input_args[kPromptFlashAttentionInputQuantScale1Index]) &&
      !CheckOptinalNone(input_args[kPromptFlashAttentionInputDeqScale1Index]) &&
      !CheckOptinalNone(input_args[kPromptFlashAttentionInputDeqScale2Index])) {
    return kFloat16;
  }
  return type;
}

TypePtr PromptFlashAttentionFuncImpl::InferType(const PrimitivePtr &primitive,
                                                const std::vector<AbstractBasePtr> &input_args) const {
  return std::make_shared<TensorType>(PromptFlashAttentionInferType(primitive, input_args));
}
}  // namespace ops
}  // namespace mindspore

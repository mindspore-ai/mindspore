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

#include <set>
#include <string>
#include <sstream>
#include "ops/prompt_flash_attention.h"
#include "abstract/ops/primitive_infer_map.h"
#include "ops/nn_ops.h"
#include "utils/check_convert_utils.h"
#include "utils/shape_utils.h"
#include "ops/op_utils.h"
#include "ops/primitive_c.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t kInputQueryBSHRankPFA = 3;
constexpr size_t kInputQueryBNSDRankPFA = 4;
constexpr int64_t SPARSE_LEFTUP_ATTENTION_SIZE = 2048;
enum SparseMode { SPARSE_MODE_0, SPARSE_MODE_1, SPARSE_MODE_2, SPARSE_MODE_3 };
constexpr int64_t ALIGN_BFLOAT_16 = 16;

ShapeValueDType GetDimension(const std::vector<ShapeValueDType> &dimensions, const std::string &op_name,
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
  if (input_arg->BuildType()->type_id() != kMetaTypeNone) {
    auto val_ptr = input_arg->BuildValue();
    if (val_ptr->isa<ValueSequence>()) {
      auto seq_length_vec = GetValue<std::vector<int64_t>>(val_ptr);
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
      auto actual_seq_length_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_arg->BuildShape())[kShape];
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
  auto sparse_mode = GetValue<int64_t>(primitive->GetAttr("sparse_mode"));
  if (sparse_mode != SPARSE_MODE_0 && sparse_mode != SPARSE_MODE_2 && sparse_mode != SPARSE_MODE_3) {
    MS_LOG(EXCEPTION) << "For primitive[" << op_name << "], sparse_mode must be 0 or 2 but got" << sparse_mode;
  }
  std::vector<ShapeVector> expect_mask_shapes = {
    {q_s, kv_s}, {1, q_s, kv_s}, {b, q_s, kv_s}, {b, 1, q_s, kv_s}, {1, 1, q_s, kv_s}};
  if (sparse_mode == SPARSE_MODE_2 || sparse_mode == SPARSE_MODE_3) {
    expect_mask_shapes = {{SPARSE_LEFTUP_ATTENTION_SIZE, SPARSE_LEFTUP_ATTENTION_SIZE},
                          {1, SPARSE_LEFTUP_ATTENTION_SIZE, SPARSE_LEFTUP_ATTENTION_SIZE},
                          {1, 1, SPARSE_LEFTUP_ATTENTION_SIZE, SPARSE_LEFTUP_ATTENTION_SIZE}};
  }
  auto atten_mask_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(
    input_args[kPromptFlashAttentionInputAttnMaskIndex]->BuildShape())[kShape];
  if (!atten_mask_shape.empty() && !IsDynamicRank(atten_mask_shape)) {
    if (!CheckTenSorShape(atten_mask_shape, expect_mask_shapes)) {
      MS_LOG(EXCEPTION) << "For primitive[" << op_name << "], atten_mask shape:  " << atten_mask_shape
                        << " dont match any of expect shape: " << expect_mask_shapes;
    }
  }

  CheckActuaSeqLength(input_args[kPromptFlashAttentionInputActualSeqLengthsIndex], q_s, b, op_name,
                      "actual_seq_lengths");
  CheckActuaSeqLength(input_args[kPromptFlashAttentionInputActualSeqLengthsKvIndex], kv_s, b, op_name,
                      "actual_seq_lengths_kv");
}

abstract::TupleShapePtr GetInputsShape(const std::vector<AbstractBasePtr> &input_args, const std::string &input_layout,
                                       const std::string &op_name) {
  auto query_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(
    input_args[kPromptFlashAttentionInputQueryIndex]->BuildShape())[kShape];
  auto key_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(
    input_args[kPromptFlashAttentionInputKeyIndex]->BuildShape())[kShape];
  auto value_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(
    input_args[kPromptFlashAttentionInputValueIndex]->BuildShape())[kShape];

  bool qeury_rank_is_dyn = IsDynamicRank(query_shape);
  bool key_rank_is_dyn = IsDynamicRank(key_shape);
  bool value_rank_is_dyn = IsDynamicRank(value_shape);
  size_t temp_rank = input_layout == "BSH" ? kInputQueryBSHRankPFA : kInputQueryBNSDRankPFA;
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
  if (key_shape.size() == 1 && value_shape.size() == 1 && key_shape[0] == 0 && value_shape[0] == 0) {
    return std::make_shared<abstract::TupleShape>(input_shape_ptr_list);
  }
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
  auto input_shapes = GetInputsShape(input_args, "BSH", op_name);
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
  auto dim_b = GetDimension({query_shape[b_index], key_shape[b_index], value_shape[b_index]}, op_name, "B");
  auto dim_q_s = query_shape[s_index];
  auto dim_kv_s = GetDimension({key_shape[s_index], value_shape[s_index]}, op_name, "KV_S");
  int64_t q_h = abstract::Shape::kShapeDimAny;
  int64_t kv_h = abstract::Shape::kShapeDimAny;
  if (num_heads_kv == 0) {
    q_h = GetDimension({query_shape[h_index], key_shape[h_index], value_shape[h_index]}, op_name, "H");
    kv_h = q_h;
  } else {
    q_h = query_shape[h_index];
    kv_h = GetDimension({key_shape[h_index], value_shape[h_index]}, op_name, "KV_H");
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
  auto query_dtype = input_args[kPromptFlashAttentionInputQueryIndex]->BuildType();
  CheckShapeAlign(query_dtype, dim_d, op_name);
  CheckOptinalInputShape(primitive, input_args, dim_b, dim_q_s, dim_kv_s);
  return ShapeVector{dim_b, dim_q_s, q_h};
}

ShapeVector InferShapeBNSD(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args,
                           int64_t num_heads, int64_t num_heads_kv) {
  auto op_name = primitive->name();
  auto input_shapes = GetInputsShape(input_args, "BNSD", op_name);
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

  auto dim_b = GetDimension({query_shape[b_index], key_shape[b_index], value_shape[b_index]}, op_name, "B");
  int64_t kv_n = abstract::Shape::kShapeDimAny;
  int64_t q_n = abstract::Shape::kShapeDimAny;
  if (num_heads_kv == 0) {
    q_n = GetDimension({query_shape[n_index], key_shape[n_index], value_shape[n_index]}, op_name, "N");
    kv_n = q_n;
  } else {
    q_n = query_shape[n_index];
    kv_n = GetDimension({key_shape[n_index], value_shape[n_index]}, op_name, "KV_N");
  }
  auto dim_q_s = query_shape[s_index];
  auto dim_kv_s = GetDimension({key_shape[s_index], value_shape[s_index]}, op_name, "KV_S");
  auto dim_d = GetDimension({query_shape[d_index], key_shape[d_index], value_shape[d_index]}, op_name, "D");
  if (q_n != abstract::Shape::kShapeDimAny && q_n != num_heads) {
    MS_LOG(EXCEPTION) << "For primitive[" << op_name << "], N must be equal num_heads, but got " << q_n << " and "
                      << num_heads;
  }
  if (num_heads_kv != 0 && kv_n != abstract::Shape::kShapeDimAny && num_heads_kv != kv_n) {
    MS_LOG(EXCEPTION) << "For primitive[" << op_name << "], KV_N must equal num_key_value_heads, but got " << kv_n
                      << " and " << num_heads_kv;
  }
  auto query_dtype = input_args[kPromptFlashAttentionInputQueryIndex]->BuildType();
  CheckShapeAlign(query_dtype, dim_d, op_name);
  CheckOptinalInputShape(primitive, input_args, dim_b, dim_q_s, dim_kv_s);
  return ShapeVector{dim_b, q_n, dim_q_s, dim_d};
}

abstract::ShapePtr PromptFlashAttentionInferShape(const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kPromptFlashAttentionInputsNum, op_name);
  auto input_layout = GetValue<std::string>(primitive->GetAttr("input_layout"));
  auto num_heads = GetValue<int64_t>(primitive->GetAttr("num_heads"));
  auto num_heads_kv = GetValue<int64_t>(primitive->GetAttr("num_key_value_heads"));
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
  if (input_layout == "BSH") {
    attention_out_shape = InferShapeBSH(primitive, input_args, num_heads, num_heads_kv);
  } else if (input_layout == "BNSD") {
    attention_out_shape = InferShapeBNSD(primitive, input_args, num_heads, num_heads_kv);
  } else {
    attention_out_shape = {abstract::Shape::kShapeRankAny};
  }
  return std::make_shared<abstract::Shape>(attention_out_shape);
}

TypePtr PromptFlashAttentionInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = prim->name();
  std::map<std::string, TypePtr> types;
  (void)types.emplace("query", input_args[kPromptFlashAttentionInputQueryIndex]->BuildType());
  (void)types.emplace("key", input_args[kPromptFlashAttentionInputKeyIndex]->BuildType());
  (void)types.emplace("value", input_args[kPromptFlashAttentionInputValueIndex]->BuildType());
  const std::set<TypePtr> valid_types = {kFloat16, kBFloat16};
  auto type = CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, op_name);
  return type;
}
}  // namespace

AbstractBasePtr PromptFlashAttentionInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kPromptFlashAttentionInputsNum, primitive->name());
  auto infer_shape = PromptFlashAttentionInferShape(primitive, input_args);
  auto infer_type = PromptFlashAttentionInferType(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(PromptFlashAttention, BaseOperator);

// AG means auto generated
class MIND_API AGPromptFlashAttentionInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return PromptFlashAttentionInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return PromptFlashAttentionInferType(primitive, input_args);
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return PromptFlashAttentionInfer(engine, primitive, input_args);
  }

  std::set<int64_t> GetValueDependArgIndices() const override { return {4, 5}; }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(PromptFlashAttention, prim::kPrimPromptFlashAttention, AGPromptFlashAttentionInfer,
                                 false);
}  // namespace ops
}  // namespace mindspore

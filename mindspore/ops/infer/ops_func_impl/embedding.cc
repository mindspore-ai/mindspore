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

#include "infer/ops_func_impl/embedding.h"
#include <memory>
#include <vector>
#include <utility>
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "ops_utils/op_constants.h"

namespace mindspore {
namespace ops {
BaseShapePtr EmbeddingFuncImpl::InferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  auto input_shape = input_args[kIndex0]->GetShape()->GetShapeVector();
  auto &weight_shape = input_args[kIndex1]->GetShape()->GetShapeVector();

  int64_t embedding_dim = abstract::Shape::kShapeDimAny;
  if (MS_LIKELY(!IsDynamicRank(weight_shape))) {
    auto constexpr kWeightRank = 2;
    MS_CHECK_VALUE(weight_shape.size() == kWeightRank,
                   CheckAndConvertUtils::FormatCheckIntegerMsg("shape of weight", weight_shape.size(), kEqual,
                                                               kWeightRank, primitive));
    embedding_dim = weight_shape.back();
  }

  if (MS_UNLIKELY(IsDynamicRank(input_shape))) {
    return std::make_shared<abstract::Shape>(input_shape);
  }

  input_shape.emplace_back(embedding_dim);
  return std::make_shared<abstract::Shape>(input_shape);
}

TypePtr EmbeddingFuncImpl::InferType(const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) const {
  auto input_type = input_args[kIndex0]->GetType();
  auto tensor_type = input_type->cast<TensorTypePtr>();
  if (tensor_type != nullptr && tensor_type->element() != nullptr) {
    auto type_id = tensor_type->element()->type_id();
    if (type_id != kNumberTypeInt32 && type_id != kNumberTypeInt64) {
      MS_EXCEPTION(TypeError) << "For '" << primitive->name()
                              << "', the data type of 'input' must be int32 or int64, but got "
                              << mindspore::TypeIdToString(type_id, true) << ".";
    }
  }
  return input_args[kIndex1]->GetType();
}

int32_t EmbeddingFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  auto weight_shape = input_args[kIndex1]->GetShape()->GetShapeVector();
  if (IsDynamicRank(weight_shape) || weight_shape.size() == 0 || weight_shape[0] == abstract::Shape::kShapeDimAny) {
    return OP_CHECK_RETRY;
  }

  auto num_embedding = weight_shape[0];
  if (num_embedding <= 0) {
    MS_EXCEPTION(ValueError) << "For primitive[" << primitive->name()
                             << "], the first dim of weight.shape must be greater than 0, but got shape: "
                             << weight_shape << ".";
  }

  auto padding_idx_type = input_args[kIndex2]->GetType();
  if (!padding_idx_type->isa<TypeNone>()) {
    auto padding_idx_opt = GetScalarValue<int64_t>(input_args[kIndex2]->GetValue());
    if (!padding_idx_opt.has_value()) {
      return OP_CHECK_RETRY;
    }
    auto padding_idx_value = padding_idx_opt.value();
    MS_CHECK_VALUE(padding_idx_value < num_embedding && padding_idx_value >= -num_embedding,
                   CheckAndConvertUtils::FormatCheckInRangeMsg("padding_idx", padding_idx_value, kIncludeLeft,
                                                               {-num_embedding, num_embedding}, primitive));
  }

  return OP_CHECK_SUCCESS;
}

ShapeArray EmbeddingFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  auto input_tensor = input_values[kIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  auto input_shape = input_tensor->shape();

  auto weight_tensor = input_values[kIndex1]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(weight_tensor);
  auto &weight_shape = weight_tensor->shape();

  auto constexpr kWeightRank = 2;
  MS_CHECK_VALUE(weight_shape.size() == kWeightRank,
                 CheckAndConvertUtils::FormatCheckIntegerMsg("shape of weight", weight_shape.size(), kEqual,
                                                             kWeightRank, primitive));

  auto num_embedding = weight_shape[0];
  auto padding_idx_opt = GetScalarValue<int64_t>(input_values[kIndex2]);
  MS_ASSERT(padding_idx_opt.has_value());
  auto padding_idx_value = padding_idx_opt.value();
  MS_CHECK_VALUE(padding_idx_value < num_embedding && padding_idx_value >= -num_embedding,
                 CheckAndConvertUtils::FormatCheckInRangeMsg("padding_idx", padding_idx_value, kIncludeLeft,
                                                             {-num_embedding, num_embedding}, primitive));

  auto embedding_dim = weight_shape.back();
  input_shape.emplace_back(embedding_dim);
  return {std::move(input_shape)};
}

TypePtrList EmbeddingFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  auto input_tensor = input_values[kIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  auto type_id = input_tensor->Dtype()->type_id();
  if (type_id != kNumberTypeInt32 && type_id != kNumberTypeInt64) {
    MS_EXCEPTION(TypeError) << "For 'Embedding', the data type of 'input' must be int32 or int64, but got "
                            << mindspore::TypeIdToString(type_id, true) << ".";
  }
  auto weight_tensor = input_values[kIndex1]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(weight_tensor);
  return {weight_tensor->Dtype()};
}
}  // namespace ops
}  // namespace mindspore

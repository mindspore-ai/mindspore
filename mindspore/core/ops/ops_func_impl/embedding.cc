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

#include "ops/ops_func_impl/embedding.h"
#include <vector>
#include <memory>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr EmbeddingFuncImpl::InferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  auto input_shape = input_args[kIndex0]->GetShape()->GetShapeVector();
  auto &weight_shape = input_args[kIndex1]->GetShape()->GetShapeVector();
  int64_t last_dim = abstract::Shape::kShapeDimAny;
  auto constexpr kWeightRank = 2;

  if (!IsDynamicRank(weight_shape)) {
    MS_CHECK_VALUE(weight_shape.size() == kWeightRank,
                   CheckAndConvertUtils::FormatCheckIntegerMsg("shape of weight", weight_shape.size(), kEqual,
                                                               kWeightRank, primitive));
    last_dim = weight_shape.back();
  }

  if (IsDynamic(input_shape)) {
    return std::make_shared<abstract::Shape>(input_shape);
  }

  input_shape.emplace_back(last_dim);
  return std::make_shared<abstract::Shape>(input_shape);
}

TypePtr EmbeddingFuncImpl::InferType(const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[kIndex1]->GetType()->Clone();
}

int32_t EmbeddingFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  auto weight_shape = input_args[kIndex1]->GetShape()->GetShapeVector();
  if (IsDynamicRank(weight_shape) || weight_shape.size() == 0 || weight_shape[0] == abstract::Shape::kShapeDimAny) {
    return OP_CHECK_RETRY;
  }

  auto num_weights = weight_shape[0];
  if (num_weights <= 0) {
    MS_EXCEPTION(ValueError) << "For primitive[" << primitive->name()
                             << "], the first dim of weight.shape must be greater than 0, but got shape: "
                             << weight_shape << ".";
  }

  auto padding_idx_type = input_args[kIndex2]->GetType();
  if (!padding_idx_type->isa<TypeNone>()) {
    auto value = input_args[kIndex2]->GetValue();
    MS_EXCEPTION_IF_NULL(value);
    auto padding_idx_opt = GetScalarValue<int64_t>(value);
    if (!padding_idx_opt.has_value()) {
      return OP_CHECK_RETRY;
    }
    auto padding_idx_value = padding_idx_opt.value();
    MS_CHECK_VALUE(padding_idx_value < num_weights && padding_idx_value >= -num_weights,
                   CheckAndConvertUtils::FormatCheckInRangeMsg("padding_idx", padding_idx_value, kIncludeLeft,
                                                               {-num_weights, num_weights}, primitive));
  }

  return OP_CHECK_SUCCESS;
}
}  // namespace ops
}  // namespace mindspore

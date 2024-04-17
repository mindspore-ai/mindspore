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

#include "ops/ops_func_impl/nllloss.h"
#include <algorithm>
#include <memory>
#include "abstract/dshape.h"
#include "mindapi/base/types.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {

void CheckNLLLossShapeValid(const PrimitivePtr &primitive, const ShapeVector &logits_shape,
                            const ShapeVector &labels_shape, const ShapeVector &weight_shape) {
  if (logits_shape.size() == 1) {
    if (logits_shape[0] > abstract::Shape::kShapeDimAny) {
      if (labels_shape[0] > abstract::Shape::kShapeDimAny) {
        MS_CHECK_VALUE(labels_shape[0] == 1, CheckAndConvertUtils::FormatCheckIntegerMsg(
                                               "labels shape", labels_shape[0], kEqual, 1, primitive));
      }
      if (weight_shape[0] > abstract::Shape::kShapeDimAny) {
        MS_CHECK_VALUE(weight_shape[0] == logits_shape[0],
                       CheckAndConvertUtils::FormatCheckIntegerMsg("weight shape", weight_shape[0], kEqual,
                                                                   logits_shape[0], primitive));
      }
    }
  } else if (logits_shape.size() == 2) {
    if (logits_shape[0] > abstract::Shape::kShapeDimAny && labels_shape[0] > abstract::Shape::kShapeDimAny) {
      MS_CHECK_VALUE(labels_shape[0] == logits_shape[0],
                     CheckAndConvertUtils::FormatCheckIntegerMsg("labels shape", labels_shape[0], kEqual,
                                                                 logits_shape[0], primitive));
    }
    if (logits_shape[1] > abstract::Shape::kShapeDimAny && weight_shape[0] > abstract::Shape::kShapeDimAny) {
      MS_CHECK_VALUE(weight_shape[0] == logits_shape[1],
                     CheckAndConvertUtils::FormatCheckIntegerMsg("weight shape", weight_shape[0], kEqual,
                                                                 logits_shape[1], primitive));
    }
  }
}

BaseShapePtr NLLLossFuncImpl::InferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  auto logits_shape_ptr = input_args[kInputIndex0]->GetShape();
  auto labels_shape_ptr = input_args[kInputIndex1]->GetShape();
  auto weight_shape_ptr = input_args[kInputIndex2]->GetShape();
  auto logits_shape = logits_shape_ptr->GetShapeVector();
  auto labels_shape = labels_shape_ptr->GetShapeVector();
  auto weight_shape = weight_shape_ptr->GetShapeVector();

  const size_t x_rank = 1;
  const size_t DIM_2 = 2;
  MS_CHECK_VALUE(labels_shape.size() == x_rank, CheckAndConvertUtils::FormatCheckIntegerMsg(
                                                  "target_rank", labels_shape.size(), kEqual, x_rank, primitive));
  MS_CHECK_VALUE(weight_shape.size() == x_rank, CheckAndConvertUtils::FormatCheckIntegerMsg(
                                                  "weight_rank", weight_shape.size(), kEqual, x_rank, primitive));
  MS_CHECK_VALUE(logits_shape.size() >= x_rank && logits_shape.size() <= DIM_2,
                 CheckAndConvertUtils::FormatCheckInRangeMsg("logits_shape_rank", logits_shape.size(), kIncludeBoth,
                                                             {1, 2}, primitive));
  CheckNLLLossShapeValid(primitive, logits_shape, labels_shape, weight_shape);
  ShapeVector loss_shape;
  ShapeVector total_weight_shape;
  abstract::ShapePtr total_weight_shape_ptr = std::make_shared<abstract::TensorShape>(total_weight_shape);
  auto reduction_value = input_args[kInputIndex3]->GetValue();
  auto reduction_opt = GetScalarValue<int64_t>(reduction_value);
  if (!reduction_opt.has_value()) {
    loss_shape.push_back(abstract::Shape::kShapeDimAny);
    auto loss_shape_ptr = std::make_shared<abstract::TensorShape>(loss_shape);
    return std::make_shared<abstract::TupleShape>(
      std::vector<abstract::BaseShapePtr>{loss_shape_ptr, total_weight_shape_ptr});
  }

  auto reduce_value_enum = static_cast<Reduction>(reduction_opt.value());
  if ((reduce_value_enum == Reduction::REDUCTION_SUM) || (reduce_value_enum == Reduction::MEAN)) {
    // shape () means 0D tensor.
    auto loss_shape_ptr = std::make_shared<abstract::TensorShape>(loss_shape);
    return std::make_shared<abstract::TupleShape>(
      std::vector<abstract::BaseShapePtr>{loss_shape_ptr, total_weight_shape_ptr});
  }

  if (reduce_value_enum == Reduction::NONE) {
    if (logits_shape.size() == DIM_2) {
      loss_shape.push_back(
        std::max({logits_shape[kInputIndex0], labels_shape[kInputIndex0], abstract::Shape::kShapeDimAny}));
    } else {
      if (logits_shape[0] == abstract::Shape::kShapeRankAny) {
        loss_shape.push_back(std::max(labels_shape[kInputIndex0], abstract::Shape::kShapeDimAny));
      } else {
        loss_shape.push_back(1);  // logits_shape is 1D
      }
    }
  }
  abstract::ShapePtr loss_shape_ptr = std::make_shared<abstract::TensorShape>(loss_shape);
  return std::make_shared<abstract::TupleShape>(
    std::vector<abstract::BaseShapePtr>{loss_shape_ptr, total_weight_shape_ptr});
}

TypePtr NLLLossFuncImpl::InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const {
  auto logits_data_type = input_args[kInputIndex0]->GetType();
  auto labels_data_type = input_args[kIndex1]->GetType();
  auto weight_data_type = input_args[kInputIndex2]->GetType();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("logits", logits_data_type, valid_types, prim->name());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("labels", labels_data_type, {kInt32, kInt64}, prim->name());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("weight", weight_data_type, valid_types, prim->name());
  return std::make_shared<Tuple>(std::vector<TypePtr>{logits_data_type->Clone(), weight_data_type->Clone()});
}
}  // namespace ops
}  // namespace mindspore

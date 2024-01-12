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

#include "ops/ops_func_impl/nllloss_grad.h"
#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include "abstract/dshape.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
void CheckNLLLossGradShapeValid(const PrimitivePtr &primitive, const ShapeVector &logits_shape,
                                const ShapeVector &labels_shape, const ShapeVector &weight_shape) {
  if (logits_shape.size() == 1) {
    if (logits_shape[0] > abstract::Shape::kShapeDimAny) {
      if (labels_shape[0] > abstract::Shape::kShapeDimAny) {
        MS_CHECK_VALUE(labels_shape[0] == 1, CheckAndConvertUtils::FormatCheckIntegerMsg(
                                               "labels_shape", labels_shape[0], kEqual, 1, primitive));
      }
      if (weight_shape[0] > abstract::Shape::kShapeDimAny) {
        MS_CHECK_VALUE(weight_shape[0] == logits_shape[0],
                       CheckAndConvertUtils::FormatCheckIntegerMsg("weight_shape", weight_shape[0], kEqual,
                                                                   logits_shape[0], primitive));
      }
    }
  } else if (logits_shape.size() == 2) {
    if (logits_shape[0] > abstract::Shape::kShapeDimAny && labels_shape[0] > abstract::Shape::kShapeDimAny) {
      MS_CHECK_VALUE(labels_shape[0] == logits_shape[0],
                     CheckAndConvertUtils::FormatCheckIntegerMsg("labels_shape", labels_shape[0], kEqual,
                                                                 logits_shape[0], primitive));
    }
    if (logits_shape[1] > abstract::Shape::kShapeDimAny && weight_shape[0] > abstract::Shape::kShapeDimAny) {
      MS_CHECK_VALUE(weight_shape[0] == logits_shape[1],
                     CheckAndConvertUtils::FormatCheckIntegerMsg("weight_shape", weight_shape[0], kEqual,
                                                                 logits_shape[1], primitive));
    }
  }
}

BaseShapePtr NLLLossGradFuncImpl::InferShape(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) const {
  auto logits_shape_ptr = input_args[kIndex0]->GetShape();
  auto labels_shape_ptr = input_args[kIndex2]->GetShape();
  auto weight_shape_ptr = input_args[kIndex3]->GetShape();
  auto logits_shape = logits_shape_ptr->GetShapeVector();
  auto labels_shape = labels_shape_ptr->GetShapeVector();
  auto weight_shape = weight_shape_ptr->GetShapeVector();

  MS_CHECK_VALUE(labels_shape.size() == 1,
                 CheckAndConvertUtils::FormatCheckIntegerMsg("labels_rank", labels_shape.size(), kEqual, 1, primitive));
  MS_CHECK_VALUE(weight_shape.size() == 1,
                 CheckAndConvertUtils::FormatCheckIntegerMsg("weight_rank", weight_shape.size(), kEqual, 1, primitive));
  MS_CHECK_VALUE(logits_shape.size() >= 1 && logits_shape.size() <= 2,
                 CheckAndConvertUtils::FormatCheckInRangeMsg("logits_shape_rank", logits_shape.size(), kIncludeBoth,
                                                             {1, 2}, primitive));
  ShapeVector dyn_out(2, abstract::Shape::kShapeDimAny);
  if (logits_shape.size() == 1) {
    if (logits_shape[0] > abstract::Shape::kShapeRankAny) {
      dyn_out[0] = 1;
      dyn_out[1] = std::max({logits_shape[0], weight_shape[0], abstract::Shape::kShapeDimAny});
    } else if (logits_shape[0] == abstract::Shape::kShapeRankAny) {
      dyn_out[0] = std::max(labels_shape[0], abstract::Shape::kShapeDimAny);
      dyn_out[1] = std::max(weight_shape[0], abstract::Shape::kShapeDimAny);
    }
  } else {
    dyn_out[0] = std::max({logits_shape[0], labels_shape[0], abstract::Shape::kShapeDimAny});
    dyn_out[1] = std::max({logits_shape[1], weight_shape[0], abstract::Shape::kShapeDimAny});
  }
  CheckNLLLossGradShapeValid(primitive, logits_shape, labels_shape, weight_shape);
  return std::make_shared<abstract::TensorShape>(std::move(dyn_out));
}

TypePtr NLLLossGradFuncImpl::InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const {
  auto x_dtype = input_args[kInputIndex0]->GetType();
  auto y_grad_dtype = input_args[kInputIndex1]->GetType();
  auto t_dtype = input_args[kInputIndex2]->GetType();
  auto w_dtype = input_args[kInputIndex3]->GetType();
  auto tw_dtype = input_args[kInputIndex4]->GetType();
  std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("logits", x_dtype, valid_types, prim->name());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("loss's grad", y_grad_dtype, valid_types, prim->name());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("labels", t_dtype, {kInt32, kInt64}, prim->name());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("weight", w_dtype, valid_types, prim->name());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("total_weight", tw_dtype, valid_types, prim->name());

  std::map<std::string, TypePtr> types;
  (void)types.emplace("weight", w_dtype);
  (void)types.emplace("total_weight", tw_dtype);
  (void)CheckAndConvertUtils::CheckTypeSame(types, prim->name());
  return x_dtype->Clone();
}
}  // namespace ops
}  // namespace mindspore

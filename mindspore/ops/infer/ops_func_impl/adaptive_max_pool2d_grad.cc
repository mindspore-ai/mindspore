/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "infer/ops_func_impl/adaptive_max_pool2d_grad.h"
#include "utils/check_convert_utils.h"
#include "utils/shape_utils.h"
#include "mindspore/ops/ops_utils/op_utils.h"

namespace mindspore {
namespace ops {
ShapeArray AdaptiveMaxPool2DGradFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                     const InferInfoPtrList &input_infos) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  auto y_grad_shape = input_infos[kInputIndex0]->GetShape();
  auto x_shape = input_infos[kInputIndex1]->GetShape();
  auto argmax_shape = input_infos[kInputIndex2]->GetShape();

  std::vector<ShapeVector> all_shapes = {y_grad_shape, x_shape, argmax_shape};
  auto is_dynamic_rank = std::any_of(all_shapes.begin(), all_shapes.end(), IsDynamicRank);
  if (is_dynamic_rank) {
    return {{abstract::TensorShape::kShapeRankAny}};
  }

  const int64_t y_grad_dims = SizeToLong(y_grad_shape.size());
  const int64_t x_dims = SizeToLong(x_shape.size());
  const int64_t argmax_dims = SizeToLong(argmax_shape.size());
  (void)CheckAndConvertUtils::CheckInteger("y_grad_dims", y_grad_dims, kEqual, x_dims, op_name);
  (void)CheckAndConvertUtils::CheckInteger("argmax_dims", argmax_dims, kEqual, x_dims, op_name);

  CheckAndConvertUtils::CheckInRange("y_grad_dim", y_grad_dims, kIncludeBoth, {3, 4}, op_name);
  CheckAndConvertUtils::CheckInRange("x_dim", x_dims, kIncludeBoth, {3, 4}, op_name);
  CheckAndConvertUtils::CheckInRange("argmax_dim", argmax_dims, kIncludeBoth, {3, 4}, op_name);

  auto is_dynamic = IsDynamic(y_grad_shape) || IsDynamic(argmax_shape);
  if (!is_dynamic && y_grad_shape != argmax_shape) {
    MS_EXCEPTION(ValueError) << "For '" << op_name
                             << "', the shape of 'y_grad' should be consistent with the shape of 'argmax'.";
  }

  return {x_shape};
}

std::vector<TypeId> AdaptiveMaxPool2DGradFuncImpl::InferType(const PrimitivePtr &primitive,
                                                             const InferInfoPtrList &input_infos) const {
  return {input_infos[kInputIndex0]->GetType()};
}
}  // namespace ops
}  // namespace mindspore

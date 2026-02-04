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
#include "infer/ops_func_impl/threshold_grad.h"
#include <utility>
#include <memory>
#include "infer/ops_func_impl/reduce_arithmetic.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "ir/dtype.h"
#include "primitive/op_name.h"
#include "utils/check_convert_utils.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore {
namespace ops {

ShapeArray ThresholdGradFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  const auto &dout = input_infos[kInputIndex0];
  const auto &y = input_infos[kInputIndex1];
  const auto dout_shape = dout->GetShape();
  const auto y_shape = y->GetShape();

  if (dout->IsDynamicRank() || y->IsDynamicRank()) {
    return {y_shape};
  } else if (dout_shape.size() != y_shape.size()) {
    MS_EXCEPTION(ValueError) << "Rank of x(" << y_shape.size() << ") and dout(" << dout_shape.size()
                             << ") not equal, primitive name: " << primitive->name() << ".";
  }

  for (size_t i = 0; i < y_shape.size(); i++) {
    if (y_shape[i] != abstract::Shape::kShapeDimAny && dout_shape[i] != abstract::Shape::kShapeDimAny &&
        y_shape[i] != dout_shape[i]) {
      MS_EXCEPTION(ValueError) << "The " << i << "th dim of x(" << y_shape[i] << ") and dout(" << dout_shape[i]
                               << ") not equal, primitive name: " << primitive->name() << ".";
    }
  }
  return {dout_shape};
}

std::vector<TypeId> ThresholdGradFuncImpl::InferType(const PrimitivePtr &primitive,
                                                     const InferInfoPtrList &input_infos) const {
  const auto &dout = input_infos[kIndex0];
  const auto &y = input_infos[kIndex1];

  const auto &dout_type = dout->GetType();
  const auto &y_type = y->GetType();

  if (dout_type != y_type) {
    MS_LOG_EXCEPTION << "For " << primitive->name() << ", the grad type must be same as input type, but got grad_type: "
                     << TypeIdToType(static_cast<TypeId>(dout_type))->ToString()
                     << " and x_type: " << TypeIdToType(static_cast<TypeId>(y_type))->ToString();
  }
  return {dout_type};
}
}  // namespace ops
}  // namespace mindspore

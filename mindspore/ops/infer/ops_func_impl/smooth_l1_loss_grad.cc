/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#include "infer/ops_func_impl/smooth_l1_loss_grad.h"
#include <algorithm>
#include <utility>
#include "abstract/dshape.h"
#include "primitive/op_name.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore {
namespace ops {
ShapeArray SmoothL1LossGradFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                const InferInfoPtrList &input_infos) const {
  auto &prediction = input_infos[kInputIndex0];
  auto &target = input_infos[kInputIndex1];
  const auto &beta_opt = input_infos[kInputIndex3]->GetScalarValue<float>();
  if (beta_opt.has_value() && beta_opt.value() < 0) {
    MS_EXCEPTION(RuntimeError) << "For 'SmoothL1LossGrad'"
                               << ", the values for beta not support negative values"
                               << ", but got " << beta_opt.value() << ".";
  }
  auto prediction_shape = prediction->GetShape();
  auto target_shape = target->GetShape();
  if (!(prediction->IsDynamic() || target->IsDynamic())) {
    MS_CHECK_VALUE(
      prediction_shape == target_shape,
      CheckAndConvertUtils::FormatCheckMsg("prediction_shape", prediction_shape, kEqual, target_shape, primitive));
  }
  return {prediction_shape};
}

std::vector<TypeId> SmoothL1LossGradFuncImpl::InferType(const PrimitivePtr &primitive,
                                                        const InferInfoPtrList &input_infos) const {
  return {input_infos[kInputIndex0]->GetType()};
}
}  // namespace ops
}  // namespace mindspore

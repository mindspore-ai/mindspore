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

#include "infer/ops_func_impl/nllloss.h"
#include <set>
#include <algorithm>
#include <memory>
#include <string>
#include "abstract/dshape.h"
#include "mindapi/base/types.h"
#include "primitive/op_name.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "ops_utils/op_constants.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace ops {
ShapeArray NLLLossFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  auto target_shape = input_infos[kInputIndex1]->GetShape();
  ShapeVector weight_out_shape = {};
  auto reduction = static_cast<Reduction>(input_infos[kInputIndex3]->GetScalarValue<int64_t>().value());
  if (reduction == Reduction::NONE) {
    if (target_shape.size() == kDim1 && target_shape[0] == abstract::TensorShape::kShapeRankAny) {
      target_shape = {-1};
    }
    return {target_shape, weight_out_shape};
  } else {
    ShapeVector out_shape = {};
    return {out_shape, weight_out_shape};
  }
}

std::vector<TypeId> NLLLossFuncImpl::InferType(const PrimitivePtr &primitive,
                                               const InferInfoPtrList &input_infos) const {
  auto input_data_type = input_infos[kInputIndex0]->GetType();
  return {input_data_type, input_data_type};
}
}  // namespace ops
}  // namespace mindspore

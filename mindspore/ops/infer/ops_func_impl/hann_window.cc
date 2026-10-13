/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include "infer/ops_func_impl/hann_window.h"

#include <set>
#include <vector>

#include "mindspore/ops/ops_utils/op_utils.h"
#include "ops_utils/op_constants.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
ShapeArray HannWindowFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  const auto window_length = input_infos[kInputIndex0]->GetScalarValue<int64_t>();
  if (!window_length.has_value()) {
    return {{abstract::TensorShape::kShapeDimAny}};
  }
  MS_CHECK_VALUE(window_length.value() >= 0, CheckAndConvertUtils::FormatCheckIntegerMsg(
                                               "window_length", window_length.value(), kGreaterEqual, 0, primitive));
  return {{window_length.value()}};
}

std::vector<TypeId> HannWindowFuncImpl::InferType(const PrimitivePtr &primitive,
                                                  const InferInfoPtrList &input_infos) const {
  TypeId output_type = kNumberTypeFloat32;
  if (!input_infos[kInputIndex2]->IsNone()) {
    output_type = static_cast<TypeId>(input_infos[kInputIndex2]->GetScalarValueWithCheck<int64_t>());
  }
  static const std::set<TypeId> valid_types = {kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat64,
                                               kNumberTypeBFloat16};
  (void)CheckAndConvertUtils::CheckTypeIdValid("dtype", output_type, valid_types, primitive->name());
  return {output_type};
}
}  // namespace ops
}  // namespace mindspore

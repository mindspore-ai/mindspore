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

#include "infer/ops_func_impl/abs.h"

#include "mindspore/ops/ops_utils/op_utils.h"

namespace mindspore::ops {
std::vector<TypeId> AbsFuncImpl::InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  const auto input_type = input_infos[kInputIndex0]->GetType();
  if (input_type == kNumberTypeComplex64) {
    return {kNumberTypeFloat32};
  }
  if (input_type == kNumberTypeComplex128) {
    return {kNumberTypeFloat64};
  }
  return {input_type};
}
}  // namespace mindspore::ops

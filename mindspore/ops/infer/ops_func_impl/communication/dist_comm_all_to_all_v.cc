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

#include "infer/ops_func_impl/communication/dist_comm_all_to_all_v.h"

#include <memory>
#include <vector>
#include <string>
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/ms_context.h"
#include "infer/ops_func_impl/communication/op_comm_func_impl.h"

namespace mindspore {
namespace ops {
ShapeArray DistCommAllToAllVFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                 const InferInfoPtrList &input_infos) const {
  auto recv_numel_list_opt = input_infos[kIndex4]->GetArrayValue<int64_t>();
  MS_CHECK_VALUE(recv_numel_list_opt.has_value(),
                 primitive->name() + " error: recv_numel_list input should has valid value.");

  const auto &recv_numel_list = recv_numel_list_opt.value();
  int64_t output_numel = 0;
  for (size_t i = 0; i < recv_numel_list.size(); i++) {
    if (recv_numel_list[i] < 0) {
      output_numel = abstract::Shape::kShapeDimAny;
      break;
    }
    output_numel += recv_numel_list[i];
  }
  if (output_numel == 0) {
    return {ShapeVector{}};
  }
  return {ShapeVector{output_numel}};
}

std::vector<TypeId> DistCommAllToAllVFuncImpl::InferType(const PrimitivePtr &primitive,
                                                         const InferInfoPtrList &input_infos) const {
  auto type = input_infos[kIndex1]->GetType();
  auto x_list = input_infos[kIndex0]->GetSequenceElements();
  for (size_t i = 0; i < x_list.size(); i++) {
    MS_EXCEPTION_IF_NULL(x_list[i]);
    auto out_type = x_list[i]->GetType();
    CheckInferTypes(primitive->name(), type, out_type);
  }
  return {type};
}
}  // namespace ops
}  // namespace mindspore

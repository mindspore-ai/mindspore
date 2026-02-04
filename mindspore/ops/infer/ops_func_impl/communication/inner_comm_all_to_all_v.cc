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

#include "infer/ops_func_impl/communication/inner_comm_all_to_all_v.h"

#include <memory>
#include <string>
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace ops {
BaseShapePtr InnerCommAllToAllVFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                    const std::vector<AbstractBasePtr> &input_args) const {
  auto recv_numel_list_opt = GetArrayValue<int64_t>(input_args[kInputIndex3]);
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
    return std::make_shared<abstract::TensorShape>(ShapeVector{});
  }
  return std::make_shared<abstract::TensorShape>(ShapeVector{output_numel});
}

TypePtr InnerCommAllToAllVFuncImpl::InferType(const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args) const {
  auto type = input_args[kIndex0]->GetType();
  if (MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET) != kAscendDevice) {
    (void)CheckAndConvertUtils::CheckTensorTypeValid("input", type, common_valid_types, primitive->name());
  } else {
    (void)CheckAndConvertUtils::CheckTensorTypeValid("input", type, common_valid_types_with_bool, primitive->name());
  }
  return type;
}
}  // namespace ops
}  // namespace mindspore

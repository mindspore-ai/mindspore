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

#include "plugin/device/ascend/kernel/internal/gather.h"

#include <memory>

#include "plugin/device/ascend/kernel/internal/internal_kernel_utils.h"
#include "plugin/device/ascend/kernel/internal/internal_kernel_in_out_map.h"

namespace mindspore {
namespace kernel {
internal::OpParamPtr Gather::CreateOpParam(const std::vector<KernelTensor *> &inputs,
                                           const std::vector<KernelTensor *> &outputs) {
  internal::OpParamPtr param_ptr = std::make_shared<internal::OpParam>();
  internal::GatherParam gather_param;

  auto axis_tensor = inputs.at(kIndex2);  // input 2 : axis
  if (axis_tensor->dtype_id() == TypeId::kNumberTypeInt64) {
    auto axis_list = axis_tensor->GetValue<std::vector<int64_t>>().value();
    for (auto axis : axis_list) {
      gather_param.axis.emplace_back(axis);
    }
  } else if (axis_tensor->dtype_id() == TypeId::kNumberTypeInt32) {
    auto axis_list = axis_tensor->GetValue<std::vector<int32_t>>().value();
    for (auto axis : axis_list) {
      gather_param.axis.emplace_back(axis);
    }
  } else {
    gather_param.axis = {kIndex0};
  }

  int64_t batch_dims_value = kDim0;
  auto batch_dims_tensor = inputs.at(kIndex3);
  if (batch_dims_tensor->dtype_id() == TypeId::kNumberTypeInt64) {
    batch_dims_value = batch_dims_tensor->GetValue<int64_t>().value();
  } else if (batch_dims_tensor->dtype_id() == TypeId::kNumberTypeInt32) {
    batch_dims_value = batch_dims_tensor->GetValue<int32_t>().value();
  } else {
    MS_LOG(EXCEPTION) << "For internal op 'Gather', batch_dims should be int32 or int64.";
  }
  gather_param.batchDims = batch_dims_value;

  param_ptr->specificParam = gather_param;
  param_ptr->opId = internal::OpId::Gather;
  return param_ptr;
}

MS_INTERNAL_KERNEL_FACTORY_REG(Gather, Gather);
}  // namespace kernel
}  // namespace mindspore

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

#include "plugin/device/ascend/kernel/internal/transpose.h"

#include <memory>

#include "plugin/device/ascend/kernel/internal/internal_kernel_utils.h"
#include "plugin/device/ascend/kernel/internal/internal_kernel_in_out_map.h"

namespace mindspore {
namespace kernel {
internal::OpParamPtr InternalTranspose::CreateOpParam(const std::vector<KernelTensor *> &inputs,
                                                      const std::vector<KernelTensor *> &outputs) {
  internal::OpParamPtr param_ptr = std::make_shared<internal::OpParam>();
  internal::TransposeParam transpose_param;

  auto perm_tensor = inputs.at(kIndex1);  // input 1 : perm
  if (perm_tensor->dtype_id() == TypeId::kNumberTypeInt64) {
    auto perm_list = perm_tensor->GetValue<std::vector<int64_t>>().value();
    for (auto axis : perm_list) {
      transpose_param.perm.emplace_back(axis);
    }
  } else {
    MS_LOG(EXCEPTION) << "InternalTranspose input[1] dtype is not kNumberTypeInt64";
  }

  param_ptr->specificParam = transpose_param;
  param_ptr->opId = internal::OpId::Transpose;
  return param_ptr;
}

MS_INTERNAL_KERNEL_FACTORY_REG(Transpose, InternalTranspose);
}  // namespace kernel
}  // namespace mindspore

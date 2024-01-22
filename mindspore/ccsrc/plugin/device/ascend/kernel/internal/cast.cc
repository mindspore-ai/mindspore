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

#include "plugin/device/ascend/kernel/internal/cast.h"
#include "plugin/device/ascend/kernel/internal/internal_kernel_utils.h"
#include "include/param/cast_param.h"

namespace mindspore {
namespace kernel {
internal::OpParamPtr InternalCast::CreateOpParam(const std::vector<KernelTensor *> &inputs,
                                                 const std::vector<KernelTensor *> &outputs) {
  auto param_ptr = std::make_shared<internal::CastParam>();
  param_ptr->in_dtype_ = InternalKernelUtils::ToInternalDType(inputs[0]->dtype_id());
  param_ptr->out_dtype_ = InternalKernelUtils::ToInternalDType(outputs[0]->dtype_id());
  param_ptr->opId = internal::OpId::Cast;
  return std::static_pointer_cast<internal::OpParam>(param_ptr);
}

void InternalCast::SetInOutIdx() {
  inputsIdxMap_[0] = 0;
  outputsIdxMap_[0] = 0;
}

MS_INTERNAL_KERNEL_FACTORY_REG(Cast, InternalCast);
}  // namespace kernel
}  // namespace mindspore

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

#include "plugin/device/ascend/kernel/internal/paged_attention.h"

#include <memory>

#include "plugin/device/ascend/kernel/internal/internal_kernel_utils.h"

namespace mindspore {
namespace kernel {
internal::OpParamPtr InternalPagedAttention::CreateOpParam(const std::vector<KernelTensor *> &inputs,
                                                           const std::vector<KernelTensor *> &outputs) {
  internal::OpParamPtr param_ptr = std::make_shared<internal::OpParam>();
  param_ptr->opId = internal::OpId::PagedAttention;
  internal::MixParam op_param;
  op_param.mixType = internal::MixParam::MixType::MIX_PAGED_ATTENTION_MASK_ND;
  op_param.headSize = static_cast<int32_t>(inputs[5]->GetValueWithCheck<int64_t>());
  op_param.tor = inputs[6]->GetValueWithCheck<float>();
  op_param.kvHead = static_cast<int32_t>(inputs[7]->GetValueWithCheck<int64_t>());

  param_ptr->specificParam = op_param;
  return param_ptr;
}

void InternalPagedAttention::SetInOutIdx() {
  inputsIdxMap_[kIndex0] = kIndex0;
  inputsIdxMap_[kIndex1] = kIndex1;
  inputsIdxMap_[kIndex2] = kIndex2;
  inputsIdxMap_[kIndex3] = kIndex4;
  inputsIdxMap_[kIndex4] = kIndex3;
  outputsIdxMap_[kIndex0] = kIndex0;
}

MS_INTERNAL_KERNEL_FACTORY_REG(PagedAttention, InternalPagedAttention);
}  // namespace kernel
}  // namespace mindspore

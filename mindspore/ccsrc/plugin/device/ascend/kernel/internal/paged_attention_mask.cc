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
#include <memory>
#include "plugin/device/ascend/kernel/internal/paged_attention_mask.h"
namespace mindspore {
namespace kernel {
internal::OpParamPtr InternalPagedAttentionMask::CreateOpParam(const std::vector<KernelTensor *> &inputs,
                                                               const std::vector<KernelTensor *> &outputs) {
  internal::OpParamPtr param_ptr = std::make_shared<internal::OpParam>();
  param_ptr->opId = internal::OpId::PagedAttention;
  return param_ptr;
}

void InternalPagedAttentionMask::SetInOutIdx() {
  inputsIdxMap_[kIndex0] = kIndex0;
  inputsIdxMap_[kIndex1] = kIndex1;
  inputsIdxMap_[kIndex2] = kIndex2;
  inputsIdxMap_[kIndex3] = kIndex4;
  inputsIdxMap_[kIndex4] = kIndex3;
  inputsIdxMap_[kIndex5] = kIndex5;
  outputsIdxMap_[kIndex0] = kIndex0;
}

MS_INTERNAL_KERNEL_FACTORY_REG(PagedAttentionMask, InternalPagedAttentionMask);
}  // namespace kernel
}  // namespace mindspore

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
#include "plugin/device/ascend/kernel/internal/internal_kernel_in_out_map.h"

namespace mindspore {
namespace kernel {
internal::OpParamPtr InternalPagedAttention::CreateOpParam(const std::vector<KernelTensor *> &inputs,
                                                           const std::vector<KernelTensor *> &outputs) {
  internal::OpParamPtr param_ptr = std::make_shared<internal::OpParam>();
  param_ptr->opId = internal::OpId::PagedAttention;
  internal::MixParam op_param;
  op_param.mixType = internal::MixParam::MixType::MIX_PAGED_ATTENTION_MASK_ND;
  op_param.headSize = static_cast<int32_t>(inputs[kIndex5]->GetValueWithCheck<int64_t>());
  op_param.tor = inputs[kIndex6]->GetValueWithCheck<float>();
  op_param.kvHead = static_cast<int32_t>(inputs[kIndex7]->GetValueWithCheck<int64_t>());

  param_ptr->specificParam = op_param;
  return param_ptr;
}

MS_INTERNAL_KERNEL_FACTORY_REG(PagedAttention, InternalPagedAttention);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(PagedAttention, 5, 0, 1, 2, 4, 3);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(PagedAttention, 1, 0);
}  // namespace kernel
}  // namespace mindspore

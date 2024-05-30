/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/kernel/internal/split.h"

#include <memory>

#include "plugin/device/ascend/kernel/internal/internal_kernel_utils.h"
#include "plugin/device/ascend/kernel/internal/internal_kernel_in_out_map.h"

namespace mindspore {
namespace kernel {
internal::OpParamPtr InternalSplit::CreateOpParam(const std::vector<KernelTensor *> &inputs,
                                                  const std::vector<KernelTensor *> &outputs) {
  internal::OpParamPtr param_ptr = std::make_shared<internal::OpParam>();
  internal::SplitParam split_param;
  param_ptr->opId = internal::OpId::Split;

  split_param.splitDim = inputs[kIndex1]->GetValueWithCheck<int64_t>();
  split_param.splitNum = inputs[kIndex2]->GetValueWithCheck<int64_t>();

  param_ptr->specificParam = split_param;
  return param_ptr;
}

MS_INTERNAL_KERNEL_FACTORY_REG(Split, InternalSplit);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(Split, INPUT_NUM_1, INDEX_0);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(Split, INTERNEL_KERNEL_IN_OUT_MUTABLE_LENGTH);
}  // namespace kernel
}  // namespace mindspore

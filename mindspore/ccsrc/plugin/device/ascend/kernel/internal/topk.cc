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
#include "plugin/device/ascend/kernel/internal/topk.h"
namespace mindspore {
namespace kernel {
internal::OpParamPtr InternalTopK::CreateOpParam(const std::vector<KernelTensor *> &inputs,
                                                 const std::vector<KernelTensor *> &outputs) {
  internal::OpParamPtr param_ptr = std::make_shared<internal::OpParam>();
  internal::SortParam op_param;

  auto num = inputs.at(kIndex1);
  auto num_list = num->GetValue<std::vector<int32_t>>().value();
  for (auto num : num_list) {
    op_param.num.emplace_back(num);
  }

  param_ptr->specificParam = op_param;
  param_ptr->opId = internal::OpId::TopK;
  return param_ptr;
}

void InternalTopK::SetInOutIdx() {
  inputsIdxMap_[kIndex0] = kIndex0;
  outputsIdxMap_[kIndex0] = kIndex0;
  outputsIdxMap_[kIndex1] = kIndex1;
}

MS_INTERNAL_KERNEL_FACTORY_REG(TopK, InternalTopK);
}  // namespace kernel
}  // namespace mindspore
/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "kernel/host/view/kernel_mod_impl/split_view.h"

#include "kernel/host/view/view_utils.h"
#include "view/split_strides_calc.h"

namespace mindspore {
namespace kernel {

void SplitView::UpdateOutputTensorInfo(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  ops::OldTensorInfoPtr old_info = GetOldTensorInfo(inputs[kIndex0]);

  auto axis = inputs[kIndex1]->GetValueWithCheck<int64_t>();
  auto output_num = inputs[kIndex2]->GetValueWithCheck<int64_t>();

  info_ = ops::SplitProcess(old_info, axis, output_num);
  for (size_t i = 0; i < outputs.size(); i++) {
    outputs[i]->set_tensor_storage_info(info_[i]);
  }
}

void SplitView::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &outputs) {
  UpdateOutputTensorInfo(inputs, outputs);
}

MS_HOST_REG_KERNEL(SplitView, SplitView);
}  // namespace kernel
}  // namespace mindspore

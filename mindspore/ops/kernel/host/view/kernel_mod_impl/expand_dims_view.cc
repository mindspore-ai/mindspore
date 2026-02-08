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
#include "kernel/host/view/kernel_mod_impl/expand_dims_view.h"
#include <vector>
#include "kernel/host/view/view_utils.h"
#include "view/expand_dims_strides_calc.h"

namespace mindspore {
namespace kernel {

void ExpandDimsView::UpdateOutputTensorInfo(const std::vector<KernelTensor *> &inputs,
                                            const std::vector<KernelTensor *> &outputs) {
  auto axis = inputs[kIndex1]->GetValueWithCheck<int64_t>();
  const auto &input = inputs[kIndex0];
  info_ =
    ops::ExpandDimsStrideCalc(input->GetShapeVector(), GetTensorStride(input), input->tensor_storage_info(), axis);
  outputs[kIndex0]->set_tensor_storage_info(info_[0]);
}

void ExpandDimsView::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &outputs) {
  UpdateOutputTensorInfo(inputs, outputs);
}

MS_HOST_REG_KERNEL(ExpandDimsView, ExpandDimsView);
}  // namespace kernel
}  // namespace mindspore

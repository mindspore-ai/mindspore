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
#include "kernel/host/view/kernel_mod_impl/transpose_view.h"

#include "kernel/host/view/view_utils.h"
#include "view/transpose_strides_calc.h"
#include "view/view_strides_calculator.h"

namespace mindspore {
namespace kernel {

void TransposeView::UpdateOutputTensorInfo(const std::vector<KernelTensor *> &inputs,
                                           const std::vector<KernelTensor *> &outputs) {
  const auto &dims = inputs[kIndex1]->GetValueWithCheck<std::vector<int64_t>>();
  const auto &input = inputs[kIndex0];
  const auto &shape = input->GetShapeVector();
  if (dims.size() != shape.size()) {
    MS_LOG(EXCEPTION) << "dims should be same with shape size which is " << dims << " , and shape " << shape;
  }
  auto infos =
    ops::TransposeStridesCalc(input->GetShapeVector(), GetTensorStride(input), input->tensor_storage_info(), dims);
  outputs[kIndex0]->set_tensor_storage_info(infos[0]);
}

void TransposeView::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) {
  UpdateOutputTensorInfo(inputs, outputs);
}

MS_HOST_REG_KERNEL(TransposeView, TransposeView);
}  // namespace kernel
}  // namespace mindspore

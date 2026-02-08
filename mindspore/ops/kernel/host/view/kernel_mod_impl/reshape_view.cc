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
#include "kernel/host/view/kernel_mod_impl/reshape_view.h"

#include "kernel/host/view/view_utils.h"
#include "view/reshape_strides_calc.h"

namespace mindspore {
namespace kernel {

void ReshapeView::UpdateOutputTensorInfo(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &outputs) {
  // If input is not contiguous, framework will make the input contiguous in launch.
  auto ori_storage_info = inputs[0]->tensor_storage_info();
  if (ori_storage_info && !ori_storage_info->is_contiguous) {
    MS_LOG(WARNING) << "For Reshape, the input is not contiguous. It is suggested to call Contiguous "
                       "explicitly. Input shape is "
                    << ori_storage_info->shape;
    is_input_not_contiguous_ = true;
    return;
  }

  ops::OldTensorInfoPtr old_info = GetOldTensorInfo(inputs[kIndex0]);
  auto shape = inputs[kIndex1]->GetValueWithCheck<std::vector<int64_t>>();
  if (std::any_of(shape.begin(), shape.end(), [](const int &shape_i) { return shape_i < -1; })) {
    MS_EXCEPTION(ValueError) << "ReshapeView the component of shape can't be less than -1, but got " << shape;
  }

  auto info =
    ops::ReshapeStridesCalc(old_info->old_shape, old_info->old_strides, inputs[kIndex0]->tensor_storage_info(), shape);
  MS_EXCEPTION_IF_NULL(info);
  outputs[kIndex0]->set_tensor_storage_info(info);
}

void ReshapeView::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &outputs) {
  UpdateOutputTensorInfo(inputs, outputs);
}

bool ReshapeView::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                         const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  if (is_input_not_contiguous_) {
    // If input is not contiguous, framework will make a contiguous input, then we need re-compute the
    // out tensor storage info.
    UpdateOutputTensorInfo(inputs, outputs);
  }
  return true;
}

MS_HOST_REG_KERNEL(ReshapeView, ReshapeView);
}  // namespace kernel
}  // namespace mindspore

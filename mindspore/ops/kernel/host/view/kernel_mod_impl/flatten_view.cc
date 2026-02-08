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
#include "kernel/host/view/kernel_mod_impl/flatten_view.h"

#include "kernel/host/view/view_utils.h"
#include "view/reshape_strides_calc.h"

namespace mindspore {
namespace kernel {

void FlattenView::UpdateOutputTensorInfo(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &outputs) {
  ops::OldTensorInfoPtr old_info = GetOldTensorInfo(inputs[kIndex0]);
  std::vector<int64_t> shape = {-1};
  auto info =
    ops::ReshapeStridesCalc(old_info->old_shape, old_info->old_strides, inputs[kIndex0]->tensor_storage_info(), shape);
  MS_EXCEPTION_IF_NULL(info);
  outputs[kIndex0]->set_tensor_storage_info(info);
}

void FlattenView::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &outputs) {
  UpdateOutputTensorInfo(inputs, outputs);
}

MS_HOST_REG_KERNEL(FlattenView, FlattenView);
}  // namespace kernel
}  // namespace mindspore

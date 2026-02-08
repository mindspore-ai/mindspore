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
#include "kernel/host/view/kernel_mod_impl/slice_ext_view.h"

#include "abstract/ops/primitive_infer_map.h"
#include "kernel/host/view/view_utils.h"
#include "view/slice_ext_strides_calc.h"
#include "view/view_strides_calculator.h"

namespace mindspore {
namespace kernel {

void SliceExtView::UpdateOutputTensorInfo(const std::vector<KernelTensor *> &inputs,
                                          const std::vector<KernelTensor *> &outputs) {
  ops::OldTensorInfoPtr old_info = GetOldTensorInfo(inputs[kIndex0]);
  const auto dim = inputs[kIndex1]->GetValueWithCheck<int64_t>();
  const auto start = inputs[kIndex2]->GetValueWithCheck<int64_t>();
  const auto end = inputs[kIndex3]->GetValueWithCheck<int64_t>();
  const auto step = inputs[kIndex4]->GetValueWithCheck<int64_t>();
  info_ = ops::SliceExtStridesCalc(old_info->old_shape, old_info->old_strides, inputs[kIndex0]->tensor_storage_info(),
                                   dim, start, end, step);
  outputs[kIndex0]->set_tensor_storage_info(info_[0]);
}

void SliceExtView::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &outputs) {
  UpdateOutputTensorInfo(inputs, outputs);
}

MS_HOST_REG_KERNEL(SliceExtView, SliceExtView);
}  // namespace kernel
}  // namespace mindspore

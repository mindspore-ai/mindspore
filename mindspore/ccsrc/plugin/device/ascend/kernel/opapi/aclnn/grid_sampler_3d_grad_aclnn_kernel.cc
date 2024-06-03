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
#include "plugin/device/ascend/kernel/opapi/aclnn/grid_sampler_3d_grad_aclnn_kernel.h"
#include <algorithm>
#include <vector>
#include <memory>
#include <functional>
#include "ir/tensor.h"
#include "runtime/device/kernel_runtime.h"
#include "transform/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {

void GridSampler3DGradAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                               const std::vector<KernelTensor *> &outputs) {
  interpolation_mode_ = transform::ConvertKernelTensor<int64_t>(inputs[kIndex3]);
  padding_mode_ = transform::ConvertKernelTensor<int64_t>(inputs[kIndex4]);
  align_corners_ = transform::ConvertKernelTensor<bool>(inputs[kIndex5]);
  GetWorkspaceForResize(inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], interpolation_mode_, padding_mode_,
                        align_corners_, output_mask_, outputs[kIndex0], outputs[kIndex1]);
}

bool GridSampler3DGradAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &workspace,
                                     const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOp(stream_ptr, workspace, inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], interpolation_mode_, padding_mode_,
        align_corners_, output_mask_, outputs[kIndex0], outputs[kIndex1]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(GridSampler3DGrad, GridSampler3DGradAscend);
}  // namespace kernel
}  // namespace mindspore

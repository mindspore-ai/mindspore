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
#include "plugin/device/ascend/kernel/opapi/aclnn/max_pool_grad_with_indices_aclnn_kernel.h"
#include <algorithm>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include "ir/tensor.h"
#include "runtime/device/kernel_runtime.h"
#include "transform/acl_ir/acl_helper.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {

void MaxPoolGradWithIndicesAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                    const std::vector<KernelTensor *> &outputs) {
  kernel_size_ = inputs[kIndex3]->GetValueWithCheck<std::vector<int64_t>>();
  strides_ = kernel_size_;
  if (inputs[kIndex2]->type_id() != kMetaTypeNone) {
    strides_ = inputs[kIndex4]->GetValueWithCheck<std::vector<int64_t>>();
  }
  pads_ = inputs[kIndex5]->GetValueWithCheck<std::vector<int64_t>>();
  dilation_ = inputs[kIndex6]->GetValueWithCheck<std::vector<int64_t>>();
  ceil_mode_ = inputs[kIndex7]->GetValueWithCheck<bool>();
  GetWorkspaceForResize(inputs[kIndex1], inputs[kIndex0], inputs[kIndex2], kernel_size_, strides_, pads_, dilation_,
                        ceil_mode_, outputs[kIndex0]);
}

bool MaxPoolGradWithIndicesAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                          const std::vector<KernelTensor *> &workspace,
                                          const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOp(stream_ptr, workspace, inputs[kIndex1], inputs[kIndex0], inputs[kIndex2], kernel_size_, strides_, pads_,
        dilation_, ceil_mode_, outputs[kIndex0]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(MaxPoolGradWithIndices, MaxPoolGradWithIndicesAscend);
}  // namespace kernel
}  // namespace mindspore

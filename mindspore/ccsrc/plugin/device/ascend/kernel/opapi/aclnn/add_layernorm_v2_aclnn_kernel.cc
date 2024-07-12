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
#include "plugin/device/ascend/kernel/opapi/aclnn/add_layernorm_v2_aclnn_kernel.h"
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

void AddLayerNormAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                          const std::vector<KernelTensor *> &outputs) {
  auto additional_out = transform::ConvertKernelTensor<bool>(inputs[kIndex5]);
  auto eps_dtype_id = inputs[kIndex4]->dtype_id();
  eps_ = (eps_dtype_id == kNumberTypeFloat32) ? static_cast<double>(inputs[kIndex4]->GetValueWithCheck<float>())
                                              : inputs[kIndex4]->GetValueWithCheck<double>();

  GetWorkspaceForResize(inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], inputs[kIndex3], nullptr, eps_,
                        additional_out, outputs[kIndex0], outputs[kIndex1], outputs[kIndex2], outputs[kIndex3]);
}

bool AddLayerNormAscend::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                                const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  auto additional_out = transform::ConvertKernelTensor<bool>(inputs[kIndex5]);

  RunOp(stream_ptr, workspace, inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], inputs[kIndex3], nullptr, eps_,
        additional_out, outputs[kIndex0], outputs[kIndex1], outputs[kIndex2], outputs[kIndex3]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(AddLayerNormV2, AddLayerNormAscend);
}  // namespace kernel
}  // namespace mindspore

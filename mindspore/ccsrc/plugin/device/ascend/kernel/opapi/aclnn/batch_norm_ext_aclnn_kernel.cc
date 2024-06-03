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
#include "plugin/device/ascend/kernel/opapi/aclnn/batch_norm_ext_aclnn_kernel.h"
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

void BatchNormExtAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                          const std::vector<KernelTensor *> &outputs) {
  auto training = transform::ConvertKernelTensor<bool>(inputs[kIndex5]);
  auto eps_dtype_id = inputs[kIndex7]->dtype_id();
  eps_ = (eps_dtype_id == kNumberTypeFloat32) ? static_cast<double>(inputs[kIndex7]->GetValueWithCheck<float>())
                                              : inputs[kIndex7]->GetValueWithCheck<double>();

  auto momentum_dtype_id = inputs[kIndex6]->dtype_id();
  momentum_ = (momentum_dtype_id == kNumberTypeFloat32)
                ? static_cast<double>(inputs[kIndex6]->GetValueWithCheck<float>())
                : inputs[kIndex7]->GetValueWithCheck<double>();

  GetWorkspaceForResize(inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], inputs[kIndex3], inputs[kIndex4], training,
                        momentum_, eps_, outputs[kIndex0], outputs[kIndex1], outputs[kIndex2]);
}

bool BatchNormExtAscend::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                                const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  auto training = transform::ConvertKernelTensor<bool>(inputs[kIndex5]);
  RunOp(stream_ptr, workspace, inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], inputs[kIndex3], inputs[kIndex4],
        training, momentum_, eps_, outputs[kIndex0], outputs[kIndex1], outputs[kIndex2]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(BatchNormExt, BatchNormExtAscend);
}  // namespace kernel
}  // namespace mindspore

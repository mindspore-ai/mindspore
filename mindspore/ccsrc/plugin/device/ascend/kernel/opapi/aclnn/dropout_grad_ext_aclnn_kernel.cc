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
#include "plugin/device/ascend/kernel/opapi/aclnn/dropout_grad_ext_aclnn_kernel.h"
#include "ir/tensor.h"
#include "runtime/device/kernel_runtime.h"

namespace mindspore {
namespace kernel {
void DropoutGradExtAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                            const std::vector<KernelTensor *> &outputs) {
  p_value_ = static_cast<double>(inputs[kIndex2]->GetValueWithCheck<float>());

  MS_LOG(DEBUG) << "(" + TypeIdToString(inputs[kIndex2]->dtype_id()) + ")p = " << p_value_;

  GetWorkspaceForResize(inputs[kIndex0], inputs[kIndex1], p_value_, outputs[kIndex0]);
}

bool DropoutGradExtAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                  const std::vector<KernelTensor *> &workspace,
                                  const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  ParseGenExecutor(GEN_EXECUTOR_BOOST(dropout_do_mask_, do_mask_hash_id_, inputs[kIndex0], inputs[kIndex1], p_value_,
                                      outputs[kIndex0]));
  RunOp(stream_ptr, workspace);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(DropoutGradExt, DropoutGradExtAscend);
}  // namespace kernel
}  // namespace mindspore

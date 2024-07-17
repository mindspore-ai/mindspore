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
#include "plugin/device/ascend/kernel/opapi/aclnn/cummin_ext_aclnn_kernel.h"
#include "ir/tensor.h"
#include "runtime/device/kernel_runtime.h"

namespace mindspore {
namespace kernel {

void CumminExtAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  auto dim = transform::ConvertKernelTensor<int64_t>(inputs[kIndex1]);
  GetWorkspaceForResize(inputs[kIndex0], dim, outputs[kIndex0], outputs[kIndex1]);
}

bool CumminExtAscend::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                             const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  auto dim = transform::ConvertKernelTensor<int64_t>(inputs[kIndex1]);
  auto input_shape = inputs[kIndex0]->GetShape()->GetShapeVector();
  dim = dim < 0 ? dim + SizeToLong(input_shape.size()) : dim;
  ParseGenExecutor(GEN_EXECUTOR_BOOST(op_type_, hash_id_, inputs[kIndex0], dim, outputs[kIndex0], outputs[kIndex1]));
  RunOp(stream_ptr, workspace);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(CumminExt, CumminExtAscend);
}  // namespace kernel
}  // namespace mindspore

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
#include "plugin/device/ascend/kernel/opapi/aclnn/quant_batch_matmul_aclnn_kernel.h"
#include <vector>
#include <memory>
#include <functional>
#include "ir/tensor.h"
#include "runtime/device/kernel_runtime.h"
#include "transform/acl_ir/op_api_convert.h"

namespace mindspore {
namespace kernel {
void QuantMatmulV3Ascend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                           const std::vector<KernelTensor *> &outputs) {
  bool transpose_x1 = inputs[kIndex5]->GetValueWithCheck<bool>();
  bool transpose_x2 = inputs[kIndex6]->GetValueWithCheck<bool>();
  GetWorkspaceForResize(inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], inputs[kIndex3], inputs[kIndex4],
                        transpose_x1, transpose_x2, outputs[kIndex0]);
}

bool QuantMatmulV3Ascend::Launch(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &workspace,
                                 const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);

  bool transpose_x1 = inputs[kIndex5]->GetValueWithCheck<bool>();
  bool transpose_x2 = inputs[kIndex6]->GetValueWithCheck<bool>();
  ParseGenExecutor(GEN_EXECUTOR_BOOST(op_type_, hash_id_, inputs[kIndex0], inputs[kIndex1], inputs[kIndex2],
                                      inputs[kIndex3], inputs[kIndex4], transpose_x1, transpose_x2, outputs[kIndex0]));
  RunOp(stream_ptr, workspace);
  return true;
}
MS_ACLNN_KERNEL_FACTORY_REG(QuantBatchMatmul, QuantMatmulV3Ascend);
}  // namespace kernel
}  // namespace mindspore

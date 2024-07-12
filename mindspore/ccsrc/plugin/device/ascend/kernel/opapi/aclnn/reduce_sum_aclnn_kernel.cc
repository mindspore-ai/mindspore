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
#include "plugin/device/ascend/kernel/opapi/aclnn/reduce_sum_aclnn_kernel.h"
#include <vector>
#include "ir/tensor.h"
#include "transform/acl_ir/acl_helper.h"
#include "abstract/ops/primitive_infer_map.h"
#include "transform/symbol/acl_rt_symbol.h"
#include "transform/symbol/symbol_utils.h"

namespace mindspore {
namespace kernel {
void ReduceSumAclnnKernelMod::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                               const std::vector<KernelTensor *> &outputs) {
  dims_ = transform::ConvertKernelTensor<std::vector<int64_t>>(inputs[kIndex1]);
  keep_dim_ = transform::ConvertKernelTensor<bool>(inputs[kIndex2]);
  dtype_ = transform::ConvertKernelTensor<TypeId>(inputs[kIndex0]);
  auto skip_mode = transform::ConvertKernelTensor<bool>(inputs[kIndex3]);
  need_skip_execute_ = false;
  if (AnfAlgo::IsDynamicShapeSkipExecute(skip_mode, inputs[kIndex1]->GetShapeVector())) {
    need_skip_execute_ = true;
    return;
  }
  GetWorkspaceForResize(inputs[kIndex0], dims_, keep_dim_, dtype_, outputs[kIndex0]);
}

bool ReduceSumAclnnKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &workspace,
                                     const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOp(stream_ptr, workspace, inputs[kIndex0], dims_, keep_dim_, dtype_, outputs[kIndex0]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(ReduceSum, ReduceSumAclnnKernelMod);
}  // namespace kernel
}  // namespace mindspore

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
#include "plugin/device/ascend/kernel/opapi/aclnn/adamw_aclnn_kernel.h"
#include <algorithm>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include "ir/tensor.h"
#include "transform/acl_ir/acl_helper.h"
#include "transform/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {
void AdamWAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &outputs) {
  lr_ = transform::ConvertKernelTensor<float>(inputs[kIndex6]);
  beta1_ = transform::ConvertKernelTensor<float>(inputs[kIndex7]);
  beta2_ = transform::ConvertKernelTensor<float>(inputs[kIndex8]);
  decay_ = transform::ConvertKernelTensor<float>(inputs[kIndex9]);
  eps_ = transform::ConvertKernelTensor<float>(inputs[kIndex10]);
  amsgrad_ = transform::ConvertKernelTensor<bool>(inputs[kIndex11]);
  maximize_ = transform::ConvertKernelTensor<bool>(inputs[kIndex12]);
  // Infer function has confirmed the actual dtype of output
  GetWorkspaceForResize(inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], inputs[kIndex3], inputs[kIndex4],
                        inputs[kIndex5], lr_, beta1_, beta2_, decay_, eps_, amsgrad_, maximize_);
}

bool AdamWAscend::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                         const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  ParseGenExecutor(GEN_EXECUTOR_BOOST(op_type_, hash_id_, inputs[kIndex0], inputs[kIndex1], inputs[kIndex2],
                                      inputs[kIndex3], inputs[kIndex4], inputs[kIndex5], lr_, beta1_, beta2_, decay_,
                                      eps_, amsgrad_, maximize_));
  RunOp(stream_ptr, workspace);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(AdamW, AdamWAscend);
}  // namespace kernel
}  // namespace mindspore

/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/kernel/opapi/aclnn/flash_attention_score_grad_aclnn_kernel.h"
#include <algorithm>
#include "ir/tensor.h"
#include "runtime/device/kernel_runtime.h"

namespace mindspore {
namespace kernel {
void FAScoreGradAclnnKernelMod::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                 const std::vector<KernelTensor *> &outputs) {
  auto scale_value = static_cast<double>(GetFAGradAttr<float>("scale_value"));
  auto keep_prob = static_cast<double>(GetFAGradAttr<float>("keep_prob"));
  auto pre_tokens = GetFAGradAttr<int64_t>("pre_tokens");
  auto next_tokens = GetFAGradAttr<int64_t>("next_tokens");
  auto head_num = GetFAGradAttr<int64_t>("head_num");
  auto input_layout = GetFAGradAttr<std::string>("input_layout");
  auto inner_precise = GetFAGradAttr<int64_t>("inner_precise");
  auto sparse_mode = GetFAGradAttr<int64_t>("sparse_mode");
  GetWorkspaceForResize(inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], inputs[kIndex3], inputs[kIndex4],
                        inputs[kIndex5], inputs[kIndex6], inputs[kIndex7], inputs[kIndex8], inputs[kIndex9],
                        inputs[kIndex10], inputs[kIndex11], nullptr, scale_value, keep_prob, pre_tokens, next_tokens,
                        head_num, input_layout, inner_precise, sparse_mode, outputs[kIndex0], outputs[kIndex1],
                        outputs[kIndex2], outputs[kIndex3]);
}

bool FAScoreGradAclnnKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &workspace,
                                       const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  ParseGenExecutor(FAGradGenerate(inputs, outputs));
  RunOp(stream_ptr, workspace);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(FlashAttentionScoreGrad, FAScoreGradAclnnKernelMod);
}  // namespace kernel
}  // namespace mindspore

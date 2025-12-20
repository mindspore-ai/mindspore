/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/lightning_indexer_grad_aclnn_kernel.h"
#include "ir/tensor.h"

namespace mindspore {
namespace kernel {
namespace lightning_indexer_grad {

namespace {
constexpr int64_t kHeadNum = 0;
constexpr bool kDeterministic = true;
}  // namespace

void LightningIndexerGradAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                  const std::vector<KernelTensor *> &outputs) {
  auto layout_query = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex7]);
  layout_query_str_ = device::ascend::FASInputLayoutMode::ConvertEnumToString(layout_query);

  sparse_mode_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex8]);
  pre_tokens_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex9]);
  next_tokens_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex10]);

  GetWorkspaceForResize(inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], inputs[kIndex3], inputs[kIndex4],
                        inputs[kIndex5], inputs[kIndex6], kHeadNum, layout_query_str_, sparse_mode_, pre_tokens_,
                        next_tokens_, kDeterministic, outputs[kIndex0], outputs[kIndex1], outputs[kIndex2]);
}

bool LightningIndexerGradAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &workspace,
                                        const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOp(stream_ptr, workspace, inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], inputs[kIndex3], inputs[kIndex4],
        inputs[kIndex5], inputs[kIndex6], kHeadNum, layout_query_str_, sparse_mode_, pre_tokens_, next_tokens_,
        kDeterministic, outputs[kIndex0], outputs[kIndex1], outputs[kIndex2]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(LightningIndexerGrad, LightningIndexerGradAscend);
}  // namespace lightning_indexer_grad
}  // namespace kernel
}  // namespace mindspore

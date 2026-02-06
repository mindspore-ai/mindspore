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

#include "kernel/ascend/aclnn/kernel_mod_impl/customize/nsa_select_attention_grad_aclnn_kernel.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"

namespace mindspore {
namespace kernel {
namespace nsa_select_attention_grad {

void NsaSelectAttentionGradAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                    const std::vector<KernelTensor *> &outputs) {
  scale_value_ = static_cast<double>(device::ascend::ConvertKernelTensor<float>(inputs[kIndex8]));
  head_num_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex9]);
  select_block_size_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex10]);
  select_block_count_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex11]);

  auto actual_seq_qlen_vector = device::ascend::ConvertKernelTensor<std::vector<int64_t>>(inputs[kIndex13]);
  auto actual_seq_kvlen_vector = device::ascend::ConvertKernelTensor<std::vector<int64_t>>(inputs[kIndex14]);
  actual_seq_qlen_vector_pair_ = std::make_pair(actual_seq_qlen_vector, true);
  actual_seq_kvlen_vector_pair_ = std::make_pair(actual_seq_kvlen_vector, true);

  GetWorkspaceForResize(inputs[kIndex1], inputs[kIndex2], inputs[kIndex3], inputs[kIndex4], inputs[kIndex0],
                        inputs[kIndex5], inputs[kIndex6], inputs[kIndex7], actual_seq_qlen_vector_pair_,
                        actual_seq_kvlen_vector_pair_, inputs[kIndex12], scale_value_, select_block_size_,
                        select_block_count_, head_num_, layout_str_, sparse_mode_, outputs[kIndex0], outputs[kIndex1],
                        outputs[kIndex2]);
}

bool NsaSelectAttentionGradAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                          const std::vector<KernelTensor *> &workspace,
                                          const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOp(stream_ptr, workspace, inputs[kIndex1], inputs[kIndex2], inputs[kIndex3], inputs[kIndex4], inputs[kIndex0],
        inputs[kIndex5], inputs[kIndex6], inputs[kIndex7], actual_seq_qlen_vector_pair_, actual_seq_kvlen_vector_pair_,
        inputs[kIndex12], scale_value_, select_block_size_, select_block_count_, head_num_, layout_str_, sparse_mode_,
        outputs[kIndex0], outputs[kIndex1], outputs[kIndex2]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(NsaSelectAttentionGrad, NsaSelectAttentionGradAscend);
}  // namespace nsa_select_attention_grad
}  // namespace kernel
}  // namespace mindspore

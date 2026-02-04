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

#include "kernel/ascend/aclnn/kernel_mod_impl/customize/nsa_select_attention_aclnn_kernel.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"

namespace mindspore {
namespace kernel {
namespace nsa_select_attention {

void NsaSelectAttentionAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                const std::vector<KernelTensor *> &outputs) {
  scale_value_ = static_cast<double>(device::ascend::ConvertKernelTensor<float>(inputs[kIndex4]));
  head_num_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex5]);
  select_block_size_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex6]);
  select_block_count_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex7]);

  auto actual_seq_qlen_vector = device::ascend::ConvertKernelTensor<std::vector<int64_t>>(inputs[kIndex9]);
  auto actual_seq_kvlen_vector = device::ascend::ConvertKernelTensor<std::vector<int64_t>>(inputs[kIndex10]);
  actual_seq_qlen_vector_pair_ = std::make_pair(actual_seq_qlen_vector, true);
  actual_seq_kvlen_vector_pair_ = std::make_pair(actual_seq_kvlen_vector, true);

  GetWorkspaceForResize(inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], inputs[kIndex3], inputs[kIndex8],
                        actual_seq_qlen_vector_pair_, actual_seq_kvlen_vector_pair_, scale_value_, head_num_,
                        layout_str_, sparse_mode_, select_block_size_, select_block_count_, outputs[kIndex1],
                        outputs[kIndex2], outputs[kIndex0]);
}

bool NsaSelectAttentionAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &workspace,
                                      const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOp(stream_ptr, workspace, inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], inputs[kIndex3], inputs[kIndex8],
        actual_seq_qlen_vector_pair_, actual_seq_kvlen_vector_pair_, scale_value_, head_num_, layout_str_, sparse_mode_,
        select_block_size_, select_block_count_, outputs[kIndex1], outputs[kIndex2], outputs[kIndex0]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(NsaSelectAttention, NsaSelectAttentionAscend);
}  // namespace nsa_select_attention
}  // namespace kernel
}  // namespace mindspore

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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/prompt_flash_attention_aclnn_kernel.h"
#include <algorithm>
#include "ir/tensor.h"

namespace mindspore {
namespace kernel {
namespace prompt_flash_attention {

void PromptFlashAttentionAscend::SetScalarParam(const std::vector<KernelTensor *> &inputs) {
  num_heads_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex12]);
  scale_value_d_ = static_cast<double>(device::ascend::ConvertKernelTensor<float>(inputs[kIndex13]));
  pre_tokens_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex14]);
  next_tokens_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex15]);
  auto input_layout = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex16]);
  input_layout_str_ = device::ascend::FASInputLayoutMode::ConvertEnumToString(input_layout);
  num_key_value_heads_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex17]);
  sparse_mode_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex18]);
  inner_precise_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex19]);

  return;
}

void PromptFlashAttentionAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                  const std::vector<KernelTensor *> &outputs) {
  SetScalarParam(inputs);

  auto &llm_manager = LLMManager::GetInstance();
  std::vector<int64_t> actual_q_lengths_vector = {};
  std::vector<int64_t> actual_kv_lengths_vector = {};
  auto ret = llm_manager.GetGraphInputToVector<int32_t, int64_t>("q_seq_lens", &actual_q_lengths_vector);
  if (!ret) {
    auto actual_seq_qlen = inputs[kIndex4];
    MS_EXCEPTION_IF_NULL(actual_seq_qlen);
    if (actual_seq_qlen->type_id() != kMetaTypeNone) {
      actual_q_lengths_vector = actual_seq_qlen->GetValueWithCheck<std::vector<int64_t>>();
    }
  }
  actual_q_lengths_vector_pair_ = std::make_pair(actual_q_lengths_vector, true);

  ret = llm_manager.GetGraphInputToVector<int32_t, int64_t>("batch_valid_length", &actual_kv_lengths_vector);
  if (!ret) {
    auto actual_seq_kvlen = inputs[kIndex5];
    MS_EXCEPTION_IF_NULL(actual_seq_kvlen);
    if (actual_seq_kvlen->type_id() != kMetaTypeNone) {
      actual_kv_lengths_vector = actual_seq_kvlen->GetValueWithCheck<std::vector<int64_t>>();
    }
  }
  actual_kv_lengths_vector_pair_ = std::make_pair(actual_kv_lengths_vector, true);

  op_type_ = "aclnnPromptFlashAttentionV3";
  GetWorkspaceForResize(inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], inputs[kIndex6], inputs[kIndex3],
                        actual_q_lengths_vector_pair_, actual_kv_lengths_vector_pair_, inputs[kIndex7], inputs[kIndex8],
                        inputs[kIndex9], inputs[kIndex10], inputs[kIndex11], num_heads_, scale_value_d_, pre_tokens_,
                        next_tokens_, input_layout_str_, num_key_value_heads_, sparse_mode_, inner_precise_,
                        outputs[kIndex0]);
}

bool PromptFlashAttentionAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &workspace,
                                        const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOp(stream_ptr, workspace, inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], inputs[kIndex6], inputs[kIndex3],
        actual_q_lengths_vector_pair_, actual_kv_lengths_vector_pair_, inputs[kIndex7], inputs[kIndex8],
        inputs[kIndex9], inputs[kIndex10], inputs[kIndex11], num_heads_, scale_value_d_, pre_tokens_, next_tokens_,
        input_layout_str_, num_key_value_heads_, sparse_mode_, inner_precise_, outputs[kIndex0]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(PromptFlashAttention, PromptFlashAttentionAscend);
}  // namespace prompt_flash_attention
}  // namespace kernel
}  // namespace mindspore

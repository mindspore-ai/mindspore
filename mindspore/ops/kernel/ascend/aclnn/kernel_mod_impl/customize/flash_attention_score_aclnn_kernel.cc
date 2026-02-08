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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/flash_attention_score_aclnn_kernel.h"
#include <algorithm>
#include <utility>
#include <vector>
#include "ir/tensor.h"

namespace mindspore {
namespace kernel {
namespace flash_attention_score {
void FlashAttentionScoreAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                 const std::vector<KernelTensor *> &outputs) {
  auto prefix = inputs[kIndex7];
  MS_EXCEPTION_IF_NULL(prefix);
  std::vector<int64_t> prefix_array;
  if (prefix->type_id() != kMetaTypeNone) {
    prefix_array = prefix->GetValueWithCheck<std::vector<int64_t>>();
  }
  auto head_num = inputs[kIndex10];
  MS_EXCEPTION_IF_NULL(head_num);
  auto head_num_value = head_num->GetValueWithCheck<int64_t>();
  auto keep_prob = inputs[kIndex11];
  MS_EXCEPTION_IF_NULL(keep_prob);
  auto keep_prob_value = static_cast<double>(keep_prob->GetValueWithCheck<float>());
  auto scale_value = inputs[kIndex12];
  MS_EXCEPTION_IF_NULL(scale_value);
  auto scale_value_value = static_cast<double>(scale_value->GetValueWithCheck<float>());
  auto pre_tokens = inputs[kIndex13];
  MS_EXCEPTION_IF_NULL(pre_tokens);
  auto pre_tokens_value = pre_tokens->GetValueWithCheck<int64_t>();
  auto next_tokens = inputs[kIndex14];
  MS_EXCEPTION_IF_NULL(next_tokens);
  auto next_tokens_value = next_tokens->GetValueWithCheck<int64_t>();
  auto inner_precise = inputs[kIndex15];
  MS_EXCEPTION_IF_NULL(inner_precise);
  auto inner_precise_value = inner_precise->GetValueWithCheck<int64_t>();
  auto input_layout = inputs[kIndex16];
  MS_EXCEPTION_IF_NULL(input_layout);
  auto input_layout_value = input_layout->GetValueWithCheck<int64_t>();
  auto input_layout_string = FASInputLayoutMode::ConvertEnumToString(input_layout_value);
  auto sparse_mode = inputs[kIndex17];
  MS_EXCEPTION_IF_NULL(sparse_mode);
  auto sparse_mode_value = sparse_mode->GetValueWithCheck<int64_t>();

  if (input_layout_string == "TND") {
    auto actual_seq_qlen = inputs[kIndex8];
    MS_EXCEPTION_IF_NULL(actual_seq_qlen);
    auto actual_seq_kvlen = inputs[kIndex9];
    MS_EXCEPTION_IF_NULL(actual_seq_kvlen);

    if (actual_seq_kvlen->type_id() == kMetaTypeNone || actual_seq_qlen->type_id() == kMetaTypeNone) {
      MS_LOG(EXCEPTION) << "For [aclnnFlashAttentionVarLenScore], actual_seq_qlen and actual_seq_kvlen must be not "
                           "none when input layout is TND.";
    }

    auto actual_seq_kvlen_array = actual_seq_kvlen->GetValueWithCheck<std::vector<int64_t>>();
    auto actual_seq_qlen_array = actual_seq_qlen->GetValueWithCheck<std::vector<int64_t>>();
    if (!CheckSeqList(actual_seq_kvlen_array, inputs[kIndex1]->GetShapeVector()) ||
        !CheckSeqList(actual_seq_qlen_array, inputs[kIndex0]->GetShapeVector())) {
      MS_LOG(EXCEPTION)
        << "For actual_seq_qlen and actual_seq_kvlen, must be increasing array and the last number is equal to T.";
    }

    op_type_ = "aclnnFlashAttentionVarLenScore";
    GetWorkspaceForResize(inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], inputs[kIndex3], inputs[kIndex4],
                          inputs[kIndex5], inputs[kIndex6], std::move(prefix_array), std::move(actual_seq_qlen_array),
                          std::move(actual_seq_kvlen_array), scale_value_value, keep_prob_value, pre_tokens_value,
                          next_tokens_value, head_num_value, std::move(input_layout_string), inner_precise_value,
                          sparse_mode_value, outputs[kIndex0], outputs[kIndex1], outputs[kIndex2], outputs[kIndex3]);
  } else {
    op_type_ = "aclnnFlashAttentionScore";
    GetWorkspaceForResize(inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], inputs[kIndex3], inputs[kIndex4],
                          inputs[kIndex5], inputs[kIndex6], std::move(prefix_array), scale_value_value, keep_prob_value,
                          pre_tokens_value, next_tokens_value, head_num_value, std::move(input_layout_string),
                          inner_precise_value, sparse_mode_value, outputs[kIndex0], outputs[kIndex1], outputs[kIndex2],
                          outputs[kIndex3]);
  }
}

bool FlashAttentionScoreAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &workspace,
                                       const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  FAGenerate(inputs, workspace, outputs, stream_ptr);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(FlashAttentionScore, FlashAttentionScoreAscend);
}  // namespace flash_attention_score
}  // namespace kernel
}  // namespace mindspore

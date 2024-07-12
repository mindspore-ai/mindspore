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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_FLASH_ATTENTION_SCORE_GRAD_ACLNN_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_FLASH_ATTENTION_SCORE_GRAD_ACLNN_KERNEL_MOD_H_
#include <vector>
#include <string>
#include <memory>
#include "ops/base_operator.h"
#include "plugin/device/ascend/kernel/opapi/aclnn_kernel_mod.h"
#include "transform/acl_ir/acl_convert.h"
#include "transform/graph_ir/op_adapter_base.h"

namespace mindspore {
using mindspore::transform::FASInputLayoutMode;
namespace kernel {
using TensorParams = transform::TensorParams;

class FlashAttentionScoreGradAscend : public AclnnKernelMod {
 public:
  FlashAttentionScoreGradAscend() : AclnnKernelMod("aclnnFlashAttentionScoreGrad") {}
  ~FlashAttentionScoreGradAscend() = default;

  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;
  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
    MS_EXCEPTION_IF_NULL(outputs[kIndex0]);
    if (outputs[kIndex0]->type_id() != kObjectTypeTensorType) {
      MS_LOG(EXCEPTION) << "now only support tensor type for EmptyKernelTensor in " << op_type_;
    }
    if (inputs[kIndex7]->dtype_id() == TypeId::kNumberTypeFloat16) {
      MS_LOG(EXCEPTION) << "Attn mask don't support float16.";
    }
    return true;
  }

 protected:
  DEFINE_GET_WORKSPACE_FOR_RESIZE()

  void FAGradGenerate(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                      const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
    auto prefix = inputs[kIndex12];
    MS_EXCEPTION_IF_NULL(prefix);
    std::vector<int64_t> prefix_array;
    if (prefix->type_id() != kMetaTypeNone) {
      prefix_array = prefix->GetValueWithCheck<std::vector<int64_t>>();
    }
    auto actual_seq_qlen = inputs[kIndex13];
    MS_EXCEPTION_IF_NULL(actual_seq_qlen);
    std::vector<int64_t> actual_seq_qlen_array;
    if (actual_seq_qlen->type_id() != kMetaTypeNone) {
      actual_seq_qlen_array = actual_seq_qlen->GetValueWithCheck<std::vector<int64_t>>();
    }
    auto actual_seq_kvlen = inputs[kIndex14];
    MS_EXCEPTION_IF_NULL(actual_seq_kvlen);
    std::vector<int64_t> actual_seq_kvlen_array;
    if (actual_seq_kvlen->type_id() != kMetaTypeNone) {
      actual_seq_kvlen_array = actual_seq_kvlen->GetValueWithCheck<std::vector<int64_t>>();
    }
    auto head_num = inputs[kIndex15];
    MS_EXCEPTION_IF_NULL(head_num);
    auto head_num_value = head_num->GetValueWithCheck<int64_t>();
    auto keep_prob = inputs[kIndex16];
    MS_EXCEPTION_IF_NULL(keep_prob);
    auto keep_prob_value = static_cast<double>(keep_prob->GetValueWithCheck<float>());
    auto scale_value = inputs[kIndex17];
    MS_EXCEPTION_IF_NULL(scale_value);
    auto scale_value_value = static_cast<double>(scale_value->GetValueWithCheck<float>());
    auto pre_tokens = inputs[kIndex18];
    MS_EXCEPTION_IF_NULL(pre_tokens);
    auto pre_tokens_value = pre_tokens->GetValueWithCheck<int64_t>();
    auto next_tokens = inputs[kIndex19];
    MS_EXCEPTION_IF_NULL(next_tokens);
    auto next_tokens_value = next_tokens->GetValueWithCheck<int64_t>();
    auto inner_precise = inputs[kIndex20];
    MS_EXCEPTION_IF_NULL(inner_precise);
    auto inner_precise_value = inner_precise->GetValueWithCheck<int64_t>();
    auto input_layout = inputs[kIndex21];
    MS_EXCEPTION_IF_NULL(input_layout);
    auto input_layout_value = input_layout->GetValueWithCheck<int64_t>();
    auto input_layout_string = FASInputLayoutMode::ConvertEnumToString(input_layout_value);
    auto sparse_mode = inputs[kIndex22];
    MS_EXCEPTION_IF_NULL(sparse_mode);
    auto sparse_mode_value = sparse_mode->GetValueWithCheck<int64_t>();

    if (input_layout_string == "TND") {
      if (actual_seq_kvlen->type_id() == kMetaTypeNone || actual_seq_qlen->type_id() == kMetaTypeNone) {
        MS_LOG(EXCEPTION) << "For [aclnnFlashAttentionUnpaddingScoreGrad], actual_seq_qlen and actual_seq_kvlen must "
                             "be not none when input layout is TND.";
      }
      op_type_ = "aclnnFlashAttentionUnpaddingScoreGrad";
      RunOp(stream_ptr, workspace, inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], inputs[kIndex3], inputs[kIndex4],
            inputs[kIndex5], inputs[kIndex6], inputs[kIndex7], inputs[kIndex8], inputs[kIndex9], inputs[kIndex10],
            inputs[kIndex11], prefix_array, actual_seq_qlen_array, actual_seq_kvlen_array, scale_value_value,
            keep_prob_value, pre_tokens_value, next_tokens_value, head_num_value, input_layout_string,
            inner_precise_value, sparse_mode_value, outputs[kIndex0], outputs[kIndex1], outputs[kIndex2],
            outputs[kIndex3]);
      return;
    }
    op_type_ = "aclnnFlashAttentionScoreGrad";
    RunOp(stream_ptr, workspace, inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], inputs[kIndex3], inputs[kIndex4],
          inputs[kIndex5], inputs[kIndex6], inputs[kIndex7], inputs[kIndex8], inputs[kIndex9], inputs[kIndex10],
          inputs[kIndex11], prefix_array, scale_value_value, keep_prob_value, pre_tokens_value, next_tokens_value,
          head_num_value, input_layout_string, inner_precise_value, sparse_mode_value, outputs[kIndex0],
          outputs[kIndex1], outputs[kIndex2], outputs[kIndex3]);
  }
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_FLASH_ATTENTION_SCORE_GRAD_ACLNN_KERNEL_MOD_H_

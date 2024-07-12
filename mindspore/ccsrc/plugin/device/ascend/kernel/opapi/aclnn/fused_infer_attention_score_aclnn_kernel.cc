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
#include "plugin/device/ascend/kernel/opapi/aclnn/fused_infer_attention_score_aclnn_kernel.h"
#include <algorithm>
#include <vector>
#include <memory>
#include <functional>
#include "ir/tensor.h"
#include "runtime/device/kernel_runtime.h"
#include "transform/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {
constexpr size_t kFIASMinNum = 28;
void FusedInferAttentionScoreAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                      const std::vector<KernelTensor *> &outputs) {
  std::vector<int64_t> dyn_input_sizes;
  std::vector<size_t> real_input_idx_;
  std::vector<KernelTensor *> key_tensors;
  std::vector<KernelTensor *> value_tensors;
  real_input_idx_.clear();
  real_input_idx_.emplace_back(0);
  dyn_input_sizes = GetValue<const std::vector<int64_t>>(primitive_->GetAttr(kAttrDynInputSizes));
  // dynamic inputs
  key_tensors.assign(inputs.begin() + kIndex1, inputs.begin() + kIndex1 + dyn_input_sizes[kIndex1]);
  value_tensors.assign(inputs.begin() + kIndex1 + dyn_input_sizes[kIndex1],
                       inputs.begin() + kIndex1 + dyn_input_sizes[kIndex1] + dyn_input_sizes[kIndex2]);
  for (size_t i = 1; i < dyn_input_sizes.size(); ++i) {
    auto pace = dyn_input_sizes[i] > 0 ? dyn_input_sizes[i] : 1;
    real_input_idx_.emplace_back(real_input_idx_[i - 1] + pace);
  }
  std::vector<int64_t> actual_seq_lengths_array;
  std::vector<int64_t> actual_seq_lengths_kv_array;
  auto actual_seq_lengths = inputs[real_input_idx_[kIndex5]];
  MS_EXCEPTION_IF_NULL(actual_seq_lengths);
  if (actual_seq_lengths->type_id() != kMetaTypeNone) {
    actual_seq_lengths_array = actual_seq_lengths->GetValueWithCheck<std::vector<int64_t>>();
  }
  auto actual_seq_lengths_kv = inputs[real_input_idx_[kIndex6]];
  MS_EXCEPTION_IF_NULL(actual_seq_lengths_kv);
  if (actual_seq_lengths_kv->type_id() != kMetaTypeNone) {
    actual_seq_lengths_kv_array = actual_seq_lengths_kv->GetValueWithCheck<std::vector<int64_t>>();
  }
  auto num_heads = transform::ConvertKernelTensor<int64_t>(inputs[real_input_idx_[kIndex17]]);
  auto scale = static_cast<double>(transform::ConvertKernelTensor<float>(inputs[real_input_idx_[kIndex18]]));
  auto pre_tokens = transform::ConvertKernelTensor<int64_t>(inputs[real_input_idx_[kIndex19]]);
  auto next_tokens = transform::ConvertKernelTensor<int64_t>(inputs[real_input_idx_[kIndex20]]);
  auto input_layout_value = transform::ConvertKernelTensor<int64_t>(inputs[real_input_idx_[kIndex21]]);
  auto input_layout_string = FASInputLayoutMode::ConvertEnumToString(input_layout_value);
  auto num_key_value_heads = transform::ConvertKernelTensor<int64_t>(inputs[real_input_idx_[kIndex22]]);
  auto sparse_mode = transform::ConvertKernelTensor<int64_t>(inputs[real_input_idx_[kIndex23]]);
  auto inner_precise = transform::ConvertKernelTensor<int64_t>(inputs[real_input_idx_[kIndex24]]);
  auto block_size = transform::ConvertKernelTensor<int64_t>(inputs[real_input_idx_[kIndex25]]);
  auto antiquant_mode = transform::ConvertKernelTensor<int64_t>(inputs[real_input_idx_[kIndex26]]);
  auto softmax_lse_flag = transform::ConvertKernelTensor<bool>(inputs[real_input_idx_[kIndex27]]);

  GetWorkspaceForResize(
    inputs[kIndex0], key_tensors, value_tensors, inputs[real_input_idx_[kIndex3]], inputs[real_input_idx_[kIndex4]],
    actual_seq_lengths_array, actual_seq_lengths_kv_array, inputs[real_input_idx_[kIndex7]],
    inputs[real_input_idx_[kIndex8]], inputs[real_input_idx_[kIndex9]], inputs[real_input_idx_[kIndex10]],
    inputs[real_input_idx_[kIndex11]], inputs[real_input_idx_[kIndex12]], inputs[real_input_idx_[kIndex13]],
    inputs[real_input_idx_[kIndex14]], inputs[real_input_idx_[kIndex15]], inputs[real_input_idx_[kIndex16]], num_heads,
    scale, pre_tokens, next_tokens, input_layout_string, num_key_value_heads, sparse_mode, inner_precise, block_size,
    antiquant_mode, softmax_lse_flag, outputs[kIndex0], outputs[kIndex1]);
}

bool FusedInferAttentionScoreAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                            const std::vector<KernelTensor *> &workspace,
                                            const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  std::vector<int64_t> dyn_input_sizes;
  std::vector<size_t> real_input_idx_;
  std::vector<KernelTensor *> key_tensors;
  std::vector<KernelTensor *> value_tensors;
  real_input_idx_.clear();
  real_input_idx_.emplace_back(0);
  dyn_input_sizes = GetValue<const std::vector<int64_t>>(primitive_->GetAttr(kAttrDynInputSizes));
  // dynamic inputs
  key_tensors.assign(inputs.begin() + kIndex1, inputs.begin() + kIndex1 + dyn_input_sizes[kIndex1]);
  value_tensors.assign(inputs.begin() + kIndex1 + dyn_input_sizes[kIndex1],
                       inputs.begin() + kIndex1 + dyn_input_sizes[kIndex1] + dyn_input_sizes[kIndex2]);
  for (size_t i = 1; i < dyn_input_sizes.size(); ++i) {
    auto pace = dyn_input_sizes[i] > 0 ? dyn_input_sizes[i] : 1;
    real_input_idx_.emplace_back(real_input_idx_[i - 1] + pace);
  }
  std::vector<int64_t> actual_seq_lengths_array;
  std::vector<int64_t> actual_seq_lengths_kv_array;
  auto actual_seq_lengths = inputs[real_input_idx_[kIndex5]];
  MS_EXCEPTION_IF_NULL(actual_seq_lengths);
  if (actual_seq_lengths->type_id() != kMetaTypeNone) {
    actual_seq_lengths_array = actual_seq_lengths->GetValueWithCheck<std::vector<int64_t>>();
  }
  auto actual_seq_lengths_kv = inputs[real_input_idx_[kIndex6]];
  MS_EXCEPTION_IF_NULL(actual_seq_lengths_kv);
  if (actual_seq_lengths_kv->type_id() != kMetaTypeNone) {
    actual_seq_lengths_kv_array = actual_seq_lengths_kv->GetValueWithCheck<std::vector<int64_t>>();
  }
  auto num_heads = transform::ConvertKernelTensor<int64_t>(inputs[real_input_idx_[kIndex17]]);
  auto scale = static_cast<double>(transform::ConvertKernelTensor<float>(inputs[real_input_idx_[kIndex18]]));
  auto pre_tokens = transform::ConvertKernelTensor<int64_t>(inputs[real_input_idx_[kIndex19]]);
  auto next_tokens = transform::ConvertKernelTensor<int64_t>(inputs[real_input_idx_[kIndex20]]);
  auto input_layout_value = transform::ConvertKernelTensor<int64_t>(inputs[real_input_idx_[kIndex21]]);
  auto input_layout_string = FASInputLayoutMode::ConvertEnumToString(input_layout_value);
  auto num_key_value_heads = transform::ConvertKernelTensor<int64_t>(inputs[real_input_idx_[kIndex22]]);
  auto sparse_mode = transform::ConvertKernelTensor<int64_t>(inputs[real_input_idx_[kIndex23]]);
  auto inner_precise = transform::ConvertKernelTensor<int64_t>(inputs[real_input_idx_[kIndex24]]);
  auto block_size = transform::ConvertKernelTensor<int64_t>(inputs[real_input_idx_[kIndex25]]);
  auto antiquant_mode = transform::ConvertKernelTensor<int64_t>(inputs[real_input_idx_[kIndex26]]);
  auto softmax_lse_flag = transform::ConvertKernelTensor<bool>(inputs[real_input_idx_[kIndex27]]);
  RunOp(stream_ptr, workspace, inputs[kIndex0], key_tensors, value_tensors, inputs[real_input_idx_[kIndex3]],
        inputs[real_input_idx_[kIndex4]], actual_seq_lengths_array, actual_seq_lengths_kv_array,
        inputs[real_input_idx_[kIndex7]], inputs[real_input_idx_[kIndex8]], inputs[real_input_idx_[kIndex9]],
        inputs[real_input_idx_[kIndex10]], inputs[real_input_idx_[kIndex11]], inputs[real_input_idx_[kIndex12]],
        inputs[real_input_idx_[kIndex13]], inputs[real_input_idx_[kIndex14]], inputs[real_input_idx_[kIndex15]],
        inputs[real_input_idx_[kIndex16]], num_heads, scale, pre_tokens, next_tokens, input_layout_string,
        num_key_value_heads, sparse_mode, inner_precise, block_size, antiquant_mode, softmax_lse_flag, outputs[kIndex0],
        outputs[kIndex1]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(FusedInferAttentionScore, FusedInferAttentionScoreAscend);
}  // namespace kernel
}  // namespace mindspore

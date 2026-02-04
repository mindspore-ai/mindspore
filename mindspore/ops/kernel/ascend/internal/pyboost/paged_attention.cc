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

#include "kernel/ascend/internal/pyboost/paged_attention.h"

#include <memory>
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"

namespace mindspore {
namespace kernel {
internal::InternalOpPtr PagedAttention::CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                                     const internal::OutputsImmutableInfoList &outputs) {
  created_flag_ = true;
  return internal::CreatePagedAttentionOp(inputs, outputs, param_, internal::kInternalPagedAttentionOpName);
}

bool PagedAttention::UpdateParam() {
  if (created_flag_) {
    created_flag_ = false;
    return true;
  }

  auto ret = internal_op_->UpdateParam(&param_);
  if (ret != internal::kInternalOk) {
    MS_LOG(ERROR) << "InternalPagedAttention UpdateParam failed, kernel_name: " << kernel_name_;
    return false;
  }
  return true;
}

uint64_t PagedAttention::GetOrGenerateOpTilingKey(const uint64_t &tiling_key) const {
  return CalcInternalOpTilingHash(kernel_name_, tiling_key, param_.q_seq_len, param_.kv_seq_len);
}

void PagedAttention::Call(const std::shared_ptr<pyboost::OpRunner> &op, const uint64_t &op_key,
                          const uint64_t &tiling_key, const TensorPtr &query, const TensorPtr &key_cache,
                          const std::optional<TensorPtr> &value_cache, const std::optional<TensorPtr> &block_tabels,
                          const std::optional<TensorPtr> &context_lens, const std::optional<TensorPtr> &antiquant_scale,
                          const std::optional<TensorPtr> &antiquant_offset, const std::optional<TensorPtr> &attn_mask,
                          const std::optional<TensorPtr> &q_seq_lens, const std::optional<TensorPtr> &alibi_mask,
                          const int64_t &head_num, const float &scale_value, const int64_t &kv_head_num,
                          const int64_t &kv_cache_quant_mode, const int64_t &mask_mode, const int64_t &mla_v_dim) {
  TensorPtrList inputs = {query,
                          key_cache,
                          value_cache.has_value() ? value_cache.value() : nullptr,
                          block_tabels.has_value() ? block_tabels.value() : nullptr,
                          context_lens.has_value() ? context_lens.value() : nullptr,
                          antiquant_scale.has_value() ? antiquant_scale.value() : nullptr,
                          antiquant_offset.has_value() ? antiquant_offset.value() : nullptr,
                          attn_mask.has_value() ? attn_mask.value() : nullptr,
                          alibi_mask.has_value() ? alibi_mask.value() : nullptr};
  TensorPtrList outputs = op->outputs();
  TransInternalShapes(inputs, outputs);

  param_.head_num = static_cast<int32_t>(head_num);
  param_.tor = scale_value;
  param_.kv_head_num = static_cast<int32_t>(kv_head_num);
  param_.kv_cache_quant_mode = static_cast<int32_t>(kv_cache_quant_mode);
  param_.mask_mode = static_cast<internal::PagedAttentionParam::MaskMode>(mask_mode);
  param_.mla_v_dim = static_cast<int32_t>(mla_v_dim);

  (void)GetSeqLenFromInputTensor(inputs[kIndex4], &param_.kv_seq_len);
  if (q_seq_lens.has_value()) {
    (void)GetSeqLenFromInputTensor(q_seq_lens.value(), &param_.q_seq_len);
  }
  param_.has_q_seq_lens = q_seq_lens.has_value();

  // Shallow copy of Tensor objects to avoid multi-threaded operations on Tensor object.
  auto new_context_lens = inputs[kIndex4] == nullptr ? nullptr : std::make_shared<tensor::Tensor>(*inputs[kIndex4]);
  inputs[kIndex4] = new_context_lens;
  runtime::DeviceAddressUtils::CreateInputTensorAddress(op->device_context(), op->stream_id(), 0, new_context_lens);
  pyboost::PyBoostUtils::MallocForInput(op->device_context(), inputs[kIndex4]);

  has_attn_mask_ = attn_mask.has_value();
  has_alibi_mask_ = alibi_mask.has_value();
  CheckMask();

  GetOrCreateKernel(op, op_key, tiling_key, inputs, outputs);
  LAUNCH_INTERNAL(kernel_name_, op, internal_op_, inputs, outputs, tiling_info_);
}
MS_INTERNAL_KERNEL_INFO_FACTORY_REG(PagedAttention, PagedAttention);
}  // namespace kernel
}  // namespace mindspore

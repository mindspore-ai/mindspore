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

#ifndef MINDSPORE_MINDSPORE_OPS_KERNEL_ASCEND_PYBOOST_INTERNAL_CUSTOMIZE_PAGED_ATTENTION_H_
#define MINDSPORE_MINDSPORE_OPS_KERNEL_ASCEND_PYBOOST_INTERNAL_CUSTOMIZE_PAGED_ATTENTION_H_

#include "include/pynative/utils/pyboost/op_runner.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
mindspore::tensor::TensorPtr InternalPagedAttentionAscendCustomize(
  const OpPtr &op, const mindspore::tensor::TensorPtr &query_tensor,
  const mindspore::tensor::TensorPtr &key_cache_tensor,
  const std::optional<mindspore::tensor::TensorPtr> &value_cache_tensor,
  const std::optional<mindspore::tensor::TensorPtr> &block_tables_tensor,
  const std::optional<mindspore::tensor::TensorPtr> &context_lens_tensor,
  const std::optional<mindspore::tensor::TensorPtr> &antiquant_scale_tensor,
  const std::optional<mindspore::tensor::TensorPtr> &antiquant_offset_tensor,
  const std::optional<mindspore::tensor::TensorPtr> &attn_mask_tensor,
  const std::optional<mindspore::tensor::TensorPtr> &q_seq_lens_tensor,
  const std::optional<mindspore::tensor::TensorPtr> &alibi_mask_tensor, const mindspore::Int64ImmPtr &head_num,
  const mindspore::FP32ImmPtr &scale_value, const mindspore::Int64ImmPtr &kv_head_num,
  const mindspore::Int64ImmPtr &kv_cache_quant_mode, const mindspore::Int64ImmPtr &mask_mode,
  const mindspore::Int64ImmPtr &mla_v_dim);
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_OPS_KERNEL_ASCEND_PYBOOST_INTERNAL_CUSTOMIZE_PAGED_ATTENTION_H_

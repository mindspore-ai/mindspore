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

#ifndef MINDSPORE_OPS_KERNEL_ASCEND_PYBOOST_CUSTOMIZE_SPARSE_FLASH_ATTENTION_H_
#define MINDSPORE_OPS_KERNEL_ASCEND_PYBOOST_CUSTOMIZE_SPARSE_FLASH_ATTENTION_H_

#include <memory>
#include <vector>
#include <optional>
#include "ir/tensor.h"
#include "ir/value.h"
#include "runtime/hardware_abstract/device_context/device_context_manager.h"
#include "mindspore/ccsrc/pyboost/op_runner.h"

namespace mindspore {
namespace kernel {
namespace pyboost {

tensor::TensorPtr SparseFlashAttentionAscendCustomize(
  const std::shared_ptr<OpRunner> &op, const TensorPtr &query, const TensorPtr &key, const TensorPtr &value,
  const TensorPtr &sparse_indices, const FP32ImmPtr &scale_value, const std::optional<TensorPtr> &block_table,
  const std::optional<TensorPtr> &actual_seq_lengths_query, const std::optional<TensorPtr> &actual_seq_lengths_kv,
  const std::optional<TensorPtr> &query_rope, const std::optional<TensorPtr> &key_rope,
  const Int64ImmPtr &sparse_block_size, const Int64ImmPtr &layout_query, const Int64ImmPtr &layout_kv,
  const Int64ImmPtr &sparse_mode, const Int64ImmPtr &pre_tokens, const Int64ImmPtr &next_tokens,
  const Int64ImmPtr &attention_mode, const BoolImmPtr &return_softmax_lse);

}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_OPS_KERNEL_ASCEND_PYBOOST_CUSTOMIZE_SPARSE_FLASH_ATTENTION_H_

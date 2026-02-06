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

#ifndef MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_CALL_FLASH_ATTENTION_SCORE_GRAD_H_
#define MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_CALL_FLASH_ATTENTION_SCORE_GRAD_H_

#include <vector>
#include <memory>
#include "ir/tensor.h"
#include "ir/value.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "include/pynative/utils/pyboost/op_runner.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr FlashAttentionScoreGradAscendCustomize(
  const std::shared_ptr<OpRunner> &op, const TensorPtr &query, const TensorPtr &key, const TensorPtr &value,
  const TensorPtr &dy, const std::optional<TensorPtr> &pse_shift, const std::optional<TensorPtr> &drop_mask,
  const std::optional<TensorPtr> &padding_mask, const std::optional<TensorPtr> &attn_mask,
  const std::optional<TensorPtr> &softmax_max, const std::optional<TensorPtr> &softmax_sum,
  const std::optional<TensorPtr> &softmax_in, const std::optional<TensorPtr> &attention_in,
  const std::optional<ValueTuplePtr> &prefix, const std::optional<ValueTuplePtr> &actual_seq_qlen,
  const std::optional<ValueTuplePtr> &actual_seq_kvlen, const Int64ImmPtr head_num, const FP32ImmPtr keep_prob,
  const FP32ImmPtr scale_value, const Int64ImmPtr pre_tokens, const Int64ImmPtr next_tokens,
  const Int64ImmPtr inner_precise, const Int64ImmPtr input_layout, const Int64ImmPtr sparse_mode);
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_CALL_FLASH_ATTENTION_SCORE_GRAD_H_

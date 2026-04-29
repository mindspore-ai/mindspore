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

#ifndef MINDSPORE_MINDSPORE_OPS_KERNEL_ASCEND_PYBOOST_CUSTOMIZE_NSA_SELECT_ATTENTION_H_
#define MINDSPORE_MINDSPORE_OPS_KERNEL_ASCEND_PYBOOST_CUSTOMIZE_NSA_SELECT_ATTENTION_H_

#include <memory>
#include <tuple>
#include <vector>
#include "ir/tensor.h"
#include "ir/value.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::tuple<tensor::TensorPtr, tensor::TensorPtr, tensor::TensorPtr> NsaSelectAttentionAscendCustomize(
  const tensor::TensorPtr &query, const tensor::TensorPtr &key, const tensor::TensorPtr &value,
  const tensor::TensorPtr &topk_indices, const FP32ImmPtr &scale_value, const Int64ImmPtr &head_num,
  const Int64ImmPtr &select_block_size, const Int64ImmPtr &select_block_count,
  const std::optional<tensor::TensorPtr> &atten_mask, const std::optional<ValueTuplePtr> &actual_seq_qlen,
  const std::optional<ValueTuplePtr> &actual_seq_kvlen);
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_OPS_KERNEL_ASCEND_PYBOOST_CUSTOMIZE_NSA_SELECT_ATTENTION_H_

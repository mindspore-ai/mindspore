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

#ifndef MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_CUSTOMIZE_INCRE_FLASH_ATTENTION_H_
#define MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_CUSTOMIZE_INCRE_FLASH_ATTENTION_H_
#include <vector>
#include <memory>
#include "ir/tensor.h"
#include "ir/value.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "include/pynative/utils/pyboost/op_runner.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr IncreFlashAttentionAscendCustomize(
  const std::shared_ptr<OpRunner> &op, const TensorPtr &query_tensor, const ValueTuplePtr &key_tensor_list,
  const ValueTuplePtr &value_tensor_list, const std::optional<TensorPtr> &attn_mask_tensor,
  const std::optional<TensorPtr> &actual_seq_lengths_tensor, const std::optional<TensorPtr> &pse_shift_tensor,
  const std::optional<TensorPtr> &dequant_scale1_tensor, const std::optional<TensorPtr> &quant_scale1_tensor,
  const std::optional<TensorPtr> &dequant_scale2_tensor, const std::optional<TensorPtr> &quant_scale2_tensor,
  const std::optional<TensorPtr> &quant_offset2_tensor, const std::optional<TensorPtr> &antiquant_scale_tensor,
  const std::optional<TensorPtr> &antiquant_offset_tensor, const std::optional<TensorPtr> &block_table_tensor,
  const std::optional<TensorPtr> &kv_padding_size_tensor, const Int64ImmPtr &num_heads, const Int64ImmPtr &input_layout,
  const FP32ImmPtr &scale_value, const Int64ImmPtr &num_key_value_heads, const Int64ImmPtr &block_size,
  const Int64ImmPtr &inner_precise);
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_CUSTOMIZE_INCRE_FLASH_ATTENTION_H_

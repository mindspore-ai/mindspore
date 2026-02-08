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

#ifndef MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_CUSTOMIZE_MOE_DISTRIBUTE_COMBINE_H_
#define MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_CUSTOMIZE_MOE_DISTRIBUTE_COMBINE_H_

#include <memory>
#include <tuple>
#include "ir/tensor.h"
#include "ir/scalar.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "include/pynative/utils/pyboost/op_runner.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void MoeDistributeCombineAscendCustomize(
  const std::shared_ptr<OpRunner> &op, const TensorPtr &expand_x, const TensorPtr &expert_ids,
  const TensorPtr &expert_idx, const TensorPtr &ep_send_count, const TensorPtr &expert_scale,
  const Int64ImmPtr &ep_world_size, const Int64ImmPtr &ep_rank_id, const Int64ImmPtr &moe_expert_num,
  const std::optional<TensorPtr> &tp_send_count, const std::optional<TensorPtr> &x_activate_mask,
  const std::optional<TensorPtr> &activate_scale, const std::optional<TensorPtr> &weight_scale,
  const std::optional<TensorPtr> &group_list, const std::optional<TensorPtr> &expand_scale,
  const std::optional<StringImmPtr> &group_ep, const std::optional<StringImmPtr> &group_tp,
  const Int64ImmPtr &tp_world_size, const Int64ImmPtr &tp_rank_id, const Int64ImmPtr &expert_shard_type,
  const Int64ImmPtr &shard_expert_num, const Int64ImmPtr &shard_expert_rank_num, const Int64ImmPtr &global_bs,
  const Int64ImmPtr &out_dtype, const Int64ImmPtr &common_quant_mode, const Int64ImmPtr &group_list_type);

}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_CUSTOMIZE_MOE_DISTRIBUTE_COMBINE_H_

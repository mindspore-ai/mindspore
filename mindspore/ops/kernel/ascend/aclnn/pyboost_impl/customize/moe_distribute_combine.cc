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

#include "kernel/ascend/aclnn/pyboost_impl/customize/moe_distribute_combine.h"
#include <cstdint>
#include <memory>
#include <vector>
#include <tuple>
#include <string>
#include "include/utils/utils.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "plugin/ascend/res_manager/collective/ascend_collective_comm_lib.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"

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
  const Int64ImmPtr &out_dtype, const Int64ImmPtr &common_quant_mode, const Int64ImmPtr &group_list_type) {
  OpRunner::InferOpOutput(op, expand_x, expert_ids, expert_idx, ep_send_count, expert_scale, expand_scale, group_ep,
                          ep_world_size, ep_rank_id, moe_expert_num, global_bs, tp_send_count, x_activate_mask,
                          activate_scale, weight_scale, group_list, group_tp, tp_world_size, tp_rank_id,
                          expert_shard_type, shard_expert_rank_num, out_dtype, common_quant_mode, group_list_type);
  // Convert ValuePtr to c++ scalar
  int64_t ep_world_size_imm = GetValue<int64_t>(ep_world_size);
  int64_t ep_rank_id_imm = GetValue<int64_t>(ep_rank_id);
  int64_t moe_expert_num_imm = GetValue<int64_t>(moe_expert_num);
  std::string group_ep_str = group_ep.has_value() ? GetValue<std::string>(group_ep.value()) : kHcclWorldGroup;
  std::string group_ep_imm = device::ascend::OpApiUtil::GetCommName(group_ep_str);
  std::string group_tp_str = group_tp.has_value() ? GetValue<std::string>(group_tp.value()) : "";
  std::string group_tp_imm = group_tp_str.empty() ? "" : device::ascend::OpApiUtil::GetCommName(group_ep_str);
  int64_t tp_world_size_imm = GetValue<int64_t>(tp_world_size);
  int64_t tp_rank_id_imm = GetValue<int64_t>(tp_rank_id);
  int64_t expert_shard_type_imm = GetValue<int64_t>(expert_shard_type);
  int64_t shard_expert_num_imm = GetValue<int64_t>(shard_expert_num);
  int64_t shard_expert_rank_num_imm = GetValue<int64_t>(shard_expert_rank_num);
  int64_t global_bs_imm = GetValue<int64_t>(global_bs);
  int64_t out_dtype_imm = GetValue<int64_t>(out_dtype);
  int64_t common_quant_mode_imm = GetValue<int64_t>(common_quant_mode);
  int64_t group_list_type_imm = GetValue<int64_t>(group_list_type);
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), expand_x, expert_ids, expert_idx, ep_send_count,
                                expert_scale, tp_send_count, x_activate_mask, activate_scale, weight_scale, group_list,
                                expand_scale);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, expand_x, expert_ids, expert_idx, ep_send_count, expert_scale, tp_send_count, x_activate_mask, activate_scale,
     weight_scale, group_list, expand_scale, group_ep_imm, ep_world_size_imm, ep_rank_id_imm, moe_expert_num_imm,
     group_tp_imm, tp_world_size_imm, tp_rank_id_imm, expert_shard_type_imm, shard_expert_num_imm,
     shard_expert_rank_num_imm, global_bs_imm, out_dtype_imm, common_quant_mode_imm, group_list_type_imm]() {
      MS_LOG(DEBUG) << "Run device task MoeDistributeCombine start";
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(op->device_context(), expand_x, expert_ids, expert_idx, ep_send_count, expert_scale,
                                   tp_send_count, x_activate_mask, activate_scale, weight_scale, group_list,
                                   expand_scale);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(op->device_context(), outputs);
      LAUNCH_ACLNN(aclnnMoeDistributeCombine, device_context, op->stream_id(), expand_x, expert_ids, expert_idx,
                   ep_send_count, expert_scale, tp_send_count, x_activate_mask, activate_scale, weight_scale,
                   group_list, expand_scale, group_ep_imm, ep_world_size_imm, ep_rank_id_imm, moe_expert_num_imm,
                   group_tp_imm, tp_world_size_imm, tp_rank_id_imm, expert_shard_type_imm, shard_expert_num_imm,
                   shard_expert_rank_num_imm, global_bs_imm, out_dtype_imm, common_quant_mode_imm, group_list_type_imm,
                   outputs[0]);
      MS_LOG(DEBUG) << "Run device task MoeDistributeCombine end";
    }));
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore

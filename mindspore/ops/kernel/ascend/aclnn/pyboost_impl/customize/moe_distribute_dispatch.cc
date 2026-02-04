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

#include "kernel/ascend/aclnn/pyboost_impl/customize/moe_distribute_dispatch.h"
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
constexpr size_t kExpandX = 0;
constexpr size_t kDynamicScales = 1;
constexpr size_t kExpandIdx = 2;
constexpr size_t kExpertTokenNums = 3;
constexpr size_t kEpRecvCounts = 4;
constexpr size_t kTpRecvCounts = 5;
constexpr size_t kExpandScales = 6;

struct MoeDispatchParams {
  int64_t ep_world_size;
  int64_t ep_rank_id;
  int64_t moe_expert_num;
  std::string group_ep;
  std::string group_tp;
  int64_t tp_world_size;
  int64_t tp_rank_id;
  int64_t expert_shard_type;
  int64_t shared_expert_num;
  int64_t shared_expert_rank_num;
  int64_t quant_mode;
  int64_t global_bs;
  int64_t expert_token_nums_type;
};

MoeDispatchParams ParseMoeDispatchParams(const Int64ImmPtr &ep_world_size, const Int64ImmPtr &ep_rank_id,
                                         const Int64ImmPtr &moe_expert_num, const std::optional<StringImmPtr> &group_ep,
                                         const std::optional<StringImmPtr> &group_tp, const Int64ImmPtr &tp_world_size,
                                         const Int64ImmPtr &tp_rank_id, const Int64ImmPtr &expert_shard_type,
                                         const Int64ImmPtr &shared_expert_num,
                                         const Int64ImmPtr &shared_expert_rank_num, const Int64ImmPtr &quant_mode,
                                         const Int64ImmPtr &global_bs, const Int64ImmPtr &expert_token_nums_type) {
  MoeDispatchParams params;
  params.ep_world_size = GetValue<int64_t>(ep_world_size);
  params.ep_rank_id = GetValue<int64_t>(ep_rank_id);
  params.moe_expert_num = GetValue<int64_t>(moe_expert_num);
  std::string group_ep_str = group_ep.has_value() ? GetValue<std::string>(group_ep.value()) : kHcclWorldGroup;
  params.group_ep = device::ascend::OpApiUtil::GetCommName(group_ep_str);
  std::string group_tp_str = group_tp.has_value() ? GetValue<std::string>(group_tp.value()) : "";
  params.group_tp = group_tp_str.empty() ? "" : device::ascend::OpApiUtil::GetCommName(group_ep_str);
  params.tp_world_size = GetValue<int64_t>(tp_world_size);
  params.tp_rank_id = GetValue<int64_t>(tp_rank_id);
  params.expert_shard_type = GetValue<int64_t>(expert_shard_type);
  params.shared_expert_num = GetValue<int64_t>(shared_expert_num);
  params.shared_expert_rank_num = GetValue<int64_t>(shared_expert_rank_num);
  params.quant_mode = GetValue<int64_t>(quant_mode);
  params.global_bs = GetValue<int64_t>(global_bs);
  params.expert_token_nums_type = GetValue<int64_t>(expert_token_nums_type);
  return params;
}

std::shared_ptr<runtime::PyBoostDeviceTask> CreateMoeDispatchTask(const std::shared_ptr<OpRunner> &op,
                                                                  const TensorPtr &x, const TensorPtr &expert_ids,
                                                                  const std::optional<TensorPtr> &expert_scales,
                                                                  const std::optional<TensorPtr> &scales,
                                                                  const std::optional<TensorPtr> &x_activate_mask,
                                                                  const MoeDispatchParams &params) {
  return std::make_shared<runtime::PyBoostDeviceTask>(
    [op, x, expert_ids, scales, x_activate_mask, expert_scales, params]() {
      MS_LOG(DEBUG) << "Run device task MoeDistributeDispatch start";
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      PyBoostUtils::MallocOpInputs(device_context, x, expert_ids, scales, x_activate_mask, expert_scales);
      PyBoostUtils::MallocOpOutputs(device_context, outputs);
      LAUNCH_ACLNN(aclnnMoeDistributeDispatch, device_context, op->stream_id(), x, expert_ids, scales, x_activate_mask,
                   expert_scales, params.group_ep, params.ep_world_size, params.ep_rank_id, params.moe_expert_num,
                   params.group_tp, params.tp_world_size, params.tp_rank_id, params.expert_shard_type,
                   params.shared_expert_num, params.shared_expert_rank_num, params.quant_mode, params.global_bs,
                   params.expert_token_nums_type, outputs[kExpandX], outputs[kDynamicScales], outputs[kExpandIdx],
                   outputs[kExpertTokenNums], outputs[kEpRecvCounts], outputs[kTpRecvCounts], outputs[kExpandScales]);
      MS_LOG(DEBUG) << "Run device task MoeDistributeDispatch end";
    });
}

void MoeDistributeDispatchAscendCustomize(
  const std::shared_ptr<OpRunner> &op, const TensorPtr &x, const TensorPtr &expert_ids,
  const Int64ImmPtr &ep_world_size, const Int64ImmPtr &ep_rank_id, const Int64ImmPtr &moe_expert_num,
  const std::optional<TensorPtr> &expert_scales, const std::optional<TensorPtr> &scales,
  const std::optional<TensorPtr> &x_activate_mask, const std::optional<StringImmPtr> &group_ep,
  const std::optional<StringImmPtr> &group_tp, const Int64ImmPtr &tp_world_size, const Int64ImmPtr &tp_rank_id,
  const Int64ImmPtr &expert_shard_type, const Int64ImmPtr &shared_expert_num, const Int64ImmPtr &shared_expert_rank_num,
  const Int64ImmPtr &quant_mode, const Int64ImmPtr &global_bs, const Int64ImmPtr &expert_token_nums_type) {
  OpRunner::InferOpOutput(op, x, expert_ids, ep_world_size, ep_rank_id, moe_expert_num, expert_scales, scales,
                          x_activate_mask, group_ep, group_tp, tp_world_size, tp_rank_id, expert_shard_type,
                          shared_expert_num, shared_expert_rank_num, quant_mode, global_bs, expert_token_nums_type);
  auto params = ParseMoeDispatchParams(ep_world_size, ep_rank_id, moe_expert_num, group_ep, group_tp, tp_world_size,
                                       tp_rank_id, expert_shard_type, shared_expert_num, shared_expert_rank_num,
                                       quant_mode, global_bs, expert_token_nums_type);
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), x, expert_ids, scales, x_activate_mask,
                                expert_scales);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());
  PyBoostUtils::DispatchRun(CreateMoeDispatchTask(op, x, expert_ids, expert_scales, scales, x_activate_mask, params));
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore

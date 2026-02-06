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

#include "kernel/ascend/aclnn/pyboost_impl/customize/all_gather_matmul.h"
#include <memory>
#include <string>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/op_register.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "mindspore/ccsrc/include/pynative/utils/pyboost/functions/auto_generate/functions.h"
#include "include/pynative/utils/pyboost/functions/auto_grad_guard.h"
#include "kernel/ascend/acl_ir/op_api_util.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
std::vector<int64_t> AllGatherMatmulGetTransposePerm(const TensorPtr &tensor) {
  std::vector<int64_t> perm(tensor->shape().size());
  perm[kDim0] = static_cast<int64_t>(kDim1);
  perm[kDim1] = static_cast<int64_t>(kDim0);
  return perm;
}
}  // namespace

std::vector<tensor::TensorPtr> AllGatherMatmulAscendCustomize(
  const std::shared_ptr<OpRunner> &op, const TensorPtr &input, const TensorPtr &x2, const StringImmPtr &group,
  const Int64ImmPtr &world_size, const std::optional<TensorPtr> &bias, const Int64ImmPtr &gather_index,
  const BoolImmPtr &gather_output, const Int64ImmPtr &comm_turn, const BoolImmPtr &trans_input,
  const BoolImmPtr &trans_x2) {
  MS_LOG(DEBUG) << op->primitive()->name() << " call start";

  OpRunner::InferOpOutput(op, input, x2, group, world_size, bias, gather_index, gather_output, comm_turn, trans_input,
                          trans_x2);
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input, x2, bias);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  auto group_imm = GetValue<std::string>(group);
  auto world_size_imm = GetValue<int64_t>(world_size);
  auto gather_index_imm = GetValue<int64_t>(gather_index);
  auto comm_turn_imm = GetValue<int64_t>(comm_turn);
  auto trans_input_imm = GetValue<bool>(trans_input);
  auto trans_x2_imm = GetValue<bool>(trans_x2);

  auto hccl_inner_comm_name_imm = mindspore::device::ascend::OpApiUtil::GetCommName(group_imm);
  mindspore::device::ascend::OpApiUtil::CheckWorldSize(group_imm, world_size_imm, op->primitive()->name());
  TensorPtr input_ = input;
  TensorPtr x2_ = x2;

  kernel::pyboost::RequireGradGuard require_grad_guard(false);
  if (trans_input_imm) {
    input_ = transpose(input, AllGatherMatmulGetTransposePerm(input));
  }
  if (trans_x2_imm) {
    x2_ = transpose(x2, AllGatherMatmulGetTransposePerm(x2));
  }

  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, input_, x2_, hccl_inner_comm_name_imm, bias, gather_index_imm, comm_turn_imm]() {
      MS_LOG(DEBUG) << op->primitive()->name() << " run device task start";
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      PyBoostUtils::MallocOpInputs(device_context, input_, x2_, bias);
      PyBoostUtils::MallocOpOutputs(device_context, outputs);
      constexpr int64_t stream_mode = 1;
      if (bias.has_value()) {
        LAUNCH_ACLNN(aclnnAllGatherMatmul, device_context, op->stream_id(), input_, x2_, bias, hccl_inner_comm_name_imm,
                     gather_index_imm, comm_turn_imm, stream_mode, outputs[0], outputs[1]);
      } else {
        LAUNCH_ACLNN(aclnnAllGatherMatmul, device_context, op->stream_id(), input_, x2_, nullptr,
                     hccl_inner_comm_name_imm, gather_index_imm, comm_turn_imm, stream_mode, outputs[0], outputs[1]);
      }
      MS_LOG(DEBUG) << op->primitive()->name() << " run device task end";
    }));
  return op->outputs();
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore

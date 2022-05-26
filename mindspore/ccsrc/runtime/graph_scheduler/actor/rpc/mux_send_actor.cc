/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include <utility>
#include "runtime/graph_scheduler/actor/rpc/mux_send_actor.h"

namespace mindspore {
namespace runtime {
void MuxSendActor::SendOutput(OpContext<DeviceTensor> *const context) {
  MS_ERROR_IF_NULL_WO_RET_VAL(context);
  MS_ERROR_IF_NULL_WO_RET_VAL(client_);
  // Step 1: Send data and control outputs.
  AbstractActor::SendOutput(context);

  // Step 2: Erase inter-process inputs for this sequential number.
  if (input_op_inter_process_.count(context->sequential_num_) != 0) {
    input_op_inter_process_.erase(context->sequential_num_);
  }

  // Step 3: Send input data(inter-process data is the input of the Send kernel) to peers.
  if (launch_info_.inputs_.empty()) {
    MS_LOG(ERROR) << "Send kernel has no output tensor.";
    return;
  }
  auto send_output = launch_info_.inputs_;

  MS_EXCEPTION_IF_NULL(mux_recv_actor_);
  std::string peer_server_url = mux_recv_actor_->from_actor_aid().Url();
  auto message = BuildRpcMessage(send_output, peer_server_url);
  MS_ERROR_IF_NULL_WO_RET_VAL(message);
  MS_LOG(INFO) << "Rpc actor send message to actor: " << mux_recv_actor_->from_actor_aid();
  client_->SendAsync(std::move(message));
}
}  // namespace runtime
}  // namespace mindspore

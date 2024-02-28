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
bool MuxSendActor::LaunchKernel(OpContext<DeviceTensor> *const context) {
  MS_ERROR_IF_NULL(client_);
  if (!KernelActor::LaunchKernel(context)) {
    MS_LOG(ERROR) << "Launching kernel for send actor failed.";
    return false;
  }

  // Send input data(inter-process data is the input of the Send kernel) to peers.
  if (launch_info_.inputs_.empty()) {
    MS_LOG(ERROR) << "Send kernel has no output tensor.";
    return false;
  }

  auto send_output = launch_info_.inputs_;
  MS_EXCEPTION_IF_NULL(mux_recv_actor_);
  std::string peer_server_url = mux_recv_actor_->from_actor_aid().Url();
  auto message = BuildRpcMessage(send_output, peer_server_url);
  MS_EXCEPTION_IF_NULL(message);
  MS_LOG(INFO) << "Rpc actor send message to: " << peer_server_url;
  client_->SendAsync(std::move(message));
  return true;
}
}  // namespace runtime
}  // namespace mindspore

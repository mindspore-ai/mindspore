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

#include "runtime/graph_scheduler/actor/rpc/recv_actor.h"

#include <utility>
#include "plugin/device/cpu/kernel/rpc/rpc_recv_kernel.h"

namespace mindspore {
namespace runtime {
void RecvActor::RunOpInterProcessData(std::unique_ptr<MessageBase> &&msg, OpContext<DeviceTensor> *const context) {
  MS_ERROR_IF_NULL_WO_RET_VAL(msg);
  MS_ERROR_IF_NULL_WO_RET_VAL(op_context_);
  auto &sequential_num = context->sequential_num_;
  (void)input_op_inter_process_[sequential_num].emplace_back(msg->From().Name());

  auto is_run = CheckRunningCondition(context);
  MS_LOG(INFO) << "Actor(" << GetAID().Name() << ") receive the input op inter-process. Edge is "
               << inter_process_edge_name_ << ". Check running condition:" << is_run;

  // Parse the message from remote peer and set to rpc recv kernel.
  auto recv_kernel_mod = dynamic_cast<kernel::RpcKernelMod *>(kernel_info_->MutableKernelMod());
  MS_ERROR_IF_NULL_WO_RET_VAL(recv_kernel_mod);
  // We set remote data by the interface of the rpc kernel, because currently there's no remote input for a kernel mod.
  recv_kernel_mod->SetRemoteInput(std::move(msg));

  if (is_run) {
    Run(context);
  }
  return;
}

void RecvActor::SetRouteInfo(uint32_t, const std::string &, const std::string &recv_src_node_name,
                             const std::string &recv_dst_node_name) {
  rpc_input_node_name_.emplace_back(recv_src_node_name);
  input_inter_process_num_++;
}

bool RecvActor::StartServer() {
  // Step 1: Create a tcp server and start listening.
  std::string server_url = ip_ + ":" + std::to_string(port_);
  server_ = std::make_unique<TCPServer>();
  MS_EXCEPTION_IF_NULL(server_);
  if (!server_->Initialize(server_url)) {
    MS_LOG(EXCEPTION) << "Failed to initialize tcp server for recv actor. Server url: " << server_url;
  }

  // Step 2: Set the message handler of the server.

  // Step 2: Register the server address to route table. The server should not be connected before this step is done.
  ActorAddress recv_actor_addresss;
  recv_actor_addresss.set_actor_id(inter_process_edge_name_);
  recv_actor_addresss.set_ip(ip_);
  recv_actor_addresss.set_port(static_cast<uint32_t>(port_));
  MS_EXCEPTION_IF_NULL(actor_route_table_proxy_);
  if (!actor_route_table_proxy_->RegisterRoute(inter_process_edge_name_, recv_actor_addresss)) {
    MS_LOG(EXCEPTION) << "Failed to register route for " << inter_process_edge_name_ << " " << server_url
                      << " when starting server.";
  }
  return true;
}

bool RecvActor::CheckRunningCondition(const OpContext<DeviceTensor> *context) const {
  MS_EXCEPTION_IF_NULL(context);
  // Step 1: Judge data and control inputs are satisfied.
  bool is_data_and_control_arrow_satisfied = AbstractActor::CheckRunningCondition(context);
  if (!is_data_and_control_arrow_satisfied) {
    return false;
  }

  if (input_inter_process_num_ != 0) {
    // Step 2: Judge inter-process inputs are satisfied.
    const auto &inter_process_iter = input_op_inter_process_.find(context->sequential_num_);
    if (inter_process_iter == input_op_inter_process_.end()) {
      return false;
    }

    const auto &current_inter_process_inputs = inter_process_iter->second;
    if (current_inter_process_inputs.size() < input_inter_process_num_) {
      return false;
    } else if (current_inter_process_inputs.size() > input_inter_process_num_) {
      MS_LOG(ERROR) << "Invalid inter process input num:" << current_inter_process_inputs.size()
                    << " need:" << input_inter_process_num_ << " for actor:" << GetAID();
      return false;
    }
  }
  return true;
}

void RecvActor::HandleMessage(std::unique_ptr<MessageBase> &&msg) {
  MS_ERROR_IF_NULL_WO_RET_VAL(msg);
  MS_ERROR_IF_NULL_WO_RET_VAL(op_context_);
  RunOpInterProcessData(std::move(msg), op_context_);
}
}  // namespace runtime
}  // namespace mindspore

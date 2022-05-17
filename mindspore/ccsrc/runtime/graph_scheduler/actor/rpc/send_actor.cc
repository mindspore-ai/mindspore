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

#include "runtime/graph_scheduler/actor/rpc/send_actor.h"

#include <utility>

namespace mindspore {
namespace runtime {
SendActor::~SendActor() {
  if (client_) {
    client_->Disconnect(server_url_);
    client_->Finalize();
  }
}

void SendActor::SetRouteInfo(uint32_t, const std::string &, const std::string &send_src_node_name,
                             const std::string &send_dst_node_name) {
  peer_actor_ids_ = inter_process_edge_names_;
  rpc_output_node_name_.emplace_back(send_dst_node_name);
}

bool SendActor::ConnectServer() {
  client_ = std::make_unique<TCPClient>();
  MS_EXCEPTION_IF_NULL(client_);
  if (!client_->Initialize()) {
    MS_LOG(EXCEPTION) << "Failed to initialize tcp server for send actor.";
  }
  // Lookup actor addresses for each peer actor.
  for (const auto &peer_actor_id : peer_actor_ids_) {
    MS_EXCEPTION_IF_NULL(actor_route_table_proxy_);
    auto peer_actor_address = actor_route_table_proxy_->LookupRoute(peer_actor_id);
    // If route is successfully looked up, peer_actor_address is not empty.
    server_url_ = peer_actor_address.ip() + ":" + std::to_string(peer_actor_address.port());
    if (!client_->Connect(server_url_)) {
      MS_LOG(EXCEPTION) << "Failed to connect to server of actor " << peer_actor_id << ", server_url: " << server_url_;
    }
    MS_LOG(INFO) << "Successfully connect to server " << server_url_ << ", inter-process edge name: " << peer_actor_id;
    peer_actor_urls_[peer_actor_id] = server_url_;
  }
  return true;
}

void SendActor::SendOutput(OpContext<DeviceTensor> *const context) {
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
  for (const auto &peer : peer_actor_urls_) {
    std::string peer_server_url = peer.second;
    auto message = BuildRpcMessage(send_output, peer_server_url);
    MS_ERROR_IF_NULL_WO_RET_VAL(message);
    MS_LOG(INFO) << "Rpc actor send message for inter-process edge: " << peer.first;
    client_->SendAsync(std::move(message));
  }
}

void SendActor::SerializeDynamicShapeMessgae(std::string *msg_body, const ShapeVector &shape_vec,
                                             const TypeId &data_type, const kernel::AddressPtr &addr) {
  MS_EXCEPTION_IF_NULL(msg_body);
  MS_EXCEPTION_IF_NULL(addr);

  rpc::DynamicShapeMessage pb_msg;
  pb_msg.set_type_id(data_type);
  *pb_msg.mutable_shape_vector() = {shape_vec.begin(), shape_vec.end()};
  std::string pb_msg_str = pb_msg.SerializeAsString();

  // 1. Magic header for dynamic shape.
  msg_body->append(kRpcDynamicShapeData);
  // 2. The size of the protobuf message DynamicShapeMessage.
  size_t pb_msg_size = pb_msg_str.size();
  msg_body->append(reinterpret_cast<char *>(&pb_msg_size), sizeof(pb_msg_size));
  // 3. Protobuf message DynamicShapeMessage.
  msg_body->append(pb_msg_str);
  // 4. The real data buffer of the input.
  msg_body->append(static_cast<char *>(addr->addr), addr->size);
}

std::unique_ptr<MessageBase> SendActor::BuildRpcMessage(const kernel::AddressPtrList &data_list,
                                                        const std::string &server_url) {
  std::unique_ptr<MessageBase> message = std::make_unique<MessageBase>();
  MS_ERROR_IF_NULL_W_RET_VAL(message, nullptr);
  message->to = AID("", server_url);

  if (is_dynamic_shape_) {
    MS_LOG(INFO) << "This send actor builds message with dynamic shape.";
    size_t input_size = common::AnfAlgo::GetInputTensorNum(kernel_);
    for (size_t i = 0; i < input_size; i++) {
      auto input_node_with_index = common::AnfAlgo::GetPrevNodeOutput(kernel_, i, false);
      auto real_input = input_node_with_index.first;
      auto real_input_index = input_node_with_index.second;
      MS_EXCEPTION_IF_NULL(real_input);

      auto shapes = trans::GetRuntimePaddingShape(real_input, real_input_index);
      for (const auto &shape : shapes) {
        MS_LOG(INFO) << "Shape of input " << real_input->fullname_with_scope() << " is " << shape;
      }
      TypeId data_type = common::AnfAlgo::GetOutputInferDataType(real_input, real_input_index);

      // Serialize the message body and append the data.
      SerializeDynamicShapeMessgae(&message->body, shapes, data_type, data_list[i]);
    }
  } else {
    size_t total_size = 0;
    total_size =
      std::accumulate(data_list.begin(), data_list.end(), total_size,
                      [](size_t total_size, const kernel::AddressPtr &output) { return total_size + output->size; });
    message->body.reserve(total_size);
    for (const auto &data : data_list) {
      message->body.append(static_cast<char *>(data->addr), data->size);
    }
  }
  return message;
}
}  // namespace runtime
}  // namespace mindspore

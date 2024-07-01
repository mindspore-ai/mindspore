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
#include "runtime/graph_scheduler/actor/memory_manager_actor.h"

namespace mindspore {
namespace runtime {
SendActor::~SendActor() {
  if (client_) {
    try {
      (void)client_->Disconnect(server_url_);
      client_->Finalize();
    } catch (const std::exception &) {
      MS_LOG(ERROR) << "Failed to disconnect and finalize for rpc client in send actor.";
    }
    client_ = nullptr;
  }
}

void SendActor::SetRouteInfo(uint32_t, const std::string &, const std::string &send_src_node_name,
                             const std::string &send_dst_node_name) {
  peer_actor_ids_ = inter_process_edge_names_;
  (void)rpc_output_node_name_.emplace_back(send_dst_node_name);
}

bool SendActor::ConnectServer() {
#ifdef ENABLE_RDMA
  if (common::GetEnv(kEnableRDMA) == "1") {
    client_ = std::make_unique<RDMAClient>();
  } else {
    client_ = std::make_unique<TCPClient>();
  }
#else
  client_ = std::make_unique<TCPClient>();
#endif
  MS_EXCEPTION_IF_NULL(client_);

  if (!client_->Initialize()) {
    MS_LOG(EXCEPTION) << "Failed to initialize rpc server for send actor.";
  }
  // Lookup actor addresses for each peer actor.
  for (const auto &peer_actor_id : peer_actor_ids_) {
    MS_EXCEPTION_IF_NULL(actor_route_table_proxy_);
    auto peer_actor_address = actor_route_table_proxy_->LookupRoute(peer_actor_id);

    // If route is successfully looked up, peer_actor_address is not empty.
    server_url_ = peer_actor_address.ip() + ":" + std::to_string(peer_actor_address.port());
    remote_func_id_ = peer_actor_address.func_id();
    auto free_callback = std::bind(&SendActor::FreeMessage, this, std::placeholders::_1);
    size_t retry_count = 60;
    if (!client_->Connect(server_url_, retry_count, free_callback)) {
      MS_LOG(EXCEPTION) << "Failed to connect to server of actor " << peer_actor_id << ", server_url: " << server_url_;
    }

    MS_LOG(INFO) << "Successfully connect to server " << server_url_ << ", remote function id: " << remote_func_id_
                 << ", inter-process edge name: " << peer_actor_id;
    peer_actor_urls_[peer_actor_id] = server_url_;
  }

  return true;
}

void SendActor::FlushData() {
  MS_EXCEPTION_IF_NULL(client_);
  if (!client_->Flush(server_url_)) {
    MS_LOG(EXCEPTION) << "Failed to flush client for server " << server_url_;
  }
}

void SendActor::Clear() {
  if (client_) {
    (void)client_->Disconnect(server_url_);
    client_->Finalize();
    client_ = nullptr;
  }
}

bool SendActor::LaunchKernel(OpContext<DeviceTensor> *const context, bool is_skip_launch) {
  if (is_skip_launch) {
    return KernelActor::LaunchKernel(context, is_skip_launch);
  }
  MS_ERROR_IF_NULL_W_RET_VAL(context, false);
  // Set context for later usage in FreeMessage.
  context_ = context;

  if (!KernelActor::LaunchKernel(context, is_skip_launch)) {
    MS_LOG(ERROR) << "Launching kernel for send actor failed.";
    return false;
  }

  // Send input data(inter-process data is the input of the Send kernel) to peers.
  if (input_device_tensors_.empty()) {
    MS_LOG(ERROR) << "Send kernel has no output tensor.";
    return false;
  }
  for (const auto &peer : peer_actor_urls_) {
    std::string peer_server_url = peer.second;
    auto message = BuildRpcMessage(peer_server_url);
    MS_ERROR_IF_NULL_W_RET_VAL(message, false);
    MS_ERROR_IF_NULL_W_RET_VAL(client_, false);
    MS_LOG(INFO) << "Rpc actor send message for inter-process edge: " << peer.first;
    client_->SendAsync(std::move(message));
  }
  return true;
}

void SendActor::EraseInput(const OpContext<DeviceTensor> *context) {
  MS_EXCEPTION_IF_NULL(context);
  AbstractActor::EraseInput(context);

  if (input_op_inter_process_.count(context->sequential_num_) != 0) {
    (void)input_op_inter_process_.erase(context->sequential_num_);
  }
}

std::unique_ptr<MessageBase> SendActor::BuildRpcMessage(const std::string &server_url) {
  std::unique_ptr<MessageBase> message = std::make_unique<MessageBase>();
  MS_ERROR_IF_NULL_W_RET_VAL(message, nullptr);
  message->to = AID("", server_url);
  message->func_id_ = remote_func_id_;

  // To reach optimal performance, we use workspace memory as the data sent to the remote. So the size must be
  // strictly checked to avoid illegal memory access.
  auto send_workspace = workspace_device_tensors_;
  if (send_workspace.empty()) {
    MS_LOG(EXCEPTION) << "RpcSendKernel's workspace should not be empty.";
  }
  // Only use one piece of workspace memory to avoid extra memory copying and serialize inputs data to one message.
  auto workspace_addr = send_workspace[kIndex0];
  if (is_dynamic_shape_) {
    MS_LOG(INFO) << "This send actor builds message with dynamic shape.";
    SerializeDynamicShapeMessage(message.get(), workspace_addr);
  } else {
    SerializeCommonMessage(message.get(), workspace_addr);
  }

  MS_LOG(DEBUG) << "RpcSend message size is " << message->size;
  return message;
}

bool SendActor::FreeMessage(void *data) {
  auto memory_free_list = FindDeviceTensorNeedsFree(data);
  ActorDispatcher::SendSync(memory_manager_aid_, &MemoryManagerActor::FreeMemory, &memory_free_list,
                            device_contexts_[0], context_, GetAID());
  return true;
}

void SendActor::Flush() {
  MS_EXCEPTION_IF_NULL(client_);
  for (const auto &url : peer_actor_urls_) {
    MS_LOG(DEBUG) << "Flush for url " << url.second;
    if (!client_->Flush(url.second)) {
      MS_LOG(EXCEPTION) << "Failed to flush for url " << url.second;
    }
  }
}

std::vector<DeviceTensor *> SendActor::FindDeviceTensorNeedsFree(const void *data) const {
  std::vector<DeviceTensor *> free_list;
  // The sent data uses the memory of workspace. So query the DeviceTensor from workspace_device_tensors_.
  for (const auto &device_tensor : workspace_device_tensors_) {
    MS_ERROR_IF_NULL_W_RET_VAL(device_tensor, {});
    if (data == device_tensor->GetMutablePtr()) {
      free_list.push_back(device_tensor);
    }
  }
  return free_list;
}

size_t SendActor::SerializeSingleDynamicShapeInput(RpcDataPtr rpc_data, const ShapeVector &shape_vec,
                                                   const TypeId &data_type, const DeviceTensor *addr) const {
  MS_EXCEPTION_IF_NULL(rpc_data);
  MS_EXCEPTION_IF_NULL(addr);

  // The serialize data size needs to be computed.
  size_t serialized_data_size = 0;

  // Serialize data's meta info to protobuffer.
  rpc::DynamicShapeMessage pb_msg;
  pb_msg.set_type_id(static_cast<int>(data_type));
  *pb_msg.mutable_shape_vector() = {shape_vec.begin(), shape_vec.end()};
  std::string pb_msg_str = pb_msg.SerializeAsString();

  // Part 1. Magic header for dynamic shape.
  size_t header_size = strlen(kRpcDynamicShapeData);
  if (!CopyRpcDataWithOffset(&rpc_data, kRpcDynamicShapeData, header_size)) {
    MS_LOG(EXCEPTION) << "Failed to copy data for kRpcDynamicShapeData.";
  }
  serialized_data_size += header_size;

  // Part 2. The size of the protobuf message DynamicShapeMessage.
  size_t pb_msg_size = pb_msg_str.size();
  if (!CopyRpcDataWithOffset(&rpc_data, &pb_msg_size, sizeof(pb_msg_size))) {
    MS_LOG(EXCEPTION) << "Failed to copy data for protobuffer data's size.";
  }
  serialized_data_size += sizeof(pb_msg_size);

  // Part 3. Protobuf message DynamicShapeMessage.
  if (!CopyRpcDataWithOffset(&rpc_data, pb_msg_str.c_str(), pb_msg_str.size())) {
    MS_LOG(EXCEPTION) << "Failed to copy data for protobuffer data.";
  }
  serialized_data_size += pb_msg_str.size();

  // Part 4. The real data buffer of the input.
  if (!CopyRpcDataWithOffset(&rpc_data, addr->GetMutablePtr(), addr->GetSize())) {
    MS_LOG(EXCEPTION) << "Failed to copy data for real input data.";
  }
  serialized_data_size += addr->GetSize();

  return serialized_data_size;
}

void SendActor::SerializeDynamicShapeMessage(MessageBase *message, const DeviceTensor *workspace_addr) const {
  MS_EXCEPTION_IF_NULL(workspace_addr);
  size_t offset = 0;
  RpcDataPtr rpc_data = static_cast<RpcDataPtr>(workspace_addr->GetMutablePtr());
  for (size_t i = 0; i < input_kernel_tensors_.size(); i++) {
    auto shapes = input_kernel_tensors_[i]->GetShapeVector();
    TypeId data_type = input_kernel_tensors_[i]->dtype_id();
    size_t serialized_data_size =
      SerializeSingleDynamicShapeInput(rpc_data + offset, shapes, data_type, input_device_tensors_[i]);
    offset += serialized_data_size;
  }

  if (workspace_addr->GetSize() != offset) {
    MS_LOG(EXCEPTION) << "Send void data size is not the same as workspace size.";
  }
  MS_EXCEPTION_IF_NULL(message);
  message->data = workspace_addr->GetMutablePtr();
  message->size = workspace_addr->GetSize();
}

void SendActor::SerializeCommonMessage(MessageBase *message, const DeviceTensor *workspace_addr) const {
  MS_EXCEPTION_IF_NULL(message);
  MS_EXCEPTION_IF_NULL(workspace_addr);
  MS_EXCEPTION_IF_NULL(workspace_addr->GetMutablePtr());
  size_t total_size = 0;
  total_size =
    std::accumulate(input_device_tensors_.begin(), input_device_tensors_.end(), total_size,
                    [](size_t total_size, const DeviceTensor *output) { return total_size + output->GetSize(); });
  if (workspace_addr->GetSize() != total_size) {
    MS_LOG(EXCEPTION) << "Workspace size should be the same as inputs size. But got " << workspace_addr->GetSize()
                      << " and " << total_size;
  }

  RpcDataPtr rpc_data = static_cast<RpcDataPtr>(workspace_addr->GetMutablePtr());
  MS_EXCEPTION_IF_NULL(rpc_data);
  for (size_t i = 0; i < input_device_tensors_.size(); i++) {
    MS_EXCEPTION_IF_NULL(input_device_tensors_[i]);
    if (!CopyRpcDataWithOffset(&rpc_data, input_device_tensors_[i]->GetMutablePtr(),
                               input_device_tensors_[i]->GetSize())) {
      MS_LOG(EXCEPTION) << "Failed to copy data for rpc send input " << i;
    }
  }
  message->data = workspace_addr->GetMutablePtr();
  message->size = workspace_addr->GetSize();
}

}  // namespace runtime
}  // namespace mindspore

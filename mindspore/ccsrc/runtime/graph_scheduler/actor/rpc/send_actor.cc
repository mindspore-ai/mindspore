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
    auto free_callback = std::bind(&SendActor::FreeMessage, this, std::placeholders::_1);
    size_t retry_count = 60;
    if (!client_->Connect(server_url_, retry_count, free_callback)) {
      MS_LOG(EXCEPTION) << "Failed to connect to server of actor " << peer_actor_id << ", server_url: " << server_url_;
    }

    MS_LOG(INFO) << "Successfully connect to server " << server_url_ << ", inter-process edge name: " << peer_actor_id;
    peer_actor_urls_[peer_actor_id] = server_url_;
  }

  return true;
}

void SendActor::FlushData() {
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

bool SendActor::LaunchKernel(OpContext<DeviceTensor> *const context) {
  MS_ERROR_IF_NULL_W_RET_VAL(context, false);
  // Set context for later usage in FreeMessage.
  context_ = context;

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
  for (const auto &peer : peer_actor_urls_) {
    std::string peer_server_url = peer.second;
    auto message = BuildRpcMessage(send_output, peer_server_url);
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

std::unique_ptr<MessageBase> SendActor::BuildRpcMessage(const kernel::AddressPtrList &data_list,
                                                        const std::string &server_url) {
  std::unique_ptr<MessageBase> message = std::make_unique<MessageBase>();
  MS_ERROR_IF_NULL_W_RET_VAL(message, nullptr);
  message->to = AID("", server_url);

  // To reach optimal performance, we use workspace memory as the data sent to the remote. So the size must be
  // strictly checked to avoid illegal memory access.
  auto send_workspace = launch_info_.workspaces_;
  if (send_workspace.empty()) {
    MS_LOG(EXCEPTION) << "RpcSendKernel's workspace should not be empty.";
  }
  // Only use one piece of workspace memory to avoid extra memory copying and serialize inputs data to one message.
  auto workspace_addr = send_workspace[kIndex0];
  if (is_dynamic_shape_) {
    MS_LOG(INFO) << "This send actor builds message with dynamic shape.";
    SerializeDynamicShapeMessage(message.get(), data_list, workspace_addr);
  } else {
    SerializeCommonMessage(message.get(), data_list, workspace_addr);
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
                                                   const TypeId &data_type, const kernel::AddressPtr &addr) const {
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
  if (!CopyRpcDataWithOffset(&rpc_data, addr->addr, addr->size)) {
    MS_LOG(EXCEPTION) << "Failed to copy data for real input data.";
  }
  serialized_data_size += addr->size;

  return serialized_data_size;
}

void SendActor::SerializeDynamicShapeMessage(MessageBase *message, const kernel::AddressPtrList &data_list,
                                             const kernel::AddressPtr &workspace_addr) const {
  MS_EXCEPTION_IF_NULL(workspace_addr);
  size_t offset = 0;
  RpcDataPtr rpc_data = static_cast<RpcDataPtr>(workspace_addr->addr);
  size_t input_size = common::AnfAlgo::GetInputTensorNum(kernel_);
  for (size_t i = 0; i < input_size; i++) {
    auto input_node_with_index = common::AnfAlgo::GetPrevNodeOutput(kernel_, i, false);
    auto real_input = input_node_with_index.first;
    auto real_input_index = input_node_with_index.second;
    MS_EXCEPTION_IF_NULL(real_input);

    auto shapes = trans::GetRuntimePaddingShape(real_input, real_input_index);
    TypeId data_type = common::AnfAlgo::GetOutputInferDataType(real_input, real_input_index);

    size_t serialized_data_size = SerializeSingleDynamicShapeInput(rpc_data + offset, shapes, data_type, data_list[i]);
    offset += serialized_data_size;
  }

  if (workspace_addr->size != offset) {
    MS_LOG(EXCEPTION) << "Send void data size is not the same as workspace size.";
  }
  message->data = workspace_addr->addr;
  message->size = workspace_addr->size;
}

void SendActor::SerializeCommonMessage(MessageBase *message, const kernel::AddressPtrList &data_list,
                                       const kernel::AddressPtr &workspace_addr) const {
  MS_EXCEPTION_IF_NULL(message);
  MS_EXCEPTION_IF_NULL(workspace_addr);
  MS_EXCEPTION_IF_NULL(workspace_addr->addr);
  size_t total_size = 0;
  total_size =
    std::accumulate(data_list.begin(), data_list.end(), total_size,
                    [](size_t total_size, const kernel::AddressPtr &output) { return total_size + output->size; });

  if (workspace_addr->size != total_size) {
    MS_LOG(EXCEPTION) << "Workspace size should be the same as inputs size. But got " << workspace_addr->size << " and "
                      << total_size;
  }

  RpcDataPtr rpc_data = static_cast<RpcDataPtr>(workspace_addr->addr);
  MS_EXCEPTION_IF_NULL(rpc_data);
  for (size_t i = 0; i < data_list.size(); i++) {
    MS_EXCEPTION_IF_NULL(data_list[i]);
    if (!CopyRpcDataWithOffset(&rpc_data, data_list[i]->addr, data_list[i]->size)) {
      MS_LOG(EXCEPTION) << "Failed to copy data for rpc send input " << i;
    }
  }
  message->data = workspace_addr->addr;
  message->size = workspace_addr->size;
}

}  // namespace runtime
}  // namespace mindspore

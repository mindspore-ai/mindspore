/**
 * Copyright 2023 Huawei Technologies Co., Ltd.
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
#include "plugin/device/ascend/kernel/hccl/mux_send_ascend_kernel.h"
#include "include/common/utils/anfalgo.h"
#include "plugin/device/ascend/hal/hardware/ascend_collective_comm_lib.h"
#include "plugin/device/ascend/hal/hccl_adapter/hccl_adapter.h"

using AscendCollectiveCommLib = mindspore::device::ascend::AscendCollectiveCommLib;
namespace mindspore {
namespace kernel {
MuxSendAscendKernel::MuxSendAscendKernel() {
  if (tcp_client_ == nullptr) {
    tcp_client_ = std::make_unique<distributed::rpc::TCPClient>();
    (void)tcp_client_->Initialize();
  }
}

MuxSendAscendKernel::~MuxSendAscendKernel() {
  if (tcp_client_ != nullptr) {
    tcp_client_->Finalize();
    tcp_client_.reset();
  }
}
bool MuxSendAscendKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                 const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  // Refresh the current destination rank id.
  int dest_rank = -1;
  // If the dest_rank attribute is set to a valid value, then the destination rank is fixed and immutable.
  if (dest_rank_ > -1) {
    dest_rank = dest_rank_;
    // If the dest_rank attribute is not specified, the `src_rank_id_` is used which is set by the prior mux recv
    // kernel.
  } else if (src_rank_id_ > -1) {
    dest_rank = src_rank_id_;
  } else {
    MS_LOG(ERROR) << "Failed to find valid dest rank id.";
    return false;
  }

  // Do the handshake with the mux recv kernel to set the source rank id.
  if (!Handshake(dest_rank)) {
    return false;
  }
  // Do the message sending by calling the hcclsend API.
  MS_EXCEPTION_IF_NULL(stream_ptr);
  auto hccl_result = hccl::HcclAdapter::GetInstance().HcclSend(inputs[0]->addr, hccl_count_, hccl_data_type_list_[0],
                                                               dest_rank, stream_ptr, comm_);
  if (hccl_result != HCCL_SUCCESS) {
    MS_LOG(ERROR) << "HcomSend failed, ret:" << hccl_result;
    return false;
  }
  // Reset the source rank id.
  src_rank_id_ = -1;
  return true;
}

bool MuxSendAscendKernel::Init(const AnfNodePtr &anf_node) {
  cgn_ = std::dynamic_pointer_cast<distributed::cluster::topology::ComputeGraphNode>(
    distributed::cluster::ClusterContext::instance()->node_base());
  MS_EXCEPTION_IF_NULL(cgn_);
  MS_EXCEPTION_IF_NULL(anf_node);
  bool ret = HcclKernel::Init(anf_node);
  if (!ret) {
    return ret;
  }
  auto kernel_node = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(kernel_node);
  auto prim = common::AnfAlgo::GetCNodePrimitive(kernel_node);
  MS_EXCEPTION_IF_NULL(prim);

  src_rank_ = UintToInt(cgn_->rank_id());
  dest_rank_ = GetValue<int64_t>(prim->GetAttr("dest_rank"));

  hccl_kernel_input_shape_list_.clear();
  hccl_kernel_output_shape_list_.clear();
  hccl_data_type_list_.clear();
  if (!HcomUtil::GetKernelInputShape(anf_node, &hccl_kernel_input_shape_list_)) {
    MS_LOG(ERROR) << "GetKernelInputShape fail!";
    return false;
  }
  if (!HcomUtil::GetKernelOutputShape(anf_node, &hccl_kernel_output_shape_list_)) {
    MS_LOG(ERROR) << "GetKernelOutputShape fail!";
    return false;
  }
  if (!HcomUtil::GetHcomDataType(anf_node, &hccl_data_type_list_)) {
    MS_LOG(ERROR) << "GetHcomDataType fail!";
    return false;
  }
  if (!HcomUtil::GetHcomCount(anf_node, hccl_data_type_list_, hccl_kernel_input_shape_list_, &hccl_count_)) {
    MS_LOG(ERROR) << "GetHcomCount fail!";
    return false;
  }
  HcomUtil::GetHcomGroup(NOT_NULL(anf_node), NOT_NULL(&group_));
  // pynative with ranktable also need hccl_comm
  comm_ = AscendCollectiveCommLib::GetInstance().HcclCommunicator(group_);
  if (common::UseHostCollective()) {
    MS_EXCEPTION_IF_NULL(comm_);
    common::AnfAlgo::SetNodeAttr(kAttrComm, MakeValue<int64_t>(reinterpret_cast<int64_t>(comm_)), anf_node);
  }
  anf_node_ = anf_node;
  loop_size_ = hccl_kernel_output_shape_list_.size();
  return true;
}

bool MuxSendAscendKernel::Handshake(const int dest_rank) {
  if (!Connect(dest_rank)) {
    return false;
  }
  std::string server_url = dest_addrs_[dest_rank];

  // Do the handshake.
  std::unique_ptr<MessageBase> message = std::make_unique<MessageBase>();
  MS_EXCEPTION_IF_NULL(message);
  message->name = ACTIVATE;
  message->from = AID("", "");
  message->to = AID("", server_url);
  message->body = std::to_string(src_rank_);

  MessageBase *response = nullptr;
  const size_t interval = 10;
  while (response == nullptr) {
    response = tcp_client_->ReceiveSync(std::move(message));
    if (response != nullptr && response->body != ACTIVATION_OK) {
      return false;
    }
    if (response == nullptr) {
      std::this_thread::sleep_for(std::chrono::nanoseconds(interval));
      MS_LOG(WARNING) << "Retry to negotiate with the mux recv kernerl.";
      continue;
    }
  }
  if (response != nullptr) {
    delete response;
    response = nullptr;
  }
  return true;
}
bool MuxSendAscendKernel::Connect(const int dest_rank) {
  if (dest_addrs_.find(dest_rank) == dest_addrs_.end()) {
    auto server_addr_key = MUX_RECV_RANK_ADDR_PREFIX + std::to_string(dest_rank);
    bool success = false;
    const size_t interval = 3;
    while (!success) {
      auto server_addr_val = cgn_->GetMetadata(server_addr_key);
      if (server_addr_val.length() > 0) {
        dest_addrs_[dest_rank] = server_addr_val;
        success = true;
      } else {
        MS_LOG(WARNING) << "Retry to get the host address of the mux recv gpu kernel for rank: " << dest_rank;
        (void)sleep(interval);
      }
    }
    if (!success) {
      return false;
    }
    // Record the address of the mux recv kernel and connect to it.
    auto server_url = dest_addrs_[dest_rank];
    const size_t retry = 1;
    while (!tcp_client_->IsConnected(server_url)) {
      if (!tcp_client_->Connect(server_url, retry)) {
        MS_LOG(ERROR) << "Failed to connect to server";
      }
      (void)sleep(interval);
    }
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore

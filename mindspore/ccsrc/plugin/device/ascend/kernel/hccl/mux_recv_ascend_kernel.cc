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
#include "plugin/device/ascend/kernel/hccl/mux_recv_ascend_kernel.h"
#include "include/common/utils/anfalgo.h"
#include "plugin/device/ascend/hal/hardware/ascend_collective_comm_lib.h"
#include "plugin/device/ascend/hal/hccl_adapter/hccl_adapter.h"

using AscendCollectiveCommLib = mindspore::device::ascend::AscendCollectiveCommLib;
namespace mindspore {
namespace kernel {
MuxRecvAscendKernel::~MuxRecvAscendKernel() {
  if (tcp_server_ != nullptr) {
    tcp_server_->Finalize();
    tcp_server_.reset();
  }
}
bool MuxRecvAscendKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                 const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  // Refresh the current destination rank id.
  while (idle_.load() || src_rank_ == -1) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1));
    continue;
  }

  std::lock_guard<std::mutex> lock(mtx_);
  if (src_rank_ == -1) {
    MS_LOG(ERROR) << "Invalid source rank id -1.";
    return false;
  }
  // Do the message sending by calling the hcclsend API.
  MS_EXCEPTION_IF_NULL(stream_ptr);
  auto hccl_result = hccl::HcclAdapter::GetInstance().HcclRecv(outputs[0]->addr, hccl_count_, hccl_data_type_list_[0],
                                                               src_rank_, stream_ptr, comm_);
  if (hccl_result != HCCL_SUCCESS) {
    MS_LOG(ERROR) << "HcclRecv failed, ret:" << hccl_result;
    return false;
  }
  // Reset the source rank id.
  src_rank_id_ = src_rank_;
  src_rank_ = -1;
  idle_.store(true);
  return true;
}

bool MuxRecvAscendKernel::Init(const AnfNodePtr &anf_node) {
  // Init the cluster component used for the address synchronization.
  cgn_ = std::dynamic_pointer_cast<distributed::cluster::topology::ComputeGraphNode>(
    distributed::cluster::ClusterContext::instance()->node_base());
  MS_EXCEPTION_IF_NULL(cgn_);
  MS_EXCEPTION_IF_NULL(anf_node);
  InitServer();
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
  if (!HcomUtil::GetHcomCount(anf_node, hccl_data_type_list_, hccl_kernel_output_shape_list_, &hccl_count_)) {
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

void MuxRecvAscendKernel::InitServer() {
  if (tcp_server_ == nullptr) {
    tcp_server_ = std::make_unique<distributed::rpc::TCPServer>();
    MS_EXCEPTION_IF_NULL(tcp_server_);

    if (!tcp_server_->Initialize()) {
      MS_LOG(ERROR) << "Failed to init the tcp server.";
      return;
    }
    tcp_server_->SetMessageHandler([this](MessageBase *const message) -> MessageBase *const {
      // This CAS operation will be success only if the `Launch` operation is ready to run.
      bool expected = true;
      const ssize_t interval = 100000;
      while (true) {
        auto rt = idle_.compare_exchange_strong(expected, false);
        if (rt) {
          break;
        } else {
          expected = true;
          std::this_thread::sleep_for(std::chrono::nanoseconds(interval));
          continue;
        }
      }

      std::lock_guard<std::mutex> lock(mtx_);
      // Update the source rank id for this send/recv communication.
      constexpr auto decimal = 10;
      src_rank_ = static_cast<int>(std::strtol(message->body.c_str(), nullptr, decimal));

      MessageBase *response = new MessageBase();
      response->name = ACTIVATE;
      response->from = AID("", "");
      response->to = AID("", "");
      response->body = ACTIVATION_OK;

      return response;
    });
    RegisterAddr();
  }
}
void MuxRecvAscendKernel::RegisterAddr() {
  auto server_addr_key = MUX_RECV_RANK_ADDR_PREFIX + std::to_string(cgn_->rank_id());
  auto server_addr_val = tcp_server_->GetIP() + ":" + std::to_string(tcp_server_->GetPort());
  const size_t interval = 3;
  bool success = false;
  while (!success) {
    success = cgn_->PutMetadata(server_addr_key, server_addr_val);
    if (!success) {
      MS_LOG(WARNING) << "Retry to register the host address of the mux recv gpu kernel for rank: " << cgn_->rank_id();
      (void)sleep(interval);
    } else {
      MS_LOG(INFO) << "The host address: " + server_addr_val + " of the mux recv gpu kernel " + server_addr_key +
                        " has been successfully registered.";
    }
  }
}
}  // namespace kernel
}  // namespace mindspore

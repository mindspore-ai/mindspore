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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MUX_SEND_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MUX_SEND_GPU_KERNEL_H_

#include <vector>
#include <string>
#include <functional>
#include <utility>
#include <map>
#include <memory>
#include "include/backend/distributed/rpc/tcp/tcp_client.h"
#include "include/backend/distributed/cluster/cluster_context.h"
#include "include/backend/distributed/cluster/topology/compute_graph_node.h"
#include "plugin/device/gpu/kernel/rl/mux_base_gpu_kernel.h"

namespace mindspore {
namespace kernel {
template <typename T>
class MuxSendGpuKernel : public MuxBaseGpuKernel {
 public:
  MuxSendGpuKernel() {
    if (tcp_client_ == nullptr) {
      tcp_client_ = std::make_unique<distributed::rpc::TCPClient>();
      tcp_client_->Initialize();
    }
  }

  ~MuxSendGpuKernel() override {
    if (tcp_client_ != nullptr) {
      tcp_client_->Finalize();
      tcp_client_.reset();
    }
  }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
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

    // Do the message sending by calling the ncclsend API.
    T *input_addr = GetDeviceAddress<T>(inputs, 0);
    (void)Send(input_addr, total_size_ / sizeof(T), nccl_data_type_, dest_rank,
               reinterpret_cast<cudaStream_t>(stream_ptr), group_name_);

    // Reset the source rank id.
    src_rank_id_ = -1;
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    // Init the cluster component used for the address synchronization.
    cgn_ = std::dynamic_pointer_cast<distributed::cluster::topology::ComputeGraphNode>(
      distributed::cluster::ClusterContext::instance()->node_base());
    MS_EXCEPTION_IF_NULL(cgn_);

    auto kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    MS_EXCEPTION_IF_NULL(kernel_node);

    auto prim = common::AnfAlgo::GetCNodePrimitive(kernel_node);
    MS_EXCEPTION_IF_NULL(prim);

    src_rank_ = cgn_->rank_id();
    dest_rank_ = GetValue<int64_t>(prim->GetAttr("dest_rank"));

    nccl_data_type_ = nccl_dtype(AnfAlgo::GetInputDeviceDataType(kernel_node, 0));
    group_name_ = GetAttr<std::string>(kernel_node, kAttrGroup);

    total_size_ = 0;
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    for (size_t i = 0; i < input_num; ++i) {
      auto shape_signed = AnfAlgo::GetInputDeviceShape(kernel_node, i);
      if (IsDynamic(shape_signed)) {
        return true;
      }
      auto input_shape = Convert2SizeTClipNeg(shape_signed);

      is_null_input_ = CHECK_SHAPE_NULL(input_shape, kernel_name, "input");
      if (is_null_input_) {
        InitSizeLists();
        return true;
      }

      size_t data_size = 0;
      auto type = AnfAlgo::GetInputDeviceDataType(kernel_node, i);
      if (type == kNumberTypeFloat32) {
        data_size = sizeof(T);
      }
      size_t input_size = std::accumulate(input_shape.begin(), input_shape.end(), data_size, std::multiplies<size_t>());
      input_size_list_.push_back(input_size);
      // Framework memory allocation ensures memory alignment.
      total_size_ += device::gpu::GPUMemoryAllocator::GetInstance().AlignMemorySize(input_size);
    }
    output_size_list_.push_back(0);

    SelectCollectiveHandle();
    return true;
  }

 protected:
  void InitSizeLists() override {}

 private:
  // Handshake with the mux recv kernel.
  bool Handshake(const int dest_rank) {
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

  // If the address of the destination rank is unknown, try to get from the cluster dynamically.
  bool Connect(const int dest_rank) {
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

  size_t total_size_{0};

  // The rank id of this mux send kernel.
  int src_rank_{-1};

  // The destination rank id which is fixed by the kernel attribute or assigned dynamically.
  int dest_rank_{-1};

  // The address of the mux recv kernels.
  std::map<int64_t, std::string> dest_addrs_;
  bool is_null_input_{false};

  // The tcp client is used to do the handshake with the mux recv kernel.
  std::unique_ptr<distributed::rpc::TCPClient> tcp_client_{nullptr};
  std::shared_ptr<distributed::cluster::topology::ComputeGraphNode> cgn_{nullptr};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MUX_SEND_GPU_KERNEL_H_

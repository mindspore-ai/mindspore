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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MUX_RECV_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MUX_RECV_GPU_KERNEL_H_

#include <vector>
#include <string>
#include <functional>
#include <chrono>
#include <thread>
#include <memory>
#include "include/backend/distributed/rpc/tcp/tcp_server.h"
#include "include/backend/distributed/cluster/cluster_context.h"
#include "include/backend/distributed/cluster/topology/compute_graph_node.h"
#include "plugin/device/gpu/kernel/rl/mux_base_gpu_kernel.h"

namespace mindspore {
namespace kernel {
template <typename T>
class MuxRecvGpuKernel : public MuxBaseGpuKernel {
 public:
  MuxRecvGpuKernel() {}

  ~MuxRecvGpuKernel() override {
    if (tcp_server_ != nullptr) {
      tcp_server_->Finalize();
      tcp_server_.reset();
    }
  }

  bool Launch(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &, const std::vector<AddressPtr> &outputs,
              void *stream_ptr) override {
    // The value of `src_rank_` will not be changed once it's value is valid unless the value of `idle_` is set to true
    // by the `Launch` method.
    while (idle_.load() || src_rank_ == -1) {
      std::this_thread::sleep_for(std::chrono::nanoseconds(1));
      continue;
    }

    std::lock_guard<std::mutex> lock(mtx_);
    if (src_rank_ == -1) {
      MS_LOG(ERROR) << "Invalid source rank id -1.";
      return false;
    }

    // Do the message receiving by calling the ncclrecv API.
    T *output_addr = GetDeviceAddress<T>(outputs, 0);
    (void)Recv(output_addr, total_size_ / sizeof(T), nccl_data_type_, src_rank_,
               reinterpret_cast<cudaStream_t>(stream_ptr), group_name_);

    // Record the source rank id of this send/recv communication for the next send operator of the same training step.
    src_rank_id_ = src_rank_;
    src_rank_ = -1;
    idle_.store(true);

    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    // Init the cluster component used for the address synchronization.
    cgn_ = std::dynamic_pointer_cast<distributed::cluster::topology::ComputeGraphNode>(
      distributed::cluster::ClusterContext::instance()->node_base());
    MS_EXCEPTION_IF_NULL(cgn_);

    InitServer();

    auto prim = common::AnfAlgo::GetCNodePrimitive(kernel_node);
    MS_EXCEPTION_IF_NULL(prim);

    auto kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    MS_EXCEPTION_IF_NULL(kernel_node);

    group_name_ = GetAttr<std::string>(kernel_node, kAttrGroup);
    nccl_data_type_ = nccl_dtype(AnfAlgo::GetOutputDeviceDataType(kernel_node, 0));

    total_size_ = 0;
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    for (size_t i = 0; i < output_num; ++i) {
      auto shape_signed = common::AnfAlgo::GetOutputInferShape(kernel_node, i);
      if (IsDynamic(shape_signed)) {
        return true;
      }
      auto output_shape = Convert2SizeTClipNeg(shape_signed);
      is_null_input_ = CHECK_SHAPE_NULL(output_shape, kernel_name, "output");
      if (is_null_input_) {
        InitSizeLists();
        return true;
      }

      size_t data_size = sizeof(T);
      size_t output_size =
        std::accumulate(output_shape.begin(), output_shape.end(), data_size, std::multiplies<size_t>());
      output_size_list_.push_back(output_size);
      // Framework memory allocation ensures memory alignment.
      total_size_ += device::gpu::GPUMemoryAllocator::GetInstance().AlignMemorySize(output_size);
    }

    SelectCollectiveHandle();

    return true;
  }

 protected:
  void InitSizeLists() override {}

 private:
  void InitServer() {
    // Create the tcp server used for the handshake between the mux send/recv kernels.
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
        src_rank_ = atoi(message->body.c_str());

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

  void RegisterAddr() {
    // Register the address of this mux recv kernel.
    auto server_addr_key = MUX_RECV_RANK_ADDR_PREFIX + std::to_string(cgn_->rank_id());
    auto server_addr_val = tcp_server_->GetIP() + ":" + std::to_string(tcp_server_->GetPort());
    const size_t interval = 3;
    bool success = false;
    while (!success) {
      success = cgn_->PutMetadata(server_addr_key, server_addr_val);
      if (!success) {
        MS_LOG(WARNING) << "Retry to register the host address of the mux recv gpu kernel for rank: "
                        << cgn_->rank_id();
        (void)sleep(interval);
      } else {
        MS_LOG(INFO) << "The host address: " + server_addr_val + " of the mux recv gpu kernel " + server_addr_key +
                          " has been successfully registered.";
      }
    }
  }

  size_t total_size_{0};

  // The source rank id of the mux send gpu kernel which is assigned in the message handler of the tcp server
  // dynamically.
  int src_rank_{-1};
  bool is_null_input_{false};

  std::mutex mtx_;
  // Indicates if this mux recv kernel is ready to launch.
  std::atomic<bool> idle_{true};

  // This server is used to do the handshake with the mux send kernel.
  std::unique_ptr<distributed::rpc::TCPServer> tcp_server_;
  std::shared_ptr<distributed::cluster::topology::ComputeGraphNode> cgn_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MUX_RECV_GPU_KERNEL_H_

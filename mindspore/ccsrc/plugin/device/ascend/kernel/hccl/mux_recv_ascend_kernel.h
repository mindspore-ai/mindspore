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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_HCCL_MUX_RECV_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_HCCL_MUX_RECV_KERNEL_H_

#include <vector>
#include <string>
#include <functional>
#include <chrono>
#include <thread>
#include <memory>
#include "plugin/device/ascend/kernel/hccl/mux_base_ascend_kernel.h"

#include "include/backend/distributed/rpc/tcp/tcp_server.h"
#include "include/backend/distributed/cluster/cluster_context.h"
#include "include/backend/distributed/cluster/topology/compute_graph_node.h"

namespace mindspore {
namespace kernel {
class MuxRecvAscendKernel : public MuxBaseAscendKernel {
 public:
  MuxRecvAscendKernel() = default;
  ~MuxRecvAscendKernel() override;
  bool Init(const AnfNodePtr &anf_node) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override;

 private:
  void InitServer();
  void RegisterAddr();
  int src_rank_{-1};
  bool is_null_input_{false};
  std::mutex mtx_;
  std::atomic<bool> idle_{true};
  std::unique_ptr<distributed::rpc::TCPServer> tcp_server_;
  std::shared_ptr<distributed::cluster::topology::ComputeGraphNode> cgn_;
};
MS_HCCL_REG_KERNEL(MuxReceive, MuxRecvAscendKernel);
}  // namespace kernel
}  // namespace mindspore
#endif

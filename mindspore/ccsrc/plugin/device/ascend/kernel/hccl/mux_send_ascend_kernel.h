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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_HCCL_MUX_SEND_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_HCCL_MUX_SEND_KERNEL_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "plugin/device/ascend/kernel/hccl/mux_base_ascend_kernel.h"

#include "include/backend/distributed/rpc/tcp/tcp_client.h"
#include "include/backend/distributed/cluster/cluster_context.h"
#include "include/backend/distributed/cluster/topology/compute_graph_node.h"

namespace mindspore {
namespace kernel {
class MuxSendAscendKernel : public MuxBaseAscendKernel {
 public:
  MuxSendAscendKernel();
  ~MuxSendAscendKernel() override;
  bool Init(const AnfNodePtr &anf_node) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override;

 private:
  bool Connect(const int dest_rank);
  bool Handshake(const int dest_rank);
  int src_rank_{-1};
  int dest_rank_{-1};
  std::map<int64_t, std::string> dest_addrs_;
  bool is_null_input_{false};
  std::unique_ptr<distributed::rpc::TCPClient> tcp_client_{nullptr};
  std::shared_ptr<distributed::cluster::topology::ComputeGraphNode> cgn_{nullptr};
};
MS_HCCL_REG_KERNEL(MuxSend, MuxSendAscendKernel);
}  // namespace kernel
}  // namespace mindspore
#endif

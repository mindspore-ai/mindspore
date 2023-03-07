/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_GPU_STREAM_ASSIGN_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_GPU_STREAM_ASSIGN_H_

#include <vector>
#include <string>
#include <memory>
#include "include/backend/kernel_graph.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace device {
namespace gpu {
enum StreamSwitchType { kAllReduceStreamSwitch, kStreamSwitchInvalidType = 255 };
struct SendRecvPair {
  StreamSwitchType stream_switch_type;
  CNodePtr mock_send_node;
  CNodePtr mock_recv_node;
  size_t send_node_offset;
  size_t recv_node_offset;
};
struct StreamSwitchNode {
  size_t offset;
  CNodePtr cnode;
  bool operator<(const StreamSwitchNode &n) const {
    if (offset < n.offset) {
      return true;
    } else if (offset == n.offset) {
      return !(common::AnfAlgo::GetCNodeName(cnode) == kRecvOpName &&
               common::AnfAlgo::GetCNodeName(n.cnode) == kSendOpName);
    } else {
      return false;
    }
  }
};
void AssignGpuStream(const std::shared_ptr<session::KernelGraph> &kernel_graph);
void AssignDefaultGpuStream(const std::shared_ptr<session::KernelGraph> &kernel_graph);
bool FindAllReduceStreamSwitchPos(const std::shared_ptr<session::KernelGraph> &kernel_graph,
                                  std::vector<SendRecvPair> *send_recv_pairs);
// Find Send node position according to "mock" recv node.
// "mock" recv node is a gpu kernel node after a real Recv node, e.g. AllReduce node.
std::vector<CNodePtr>::iterator FindSendNodePos(std::vector<CNodePtr>::iterator begin,
                                                std::vector<CNodePtr>::iterator end, const CNodePtr mock_recv_node,
                                                StreamSwitchType stream_switch_type);
// Find Recv node position according to "mock" send node.
// "mock" send node is a gpu kernel node before a real send node, e.g. AllReduce node.
std::vector<CNodePtr>::iterator FindRecvNodePos(std::vector<CNodePtr>::iterator begin,
                                                std::vector<CNodePtr>::iterator end, const CNodePtr mock_send_node,
                                                StreamSwitchType stream_switch_type);
void InsertStreamSwitchNode(const std::shared_ptr<session::KernelGraph> &kernel_graph,
                            const std::vector<SendRecvPair> &send_recv_pairs);
bool GenSendRecvCNodesForAllReduce(const std::shared_ptr<session::KernelGraph> &kernel_graph,
                                   const CNodePtr &mock_send_node, const CNodePtr &mock_recv_node, CNodePtr *send_node,
                                   CNodePtr *recv_node);
CNodePtr CreateStreamSwitchNode(const std::shared_ptr<session::KernelGraph> &kernel_graph, const std::string &name);

// Cache the allreduce kernel to send/recv nodes in the kernel graph.
void CacheSendRecvCNodesForAllReduce(const std::shared_ptr<session::KernelGraph> &kernel_graph,
                                     const CNodePtr &mock_send_node, const CNodePtr &mock_recv_node,
                                     const CNodePtr &send_node, const CNodePtr &recv_node);
}  // namespace gpu
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_GPU_STREAM_ASSIGN_H_

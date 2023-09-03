/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_INTER_PROCESS_EDGE_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_INTER_PROCESS_EDGE_H_

#include <string>
#include <vector>
#include <memory>
#include "include/backend/distributed/constants.h"

namespace mindspore {
namespace distributed {
// Inter-process communication edges have different types depending on input and user node types.
// For example a connection in the graph like: MatMul->Add, the inter-process edge type is simply tensor transmission.
// But a connection like AssignAdd->UpdateState, its a controlling edge and we just send a tensor with shape [1] to
// represent this semantics.
enum class EdgeType {
  kDataTrans = 0,   // Simply sends one node's output to the remote.
  kSideEffectSync,  // Synchronize parameters after node with side-effect is executed.
  kControl,         // Sends a dummy tensor with shape [1] to the remote.
};

// InterProcessEdge is a base class that denotes the communication edge info between MindSpore processes.
// One edge could have multiple times of collective or p2p communication semantics to complete data transmission.
class InterProcessEdge {
 public:
  InterProcessEdge() = default;
  virtual ~InterProcessEdge() = default;

 protected:
  virtual std::string ToString() const = 0;

  // Meaning of this communication edge. Please refer to enum class 'EdgeType' for detailed description.
  EdgeType edge_type_;
};

using InterProcessEdgePtr = std::shared_ptr<InterProcessEdge>;
using InterProcessEdgePtrList = std::vector<InterProcessEdgePtr>;
}  // namespace distributed
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DISTRIBUTED_INTER_PROCESS_EDGE_H_

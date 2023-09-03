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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_P2P_EDGE_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_P2P_EDGE_H_

#include <string>
#include <vector>
#include "include/backend/distributed/mode/inter_prcess_edge.h"

namespace mindspore {
namespace distributed {
// Class P2PEdge denotes a point to point edge between two processes. It could have multiple Send/Recv pairs.
// For example, three MatMul nodes execute on process 1 and three Relu nodes, that take these MatMul nodes as inputs,
// execute on process 2. After graph partitioning and adding P2PEdge, the DAG should be like:

// Process 1:   MatMul1        MatMul2        MatMul3
//                ↓               ↓              ↓
// Process 1:    Send            Send           Send
//                ↓               ↓              ↓
// Process 2:    Recv            Recv           Recv
//                ↓               ↓              ↓
// Process 2:    Relu1           Relu2          Relu3

// After communication operator fusion operation, the DAG could be like:
// Process 1:   MatMul1        MatMul2        MatMul3
//                ↘               ↓              ↙
// Process 1:                    Send
//                                ↓
// Process 2:                    Recv
//                ↙               ↓              ↘
// Process 2:    Relu1           Relu2          Relu3
class P2PEdge : public InterProcessEdge {
 public:
  P2PEdge();
  ~P2PEdge() override;

 protected:
  std::string ToString() const override;
};
}  // namespace distributed
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DISTRIBUTED_P2P_EDGE_H_

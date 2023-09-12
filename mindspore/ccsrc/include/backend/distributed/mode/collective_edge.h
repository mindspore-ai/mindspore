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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_COLLECTIVE_EDGE_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_COLLECTIVE_EDGE_H_

#include <string>
#include <vector>
#include "include/backend/distributed/mode/inter_prcess_edge.h"

namespace mindspore {
namespace distributed {
// In some cases, P2P operators are not efficient enough to transfer data between processes, like Gather(all-to-one) or
// Scatter(on-to-all).
// So we abstract CollectiveEdge, a base class that denotes collective communication between multiple processes.
// Subclasses could be derived from it to denote different types of collective semantics.
class CollectiveEdge : public InterProcessEdge {
 public:
  CollectiveEdge();
  ~CollectiveEdge() override;

 protected:
  std::string ToString() const override;
};
}  // namespace distributed
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DISTRIBUTED_COLLECTIVE_EDGE_H_

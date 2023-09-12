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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_CROSS_MESH_EDGE_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_CROSS_MESH_EDGE_H_

#include <string>
#include <vector>
#include <memory>
#include "include/backend/distributed/constants.h"
#include "include/backend/distributed/mode/inter_process_edge.h"

namespace mindspore {
namespace distributed {
// In MPMD mode, communication edges are always involved with different device meshes. For example: 4 devices-> 8
// devices. In this case, we consider CrossMeshEdge to describe the communication pattern between device meshes. It
// could consists of multiple InterProcessEdge objects.
class CrossMeshEdge {
 public:
  CrossMeshEdge() = default;
  virtual ~CrossMeshEdge() = default;

 private:
  InterProcessEdgePtrList edges_;
};

using CrossMeshEdgePtr = std::shared_ptr<CrossMeshEdge>;
using CrossMeshEdgePtrList = std::vector<CrossMeshEdgePtr>;
}  // namespace distributed
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DISTRIBUTED_CROSS_MESH_EDGE_H_

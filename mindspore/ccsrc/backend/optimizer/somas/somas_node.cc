/**
 * Copyright 2020 Huawei Technologies Co., Ltd

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#include "backend/optimizer/somas/somas_node.h"
#include <algorithm>

namespace mindspore {
namespace somas {
void SomasNode::ComputeAncestorNodes() {
  // Fast algorithm: assuming nodes execute this function in the received topological order
  int64_t thisId = this->GetStream()->GetId();

  for (SomasNodePtr node : ancestor_nodes_) {
    int64_t ancestorId = node->GetStream()->GetId();
    // Map Improvement for max_ancestor_order
    if (thisId != ancestorId) {
      this->anc_stream_max_order_[ancestorId] = std::max(this->anc_stream_max_order_[ancestorId], node->GetId());
    }
    for (SomasStreamPtr stream : node->GetStream()->ancestor_streams_) {
      this->anc_stream_max_order_[stream->GetId()] =
        std::max(this->anc_stream_max_order_[stream->GetId()], node->anc_stream_max_order_[stream->GetId()]);
    }
  }
}

void SomasNode::PresetAncestorStreams(const std::vector<SomasStreamPtr> stream_vector) {
  for (SomasStreamPtr stream : stream_vector) {
    anc_stream_max_order_[stream->GetId()] = 0;
  }
}
}  // namespace somas
}  // namespace mindspore

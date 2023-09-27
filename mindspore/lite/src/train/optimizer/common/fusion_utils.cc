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
#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include "src/common/log_util.h"
#include "src/train/optimizer/common/fusion_utils.h"

namespace mindspore {
namespace opt {
STATUS GetMatchNodeIndex(schema::MetaGraphT *graph,
                         const std::unordered_map<std::string, std::shared_ptr<lite::Path>> &matched_path,
                         const std::string &node_name, size_t *node_index) {
  auto node_path_iter = matched_path.find(node_name);
  MS_CHECK_TRUE_MSG(node_path_iter != matched_path.end(), RET_ERROR, "cannot find node_path");
  const auto &node_path = node_path_iter->second;
  MS_CHECK_TRUE_MSG(node_path != nullptr, RET_NULL_PTR, "node_path is empty");
  *node_index = node_path->nodeIdx;
  MS_CHECK_TRUE_MSG(*node_index < graph->nodes.size(), RET_ERROR, "node_index is out of range");
  return RET_OK;
}

bool IsMultiOutputNode(schema::MetaGraphT *graph, size_t out_node_index) {
  uint32_t count = 0;
  for (auto &node : graph->nodes) {
    if (std::find(node->inputIndex.begin(), node->inputIndex.end(), out_node_index) != node->inputIndex.end()) {
      count++;
    }
    if (count > 1) {
      return true;
    }
  }
  return false;
}
}  // namespace opt
}  // namespace mindspore

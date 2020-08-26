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

#include <fstream>
#include <sstream>
#include <utility>
#include "src/common/graph_util.h"
#include "src/common/utils.h"
#include "utils/log_adapter.h"
#include "include/errorcode.h"

namespace mindspore {
namespace lite {
std::vector<size_t> GetGraphInputNodes(const schema::MetaGraph *meta_graph) {
  MS_ASSERT(nullptr != meta_graph);
  std::vector<size_t> ret;
  for (auto graph_in_index : *(meta_graph->inputIndex())) {
    for (size_t j = 0; j < meta_graph->nodes()->size(); j++) {
      auto *cNode = meta_graph->nodes()->GetAs<schema::CNode>(j);
      MS_ASSERT(nullptr != cNode);
      MS_ASSERT(nullptr != cNode->inputIndex());
      if (std::any_of(cNode->inputIndex()->begin(), cNode->inputIndex()->end(),
                      [&](const uint32_t &node_in_index) { return node_in_index == graph_in_index; })) {
        if (!IsContain<size_t>(ret, j)) {
          ret.emplace_back(j);
        }
      }
    }
  }
  return ret;
}

std::vector<size_t> GetGraphOutputNodes(const schema::MetaGraph *meta_graph) {
  MS_ASSERT(nullptr != meta_graph);
  std::vector<size_t> ret;
  for (auto graph_out_index : *(meta_graph->outputIndex())) {
    for (size_t j = 0; j < meta_graph->nodes()->size(); j++) {
      auto *cNode = meta_graph->nodes()->GetAs<schema::CNode>(j);
      MS_ASSERT(nullptr != cNode);
      MS_ASSERT(nullptr != cNode->outputIndex());
      if (std::any_of(cNode->outputIndex()->begin(), cNode->outputIndex()->end(),
                      [&](const uint32_t &node_out_index) { return node_out_index == graph_out_index; })) {
        if (!IsContain<size_t>(ret, j)) {
          ret.emplace_back(j);
        }
      }
    }
  }
  return ret;
}
}  // namespace lite
}  // namespace mindspore

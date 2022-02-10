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
#include <algorithm>
#include <memory>
#include "src/common/log.h"
#include "src/train/graph_dropout.h"
#include "tools/converter/optimizer.h"
#include "tools/converter/legacy_optimizer/graph/dropout_node_remove_pass.h"
#include "tools/converter/legacy_optimizer/graph/isolated_node_remove_pass.h"
#include "tools/converter/legacy_optimizer/graph/subgraph_node_pass.h"

namespace mindspore {
namespace lite {
std::vector<schema::CNodeT *> GetGraphNodes(const schema::MetaGraphT &graph_defT) {
  std::vector<schema::CNodeT *> old_nodes{};
  old_nodes.resize(graph_defT.nodes.size());
  std::transform(graph_defT.nodes.begin(), graph_defT.nodes.end(), old_nodes.begin(),
                 [](const std::unique_ptr<schema::CNodeT> &node) { return node.get(); });
  return old_nodes;
}

STATUS GraphDropout::Run(schema::MetaGraphT *graph) {
  if (graph == nullptr) {
    MS_LOG(ERROR) << "graph is nullptr.";
    return RET_ERROR;
  }
  Optimizer dropout_optimizer;
  auto old_nodes = GetGraphNodes(*graph);
  dropout_optimizer.AddPass(new (std::nothrow) DropoutNodeRemovePass());
  dropout_optimizer.AddPass(new (std::nothrow) IsolatedNodeRemovePass());
  dropout_optimizer.AddPass(new (std::nothrow) SubgraphNodePass(old_nodes));
  auto status = dropout_optimizer.Run(graph);
  if (status != RET_OK && status != RET_NO_CHANGE) {
    MS_LOG(ERROR) << "graph fusion failed.";
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore

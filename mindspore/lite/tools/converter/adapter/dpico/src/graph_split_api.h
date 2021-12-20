/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef DPICO_SRC_GRAPH_SPLITTER_H_
#define DPICO_SRC_GRAPH_SPLITTER_H_

#include <memory>
#include <map>
#include <vector>
#include "api/ir/func_graph_manager.h"

namespace mindspore {
namespace dpico {
struct Subgraph;
struct GraphSplitInfo;

int GraphSplit(const std::vector<api::FuncGraphPtr> &func_graphs, GraphSplitInfo *graph_split_info);
AnfNodePtrList GetSubgraphInputs(const Subgraph &subgraph, const api::FuncGraphPtr &func_graph);
AnfNodePtrList GetSubgraphOutputs(const Subgraph &subgraph, const api::FuncGraphManagerPtr &manager);
int FillSubgraphOutputsFormat(Subgraph *subgraph, const api::FuncGraphPtr &func_graph);
}  // namespace dpico
}  // namespace mindspore
#endif  // DPICO_SRC_GRAPH_SPLITTER_H_

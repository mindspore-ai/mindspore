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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_SRC_GRAPH_SPLITT_API_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_SRC_GRAPH_SPLITT_API_H_

#include <memory>
#include <map>
#include <vector>
#include "mindapi/ir/common.h"

namespace mindspore {
namespace dpico {
struct Subgraph;
struct GraphSplitInfo;

int GraphSplit(const std::vector<api::FuncGraphPtr> &func_graphs, GraphSplitInfo *graph_split_info);
api::AnfNodePtrList GetSubgraphInputs(const Subgraph &subgraph, const api::FuncGraphPtr &func_graph);
api::AnfNodePtrList GetSubgraphOutputs(const Subgraph &subgraph, const api::FuncGraphManagerPtr &manager);
int FillSubgraphOutputsFormat(Subgraph *subgraph, const api::FuncGraphPtr &func_graph);
}  // namespace dpico
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_SRC_GRAPH_SPLITT_API_H_

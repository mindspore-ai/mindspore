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
#ifndef MINDSPORE_CCSRC_FRONTEND_EXPANDER_PACK_PACKFUNC_GRAD_H_
#define MINDSPORE_CCSRC_FRONTEND_EXPANDER_PACK_PACKFUNC_GRAD_H_

#include <string>
#include <memory>
#include <utility>
#include <vector>
#include "ir/anf.h"

namespace mindspore {
namespace expander {

// store information needed for Graph grad
struct GraphGradInfo {
  FuncGraphPtr ori_graph;
  abstract::AbstractBasePtr ori_output_abs;
  FuncGraphPtr graph_set_forward;
  std::vector<std::pair<ValueNodePtr, size_t>> forward_vnodes;
  std::vector<size_t> forward_node_output_index;
  std::vector<size_t> forward_node_output_unused;
  FuncGraphPtr graph;
  size_t added_output_size{0};
  int64_t graph_id{0};
};

using GraphGradInfoPtr = std::shared_ptr<GraphGradInfo>;

GraphGradInfoPtr GenGraphGradInfo(const FuncGraphPtr &func_graph);

const GraphGradInfoPtr &GetGraphGradInfo(int64_t graph_id);

void ClearGraphGradInfoCache();

const mindspore::HashSet<size_t> GetUnusedInputs(const FuncGraphPtr &func_graph);

ValuePtrList GetForwardNodesValue(const ValuePtr &out_value, const expander::GraphGradInfoPtr &graph_grad_info);
}  // namespace expander
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_EXPANDER_PACK_PACKFUNC_GRAD_H_

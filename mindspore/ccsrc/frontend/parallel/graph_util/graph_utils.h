/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_GRAPH_UTIL_GRAPH_UTILS_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_GRAPH_UTIL_GRAPH_UTILS_H_
#include <set>
#include <vector>
#include <string>
#include "mindspore/core/base/base.h"
#include "mindspore/core/ir/anf.h"
#include "frontend/parallel/status.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"
#include "frontend/parallel/ops_info/operator_info.h"

namespace mindspore::parallel {
std::set<FuncGraphPtr> FindForwardGraphByRootNodes(const AnfNodeSet &root_all_nodes);
AnfNodePtr GetAccuGrad(const std::vector<AnfNodePtr> &parameters, const std::string &weight_name);
std::vector<AnfNodePtr> CreateInput(const Operator &op, const AnfNodePtr &node, const std::string &instance_name,
                                    const TensorRedistribution &tensor_redistribution);
std::vector<AnfNodePtr> CreateMirrorInput(const FuncGraphPtr &root, const Operator &op, const AnfNodePtr &node,
                                          const std::string &instance_name, const std::string &weight_name);
void InsertNode(const Operator &op, const CNodePtr &node, size_t index, const AnfNodePtr &pre_node,
                const FuncGraphPtr &func_graph, const std::string &instance_name, const std::string &param_name = "",
                const FuncGraphPtr &root = nullptr,
                const TensorRedistribution &tensor_redistribution = TensorRedistribution());
}  // namespace mindspore::parallel

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_GRAPH_UTIL_GRAPH_UTILS_H_

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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_GRAPH_UTIL_GRAPH_INFO_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_GRAPH_UTIL_GRAPH_INFO_H_

#include <string>
#include <vector>

#include "ir/anf.h"

namespace mindspore {
namespace parallel {
std::vector<PrimitivePtr> FindPrimtive(const FuncGraphPtr &graph, const std::string &name);
void DumpGraph(const FuncGraphPtr &root, const std::string &name);
bool GetLoopIndexFromCNode(const CNodePtr &cnode, size_t *loop_index);
void SetOpsNumToExecutor(size_t num_ops);
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_GRAPH_UTIL_GRAPH_INFO_H_

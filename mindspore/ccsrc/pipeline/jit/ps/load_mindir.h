/**
 * Copyright 2019-2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PIPELINE_JIT_LOAD_MINDIR_H_
#define MINDSPORE_CCSRC_PIPELINE_JIT_LOAD_MINDIR_H_

#include <vector>
#include <set>

#include "base/base.h"
#include "ir/manager.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "pipeline/jit/ps/resource.h"

namespace mindspore {
namespace pipeline {
bool ModifyGraphGeneratedByMindIR(const ResourcePtr &resource);
void ModifyOneFuncGraph(const FuncGraphPtr &func_graph, std::set<FuncGraphPtr> *func_graph_set,
                        std::set<FuncGraphPtr> *func_graph_modified);
void ModifyOneCNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode);
std::vector<AnfNodePtr> ArgsNeededToConvert(const PrimitivePtr &prim);
bool InferMindIR(const ResourcePtr &resource);
}  // namespace pipeline
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_JIT_LOAD_MINDIR_H_

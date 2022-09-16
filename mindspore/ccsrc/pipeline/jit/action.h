/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PIPELINE_JIT_ACTION_H_
#define MINDSPORE_CCSRC_PIPELINE_JIT_ACTION_H_

#include <vector>
#include <functional>
#include <utility>
#include <string>
#include "pipeline/jit/resource.h"
#include "backend/graph_compiler/segment_runner.h"
#include "backend/graph_compiler/backend.h"

namespace mindspore {
namespace pipeline {
using ActionItem = std::pair<std::string, std::function<bool(ResourcePtr)>>;

bool ParseAction(const ResourcePtr &resource);
bool SymbolResolveAction(const ResourcePtr &resource);
bool AutoMonadAction(const ResourcePtr &resource);
bool AbstractSpecializeAction(const ResourcePtr &resource);
bool GeOptimizeAction(const ResourcePtr &resource);
bool VmOptimizeAction(const ResourcePtr &resource);
bool TaskEmitAction(const ResourcePtr &resource);
bool ExecuteAction(const ResourcePtr &resource);
bool StartPSSchedulerAction(const ResourcePtr &resource);
bool DistributedSplitAction(const ResourcePtr &resource);

std::vector<ActionItem> GePipeline();
std::vector<ActionItem> VmPipeline(const ResourcePtr &resource);
std::vector<ActionItem> MindIRPipeline();
std::vector<ActionItem> PSchedulerPipeline(const ResourcePtr &resource);
abstract::AnalysisResult AbstractAnalyze(const ResourcePtr &resource, const FuncGraphPtr &func_graph,
                                         const abstract::AbstractBasePtrList &args_abs, bool clear = false);
FuncGraphPtr ProgramSpecialize(const ResourcePtr &resource, const FuncGraphPtr &func_graph,
                               const abstract::AnalysisContextPtr &context);
FuncGraphPtr Renormalize(const ResourcePtr &resource, const FuncGraphPtr &func_graph,
                         const abstract::AbstractBasePtrList &args_abs);
void SetRunMode(const FuncGraphPtr &func_graph, compile::Backend *backend_ptr);
}  // namespace pipeline
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_JIT_ACTION_H_

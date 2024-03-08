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
#include "pipeline/jit/ps/resource.h"
#include "backend/graph_compiler/segment_runner.h"
#include "backend/graph_compiler/backend.h"

namespace mindspore {
namespace pipeline {
using ActionItem = std::pair<std::string, std::function<bool(ResourcePtr)>>;

bool ParseAction(const ResourcePtr &resource);
bool SymbolResolveAction(const ResourcePtr &resource);
bool AutoMonadAction(const ResourcePtr &resource);
bool PreCConvAction(const ResourcePtr &resource);
bool AbstractSpecializeAction(const ResourcePtr &resource);
bool VmOptimizeAction(const ResourcePtr &resource);
bool TaskEmitAction(const ResourcePtr &resource);
bool ExecuteAction(const ResourcePtr &resource);
#if defined(__linux__) && defined(WITH_BACKEND)
bool StartPSSchedulerAction(const ResourcePtr &resource);
bool DistributedSplitAction(const ResourcePtr &resource);
#endif

std::vector<ActionItem> VmPipeline(const ResourcePtr &resource, bool trace_flag = false);
std::vector<ActionItem> MindIRPipeline();
#if defined(__linux__) && defined(WITH_BACKEND)
std::vector<ActionItem> PSchedulerPipeline(const ResourcePtr &resource);
#endif
abstract::AnalysisResult AbstractAnalyze(const abstract::AnalysisEnginePtr &engine, const FuncGraphPtr &func_graph,
                                         const abstract::AbstractBasePtrList &args_abs, bool is_load_resoure,
                                         bool clear = false);
abstract::AnalysisResult AbstractAnalyze(const ValuePtr &value, const abstract::AbstractBasePtrList &args_abs,
                                         bool clear = false);
FuncGraphPtr ProgramSpecialize(const abstract::AnalysisEnginePtr &engine, const FuncGraphPtr &func_graph,
                               const abstract::AnalysisContextPtr &context);
FuncGraphPtr Renormalize(const ResourcePtr &resource, const FuncGraphPtr &func_graph,
                         const abstract::AbstractBasePtrList &args_abs);
FuncGraphPtr Renormalize(const ValuePtr &value, const abstract::AbstractBasePtrList &args_abs);
void SetRunMode(const FuncGraphPtr &func_graph, compile::Backend *backend_ptr, std::string *kbk_reason = nullptr);
AbstractBasePtr GetDefaultValueAbstract(const ParameterPtr &param);
}  // namespace pipeline
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_JIT_ACTION_H_

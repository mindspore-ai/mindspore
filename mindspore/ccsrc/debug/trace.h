/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_DEBUG_TRACE_H_
#define MINDSPORE_CCSRC_DEBUG_TRACE_H_

#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include <stack>
#include <deque>

#include "utils/trace_base.h"
#include "utils/info.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "pipeline/jit/static_analysis/static_analysis.h"
#include "utils/any.h"

namespace mindspore {
namespace trace {
using TraceGraphEvalStack = std::deque<std::pair<abstract::AnalysisContextPtr, abstract::AnfNodeConfigPtr>>;
using TraceCNodeEvalStack = std::vector<abstract::AnfNodeConfigPtr>;
DebugInfoPtr GetSourceCodeDebugInfo(const DebugInfoPtr &info);
void TraceGraphEval();
void GetEvalStackInfo(std::ostringstream &oss);
void TraceGraphEvalEnter(const abstract::AnalysisContextPtr &context, const abstract::AnfNodeConfigPtr &node);
void TraceGraphEvalLeave(const abstract::AnalysisContextPtr &context);
void TraceGraphEvalStackPrepare(const TraceGraphEvalStack &graphEvals);
void TraceEvalCNodeStackPrepare(const TraceCNodeEvalStack &cnodeEvals);
void TraceEvalCNodeEnter(const abstract::AnfNodeConfigPtr &node_cfg);
void TraceEvalCNodeLeave();
TraceCNodeEvalStack &GetCNodeDebugStack();
TraceGraphEvalStack &GetCurrentGraphEvalStack();
void GetTraceStackInfo(std::ostringstream &oss);
std::string GetAbstractStr(const abstract::AbstractBasePtr &abs);
void ClearTraceStack();
}  // namespace trace
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_DEBUG_TRACE_H_

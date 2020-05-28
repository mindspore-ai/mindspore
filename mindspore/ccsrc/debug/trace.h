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
#ifndef MINDSPORE_CCSRC_DEBUG_TRACE_H_
#define MINDSPORE_CCSRC_DEBUG_TRACE_H_

#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include <stack>

#include "ir/anf.h"
#include "utils/any.h"
#include "ir/func_graph.h"
#include "debug/info.h"
#include "pipeline/static_analysis/static_analysis.h"

namespace mindspore {
namespace trace {
std::string GetDebugInfo(const DebugInfoPtr &info, SourceLineTip tip = kSourceLineTipNextLine);
std::string GetDebugInfo(const DebugInfoPtr &info, const std::string &prefix,
                         SourceLineTip tip = kSourceLineTipNextLine);
DebugInfoPtr GetSourceCodeDebugInfo(const DebugInfoPtr &info);
void TraceGraphEval();
void GetEvalStackInfo(std::ostringstream &oss);
void TraceGraphEvalEnter(const abstract::EvaluatorPtr &eval, const abstract::AnfNodeConfigPtr &node);
void TraceGraphEvalLeave(const abstract::EvaluatorPtr &eval);
void TraceEvalCNodeEnter(const abstract::AnfNodeConfigPtr &node_cfg);
void TraceEvalCNodeLeave();
std::vector<abstract::AnfNodeConfigPtr> &GetCNodeDebugStack();
std::stack<std::pair<abstract::EvaluatorPtr, abstract::AnfNodeConfigPtr>> &GetCurrenGraphInferStack();
std::string GetAbstractStr(const abstract::AbstractBasePtr &abs);
void ClearTraceStack();
}  // namespace trace
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_DEBUG_TRACE_H_

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

#include "debug/trace.h"

#include <iostream>
#include <fstream>
#include <map>
#include <unordered_map>
#include <vector>
#include <string>
#include <sstream>
#include <utility>
#include <stack>
#include <algorithm>

#include "ir/meta_func_graph.h"
#include "utils/graph_utils.h"
#include "operator/composite/composite.h"
#include "ir/meta_tensor.h"
#include "debug/anf_ir_utils.h"
#include "pipeline/static_analysis/evaluator.h"

namespace mindspore {
// namespace to support debug trace infomation
namespace trace {
std::string GetAbstractStr(const abstract::AbstractBasePtr &abs) {
  if (abs == nullptr) {
    return "Null Abstract";
  }
  auto shape = abs->BuildShape()->cast<abstract::ShapePtr>();
  TypePtr type = abs->BuildType();
  std::ostringstream oss;
  if ((shape != nullptr) && (type != nullptr)) {
    oss << type->DumpText() << shape->DumpText();
  } else if (type != nullptr) {
    oss << type->DumpText();
  } else {
    oss << "Undefined";
  }
  return oss.str();
}

std::vector<DebugInfoPtr> GetSourceCodeDebugInfoVec(DebugInfoPtr debug_info) {
  std::vector<DebugInfoPtr> debug_with_loc_vec;
  while (debug_info != nullptr) {
    if (debug_info->location() != nullptr) {
      debug_with_loc_vec.push_back(debug_info);
    }
    if (debug_info->trace_info() != nullptr) {
      debug_info = debug_info->trace_info()->debug_info();
    } else {
      break;
    }
  }
  return debug_with_loc_vec;
}

DebugInfoPtr GetSourceCodeDebugInfo(const DebugInfoPtr &info) {
  auto debug_with_loc_vec = GetSourceCodeDebugInfoVec(info);
  if (debug_with_loc_vec.size() > 0) {
    return debug_with_loc_vec[0];
  } else {
    return info;
  }
}

std::string GetDebugInfo(const DebugInfoPtr &info, SourceLineTip tip) {
  if (info == nullptr) {
    return "";
  }
  auto src_info = GetSourceCodeDebugInfo(info);
  if (src_info->location() != nullptr) {
    return src_info->location()->ToString(tip);
  }
  return "";
}

// a trace info identifies a node transform, so we can trace the node transform through
// a link of trace info and debug info
std::string GetInfoWithAction(const std::vector<DebugInfoPtr> &info_vec, SourceLineTip tip) {
  if (info_vec.size() < 1) {
    return "";
  }
  if (info_vec.size() == 1) {
    return info_vec[0]->location()->ToString(tip);
  }
  std::string traced_info = info_vec[0]->location()->ToString(tip);
  for (size_t i = 1; i < info_vec.size(); i++) {
    auto action_name = info_vec[i - 1]->trace_info()->GetActionBetweenNode(info_vec[i]);
    if (action_name == "") {
      break;
    }
    traced_info = traced_info + action_name + info_vec[i]->location()->ToString(tip);
  }
  return traced_info;
}

std::string GetTracedDebugInfo(const DebugInfoPtr &info, SourceLineTip tip) {
  if (info == nullptr) {
    return "";
  }
  auto info_vec = GetSourceCodeDebugInfoVec(info);
  if (info_vec.size() == 0) {
    return "";
  } else if (info_vec.size() == 1) {
    return info_vec[0]->location()->ToString(tip);
  } else if (info_vec.size() > 1) {
    return GetInfoWithAction(info_vec, tip);
  }
  return "";
}

std::string GetDebugInfo(const DebugInfoPtr &info, const std::string &prefix, SourceLineTip tip) {
  std::ostringstream oss;
  if (info == nullptr) {
    return "";
  }

  auto debug_info = GetTracedDebugInfo(info, tip);
  if (tip == kSourceLineTipDiscard) {
    std::replace(debug_info.begin(), debug_info.end(), '\r', '/');
    std::replace(debug_info.begin(), debug_info.end(), '\n', '/');
  }
  oss << prefix << debug_info;
  return oss.str();
}

std::string GetGraphParamString(const FuncGraphPtr &graph, abstract::AbstractBasePtrList args_spec_list) {
  std::ostringstream oss;
  oss << "graph:" << graph->ToString() << " with args[";
  auto params = graph->parameters();
  for (size_t i = 0; i < args_spec_list.size(); i++) {
    oss << params[i]->ToString() << ":<" << GetAbstractStr(args_spec_list[i]) << ">,";
  }
  oss << "]";
  oss << GetDebugInfo(graph->debug_info(), kSourceLineTipDiscard);
  return oss.str();
}

void DumpInferStack(std::ostringstream &oss) {
  auto &infer_stack = GetCurrenGraphInferStack();
  if (infer_stack.empty()) {
    return;
  }
  std::vector<std::pair<abstract::EvaluatorPtr, abstract::AnfNodeConfigPtr>> infer_vec;
  while (!infer_stack.empty()) {
    auto top = infer_stack.top();
    infer_vec.push_back(top);
    infer_stack.pop();
  }
  std::reverse(infer_vec.begin(), infer_vec.end());
  int index = 0;
  for (auto &item : infer_vec) {
    auto graph_infer = std::dynamic_pointer_cast<abstract::BaseFuncGraphEvaluator>(item.first);
    if (graph_infer == nullptr) {
      MS_LOG(WARNING) << "DumpInferStack failed, got null graph evaluator";
      infer_vec.clear();
      break;
    }
    auto graph_context = graph_infer->graph_context();
    if (graph_context == nullptr) {
      MS_LOG(INFO) << "Null context continue";
      continue;
    }
    auto graph = graph_context->func_graph();
    auto args_spec_list = graph_context->args_spec_list();
    oss << "    #" << index++ << " " << GetGraphParamString(graph, args_spec_list);
  }
}

void TraceGraphInfer() {
  auto &infer_stack = GetCurrenGraphInferStack();
  std::ostringstream oss;
  if (infer_stack.empty()) {
    return;
  }
  MS_LOG(INFO) << "\n*******************************graph evaluate stack**********************************";
  oss << std::endl;
  DumpInferStack(oss);
  MS_LOG(INFO) << oss.str();
  MS_LOG(INFO) << "\n*************************************************************************************";
}

class AnalyzedFuncGraphExporter : public AnfExporter {
 public:
  AnalyzedFuncGraphExporter() : AnfExporter("", true, false) {}
  ~AnalyzedFuncGraphExporter() override = default;

  void ExportFuncGraph(const std::string &filename, const std::vector<abstract::AnfNodeConfigPtr> &node_cfgs);

 private:
  std::string GetNodeType(const AnfNodePtr &nd) override;
};

std::unordered_map<FuncGraphPtr, TaggedNodeMap> CalcTaggedFuncGraphs() {
  std::unordered_map<FuncGraphPtr, TaggedNodeMap> tagged_func_graphs;
  auto &list = GetCNodeDebugStack();
  for (size_t i = 0; i < list.size(); ++i) {
    auto node_cfg = list[i];
    auto fg = node_cfg->context()->func_graph();
    auto node = node_cfg->node();
    tagged_func_graphs[fg][node] = i;
  }
  return tagged_func_graphs;
}

void OutputAnalyzedGraphWithType() {
  AnalyzedFuncGraphExporter exporter;
  exporter.ExportFuncGraph("analyze_fail.dat", GetCNodeDebugStack());
}

std::string AnalyzedFuncGraphExporter::GetNodeType(const AnfNodePtr &node) {
  if (node_cfg_ == nullptr) {
    return AnfExporter::GetNodeType(node);
  }
  auto ctx = node_cfg_->context();
  auto engine = node_cfg_->engine();
  auto cfg = engine->MakeConfig(node, ctx);
  auto abs = engine->cache().GetValue(cfg);

  if (abs == nullptr) {
    return "Undefined";
  }
  auto dtype = abs->BuildType();
  auto shape = abs->BuildShape();
  std::ostringstream oss;
  if (dtype != nullptr && abs->isa<abstract::AbstractTensor>() && shape != nullptr) {
    oss << dtype->DumpText() << shape->DumpText();
  } else if (dtype != nullptr) {
    oss << dtype->DumpText();
  } else {
    oss << "Undefined";
  }
  return oss.str();
}

void AnalyzedFuncGraphExporter::ExportFuncGraph(const std::string &filename,
                                                const std::vector<abstract::AnfNodeConfigPtr> &node_cfgs) {
  if (node_cfgs.empty()) {
    MS_LOG(DEBUG) << "Node configs is empty";
    return;
  }

  std::ofstream ofs(filename);
  if (!ofs.is_open()) {
    MS_LOG(ERROR) << "Open file '" << filename << "' failed!";
    return;
  }

  param_index = 1;
  auto tagged_func_graphs = CalcTaggedFuncGraphs();

  // first output graph on the analysis stack
  for (const auto &node_cfg : node_cfgs) {
    auto fg = node_cfg->context()->func_graph();
    // the graph is already output, skip it
    if (exported.find(fg) != exported.end()) {
      continue;
    }
    // set node_cfg info for getting type
    node_cfg_ = node_cfg;
    tagged_cnodes_ = tagged_func_graphs[fg];
    ExportOneFuncGraph(ofs, fg);
    ofs << "\n\n";
  }

  node_cfg_ = nullptr;
  tagged_cnodes_.clear();

  // print seperator between function graphs on analyzed graph call stack and others
  ofs << "#===============================================================================\n\n\n";

  // second output other graphs
  while (!func_graph_set.empty()) {
    FuncGraphPtr fg = *func_graph_set.begin();
    ExportOneFuncGraph(ofs, fg);
    ofs << "\n\n";
    (void)func_graph_set.erase(fg);
  }
  ofs << "# num of total function graphs: " << exported.size();

  ofs.close();
}

void GetInferStackInfo(std::ostringstream &oss) {
  MS_LOG(INFO) << "Get graph analysis information begin";
  auto stack = GetCNodeDebugStack();
  if (stack.empty()) {
    MS_LOG(INFO) << "Length of analysis information stack is empty.";
    return;
  }

  OutputAnalyzedGraphWithType();
  oss << "\nThe function call stack:\n";

  int index = 0;
  std::string last_py_func = "";
  for (size_t i = 0; i < stack.size(); ++i) {
    auto node_cfg = stack[i];

    auto cnode = dyn_cast<CNode>(node_cfg->node());
    if (cnode == nullptr) {
      MS_LOG(DEBUG) << "CNode of elements[" << i << "] is nullptr.";
      continue;
    }

    auto debug_info = cnode->debug_info();
    auto this_py_func = debug_info->get_python_func_belonged();
    if (i > 0 && (this_py_func == last_py_func)) {
      MS_LOG(DEBUG) << "Python function of elements[" << i << "] is same as previous.";
      continue;
    }
    last_py_func = this_py_func;
    oss << "# " << index++ << " " << trace::GetDebugInfo(debug_info, std::string(""));
  }

  stack.clear();
  MS_LOG(INFO) << "Get graph analysis information *end*";
}

// trace the graph evaluator stack
static std::stack<std::pair<abstract::EvaluatorPtr, abstract::AnfNodeConfigPtr>> graph_infer_stack;
// trace the cnode infer debug info
static std::vector<abstract::AnfNodeConfigPtr> cnode_debug_stack{};
void TraceGraphInferEnter(const abstract::EvaluatorPtr &eval, const abstract::AnfNodeConfigPtr &node) {
  if (eval == nullptr) {
    MS_LOG(EXCEPTION) << "GraphInferEnter got null eval";
  }
  if (eval->isa<abstract::FuncGraphEvaluator>() || eval->isa<abstract::MetaFuncGraphEvaluator>()) {
    graph_infer_stack.emplace(std::pair<abstract::EvaluatorPtr, abstract::AnfNodeConfigPtr>(eval, node));
  }
}

void TraceGraphInferLeave(const abstract::EvaluatorPtr &eval) {
  if (eval == nullptr) {
    MS_LOG(EXCEPTION) << "GraphInferEnter got null eval";
  }
  if (eval->isa<abstract::FuncGraphEvaluator>() || eval->isa<abstract::MetaFuncGraphEvaluator>()) {
    graph_infer_stack.pop();
  }
}

void TraceInferCNodeEnter(const abstract::AnfNodeConfigPtr &node_cfg) { cnode_debug_stack.push_back(node_cfg); }

void TraceInferCNodeLeave() { cnode_debug_stack.pop_back(); }

std::vector<abstract::AnfNodeConfigPtr> &GetCNodeDebugStack() { return cnode_debug_stack; }

std::stack<std::pair<abstract::EvaluatorPtr, abstract::AnfNodeConfigPtr>> &GetCurrenGraphInferStack() {
  return graph_infer_stack;
}
void ClearTraceStack() {
  while (!graph_infer_stack.empty()) {
    graph_infer_stack.pop();
  }
  cnode_debug_stack.clear();
}
}  // namespace trace
}  // namespace mindspore

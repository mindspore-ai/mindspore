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

#include "pipeline/jit/debug/trace.h"

#include <iostream>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>

#include "utils/hash_map.h"
#include "utils/hash_set.h"
#include "frontend/operator/composite/composite.h"
#include "ir/tensor.h"
#include "pipeline/jit/debug/anf_ir_utils.h"
#include "include/common/debug/common.h"
#include "include/common/debug/anf_ir_dump.h"
#include "pipeline/jit/static_analysis/evaluator.h"
#include "pipeline/jit/static_analysis/async_eval_result.h"
#include "utils/log_adapter.h"
#include "include/common/utils/comm_manager.h"
#include "abstract/abstract_value.h"
#include "utils/file_utils.h"

namespace mindspore {
// namespace to support debug trace information
namespace trace {
using abstract::AbstractBasePtr;
using abstract::AnalysisContextPtr;
using abstract::AnalysisEnginePtr;
using abstract::AnfNodeConfigPtr;

std::string GetAbstractStr(const abstract::AbstractBasePtr &abs) {
  if (abs == nullptr) {
    return "<NullAbstract>";
  }
  auto shape = abs->BuildShape()->cast<abstract::BaseShapePtr>();
  TypePtr type = abs->BuildType();
  std::ostringstream oss;
  if ((shape != nullptr) && (type != nullptr)) {
    oss << "<" << type << ", " << shape->ToString() << ">";
  } else if (type != nullptr) {
    oss << "<" << type << ">";
  } else {
    oss << "<null>";
  }
  return oss.str();
}

std::string GetGraphParamString(const FuncGraphPtr &graph, const abstract::AbstractBasePtrList &args_spec_list) {
  MS_EXCEPTION_IF_NULL(graph);
  std::ostringstream oss;
  oss << "graph:" << graph->ToString() << " with args[";
  auto params = graph->parameters();
  if (params.size() < args_spec_list.size()) {
    MS_EXCEPTION(TypeError) << "The size of parameters less than args_spec_list's size.";
  }
  for (size_t i = 0; i < args_spec_list.size(); i++) {
    auto parameter = params[i];
    MS_EXCEPTION_IF_NULL(parameter);
    oss << parameter->ToString() << ":<" << GetAbstractStr(args_spec_list[i]) << ">,";
  }
  oss << "]";
  oss << GetDebugInfo(graph->debug_info(), kSourceLineTipDiscard);
  return oss.str();
}

void DumpInferStack(std::ostringstream &oss) {
  auto &graph_stack = GetCurrentGraphEvalStack();
  if (graph_stack.empty()) {
    return;
  }
  std::vector<std::pair<abstract::AnalysisContextPtr, abstract::AnfNodeConfigPtr>> infer_vec;
  while (!graph_stack.empty()) {
    auto top = graph_stack.back();
    infer_vec.push_back(top);
    graph_stack.pop_back();
  }
  std::reverse(infer_vec.begin(), infer_vec.end());
  int index = 0;
  for (const auto &item : infer_vec) {
    auto context = item.first;
    if (context == nullptr) {
      MS_LOG(EXCEPTION) << "DumpInferStack failed, got null graph context";
    }
    auto graph = context->func_graph();
    if (graph == nullptr) {  // Top context.
      continue;
    }
    const auto &args_spec_list = context->args_spec_list();
    if (graph->parameters().size() < args_spec_list.size()) {
      continue;
    }
    oss << "    #" << index++ << " " << GetGraphParamString(graph, args_spec_list) << "\n";
  }
}

void TraceGraphEval() {
  auto &graph_stack = GetCurrentGraphEvalStack();
  if (graph_stack.empty()) {
    MS_LOG(INFO) << "Length of analysis graph stack is empty.";
    return;
  }
  std::ostringstream oss;
  oss << "\n*******************************graph evaluate stack**********************************";
  oss << std::endl;
  DumpInferStack(oss);
  oss << "\n*************************************************************************************";
  MS_LOG(INFO) << oss.str();
}

class AnalyzeFailExporter : public AnfExporter {
 public:
  AnalyzeFailExporter() : AnfExporter(true, false) {}
  ~AnalyzeFailExporter() override = default;

  bool ExportFuncGraph(const std::string &filename, const TraceCNodeEvalStack &node_config_stack);

 protected:
  void OutputCNode(std::ostringstream &oss, const CNodePtr &cnode, const FuncGraphPtr &func_graph, int *idx,
                   std::map<AnfNodePtr, int> *const apply_map) override;
  std::string GetNodeType(const AnfNodePtr &node) override;
  AbstractBasePtr GetNodeAbstract(const AnfNodePtr &node);
  AnfNodeConfigPtr GetForwardConfig(const AnfNodeConfigPtr &cfg);
  void ProcessFuncGraphCall(const CNodePtr &node, std::string *const op_comment);
  mindspore::HashMap<FuncGraphPtr, TaggedNodeMap> CreateTaggedNodeMap(
    const std::vector<abstract::AnfNodeConfigPtr> &node_config_stack);

 private:
  AnalysisContextPtr current_context_ = nullptr;
  AnalysisEnginePtr engine_ = nullptr;
};

mindspore::HashMap<FuncGraphPtr, TaggedNodeMap> AnalyzeFailExporter::CreateTaggedNodeMap(
  const std::vector<abstract::AnfNodeConfigPtr> &node_config_stack) {
  mindspore::HashSet<abstract::AnfNodeConfigPtr> forwarded_configs;  // Check if config. is forwarded.
  mindspore::HashMap<FuncGraphPtr, TaggedNodeMap> tagged_func_graphs;
  size_t index = 0;
  for (auto &node_config : node_config_stack) {
    MS_EXCEPTION_IF_NULL(node_config);

    // Record new config in set.
    auto new_config = GetForwardConfig(node_config);
    if (new_config != node_config) {
      MS_LOG(DEBUG) << "The node_config is forwarded, old config: " << node_config->ToString()
                    << ", new_config: " << new_config->ToString();
      forwarded_configs.emplace(new_config);
    }

    // Ignore the new config.
    if (forwarded_configs.find(node_config) != forwarded_configs.end()) {
      continue;
    }

    auto fg = node_config->func_graph();
    MS_EXCEPTION_IF_NULL(fg);
    auto node = node_config->node();
    tagged_func_graphs[fg][node] = index;
    index++;
  }
  return tagged_func_graphs;
}

bool OutputAnalyzedGraphWithType(const string &file_path) {
  AnalyzeFailExporter exporter;
  return exporter.ExportFuncGraph(file_path, GetCNodeDebugStack());
}

std::string AnalyzeFailExporter::GetNodeType(const AnfNodePtr &node) {
  if (current_context_ == nullptr) {
    return AnfExporter::GetNodeType(node);
  }

  MS_EXCEPTION_IF_NULL(engine_);
  try {
    FuncGraphPtr dummy_call_func_graph = nullptr;
    auto cfg = engine_->MakeConfig(node, current_context_, dummy_call_func_graph);
    auto res = abstract::AnalysisResultCacheMgr::GetInstance().GetValue(cfg);
    if (res != nullptr) {
      return GetAbstractStr(res->abstract());
    }
  } catch (const std::exception &e) {
    MS_LOG(INFO) << "Exception: " << e.what();
  }
  return "<null>";
}

AbstractBasePtr AnalyzeFailExporter::GetNodeAbstract(const AnfNodePtr &node) {
  if (current_context_ == nullptr) {
    return nullptr;
  }
  MS_EXCEPTION_IF_NULL(engine_);
  try {
    FuncGraphPtr dummy_call_func_graph = nullptr;
    auto cfg = engine_->MakeConfig(node, current_context_, dummy_call_func_graph);
    auto res = abstract::AnalysisResultCacheMgr::GetInstance().GetValue(cfg);
    return res == nullptr ? nullptr : res->abstract();
  } catch (const std::exception &e) {
    MS_LOG(INFO) << "Exception: " << e.what();
  }
  return nullptr;
}

AnfNodeConfigPtr AnalyzeFailExporter::GetForwardConfig(const AnfNodeConfigPtr &cfg) {
  MS_EXCEPTION_IF_NULL(cfg);
  MS_EXCEPTION_IF_NULL(engine_);
  AnfNodeConfigPtr cur_cfg = cfg;
  auto iter = engine_->anfnode_config_map().find(cur_cfg);
  while (iter != engine_->anfnode_config_map().end()) {
    auto node = cur_cfg->node();
    cur_cfg = iter->second;
    MS_LOG(DEBUG) << "Get forward node: " << node << "[" << node->DebugString() << "] --> " << cur_cfg->node() << "["
                  << cur_cfg->node()->DebugString() << "]";
    iter = engine_->anfnode_config_map().find(cur_cfg);
  }
  return cur_cfg;
}

void AnalyzeFailExporter::ProcessFuncGraphCall(const CNodePtr &node, std::string *const op_comment) {
  if (node == nullptr) {
    MS_LOG(ERROR) << "Node is nullptr";
    return;
  }
  CNodePtr cnode = nullptr;
  try {
    FuncGraphPtr dummy_call_func_graph = nullptr;
    auto cfg = engine_->MakeConfig(node, current_context_, dummy_call_func_graph);
    cfg = GetForwardConfig(cfg);
    MS_EXCEPTION_IF_NULL(cfg);
    cnode = dyn_cast<CNode>(cfg->node());
  } catch (const std::exception &e) {
    MS_LOG(INFO) << "Exception: " << e.what();
  }
  if (cnode == nullptr) {
    MS_LOG(INFO) << "CNode is nullptr";
    return;
  }

  const auto &inputs = cnode->inputs();
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto op_abs = GetNodeAbstract(inputs[i]);
    if (op_abs == nullptr) {
      MS_LOG(DEBUG) << "Abstract of inputs[" << i << "] of cnode " << cnode->ToString() << "  is nullptr";
      continue;
    }

    if (!op_abs->isa<abstract::FuncGraphAbstractClosure>() && !op_abs->isa<abstract::MetaFuncGraphAbstractClosure>()) {
      MS_LOG(DEBUG) << "Inputs[" << i << "] of cnode " << cnode->ToString() << " is of type " << op_abs->type_name()
                    << ", not function, ignore it";
      // Get prototype of VirtualEvaluator for printing
      if (i == 0 && op_abs->isa<abstract::VirtualAbstractClosure>()) {
        auto func = dyn_cast<abstract::VirtualAbstractClosure>(op_abs);
        std::ostringstream oss;
        oss << "(";
        bool first_flag = false;
        for (const auto &arg : func->args_spec_list()) {
          if (!first_flag) {
            first_flag = true;
          } else {
            oss << ", ";
          }
          oss << GetAbstractStr(arg);
        }
        oss << ") -> " << GetAbstractStr(func->output()) << " ";
        *op_comment = oss.str();
      }
    }
  }
}

void AnalyzeFailExporter::OutputCNode(std::ostringstream &oss, const CNodePtr &cnode, const FuncGraphPtr &func_graph,
                                      int *idx, std::map<AnfNodePtr, int> *const apply_map) {
  OutputCNodeText(oss, cnode, func_graph, idx, apply_map);
  // Process function graph call
  std::string op_comment;
  ProcessFuncGraphCall(cnode, &op_comment);
  if (!op_comment.empty()) {
    auto &inputs = cnode->inputs();
    oss << "    #" << GetAnfNodeText(func_graph, inputs[0], *apply_map) << ".prototype = " << op_comment;
  }
  // Output comment
  OutputStatementComment(cnode, func_graph, oss);
  oss << "\n";
}

bool AnalyzeFailExporter::ExportFuncGraph(const std::string &filename, const TraceCNodeEvalStack &node_config_stack) {
  if (node_config_stack.empty()) {
    MS_LOG(DEBUG) << "Node configs is empty";
    return false;
  }
  auto real_filepath = Common::CreatePrefixPath(filename);
  if (!real_filepath.has_value()) {
    MS_LOG(ERROR) << "The export ir path: " << filename << " is not illegal.";
    return false;
  }
  ChangeFileMode(real_filepath.value(), S_IWUSR);
  std::ofstream ofs(real_filepath.value());
  if (!ofs.is_open()) {
    MS_LOG(ERROR) << "Open file '" << real_filepath.value() << "' failed!" << ErrnoToString(errno);
    return false;
  }
  ofs << "# 1.This file shows the parsed IR info when graph evaluating failed to help find the problem.\n";
  ofs << "# 2.You can search the last `------------------------>` to the node which is inferred failed.\n";
  ofs << "# 3.Refer to https://www.mindspore.cn/search?inputValue=analyze_fail.ir to get more instructions.\n";
  ofs << "# ===============================================================================\n\n";

  if (engine_ == nullptr) {
    engine_ = node_config_stack.front()->engine();
  }

  auto tagged_func_graphs = CreateTaggedNodeMap(node_config_stack);
  mindspore::HashSet<FuncGraphPtr> printed_func_graphs;  // Check if func graph has been printed.
  // Output graph on the analysis stack
  for (const auto &node_config : node_config_stack) {
    MS_EXCEPTION_IF_NULL(node_config);
    auto fg = node_config->func_graph();
    MS_LOG(INFO) << "Node: " << node_config->node()->DebugString()
                 << ", FV: " << (node_config->func_graph() != node_config->context()->func_graph())
                 << ", calling func graph: " << node_config->func_graph()->ToString()
                 << ", context func graph: " << node_config->context()->func_graph()->ToString();
    if (fg == nullptr) {
      MS_LOG(ERROR) << "FuncGraph is null, context: " << node_config->ToString();
      continue;
    }
    if (printed_func_graphs.find(fg) != printed_func_graphs.end()) {
      continue;
    }
    (void)printed_func_graphs.emplace(fg);

    current_context_ = node_config->context();  // Set current context.
    std::ostringstream buffer;
    ExportOneFuncGraph(fg, tagged_func_graphs[fg], buffer);
    ofs << buffer.str() << "\n\n";
  }

  ofs << "#===============================================================================\n";
  ofs << "# num of function graphs in stack: ";
  auto ignored_num = (node_config_stack.size() - printed_func_graphs.size());
  if (ignored_num == 0) {
    ofs << node_config_stack.size() << "\n";
  } else {
    ofs << printed_func_graphs.size() << "/" << node_config_stack.size() << " (Ignored " << ignored_num
        << " internal frames).\n";
  }
  ofs.close();
  ChangeFileMode(real_filepath.value(), S_IRUSR);
  return true;
}

std::string GetEvalFailDatPath() {
  std::string path;
  auto ms_om_path = common::GetEnv("MS_OM_PATH");
  if (!ms_om_path.empty()) {
    path = ms_om_path;
  } else {
    path = ".";
  }
  path += "/rank_" + std::to_string(GetRank()) + "/om/analyze_fail.ir";
  // Support "../" in path.
  auto realpath = Common::CreatePrefixPath(path, true);
  if (!realpath.has_value()) {
    MS_EXCEPTION(ValueError) << "Get real path failed. path=" << path;
  }
  return realpath.value();
}

void GetEvalStackInfo(std::ostringstream &oss) {
  MS_LOG(INFO) << "Get graph analysis information begin";
  auto stack = GetCNodeDebugStack();
  if (stack.empty()) {
    MS_LOG(INFO) << "Length of analysis information stack is empty.";
    return;
  }
  oss << "\nThe function call stack";
#ifndef ENABLE_SECURITY
  std::string file_name = GetEvalFailDatPath();
  auto ret = OutputAnalyzedGraphWithType(file_name);
  if (ret) {
    oss << " (See file '" << file_name
        << "' for more details. Get instructions about `analyze_fail.ir` at "
           "https://www.mindspore.cn/search?inputValue=analyze_fail.ir)";
  }
#endif
  oss << ":\n";

  int index = 0;
  std::string last_location_info = "";
  for (size_t i = 0; i < stack.size(); ++i) {
    auto node_config = stack[i];
    MS_EXCEPTION_IF_NULL(node_config);
    auto cnode = dyn_cast<CNode>(node_config->node());
    if (cnode == nullptr) {
      MS_LOG(DEBUG) << "CNode of elements[" << i << "] is nullptr.";
      continue;
    }

    auto debug_info = cnode->debug_info();
    auto this_location_info = trace::GetDebugInfo(debug_info, std::string(""));
    if (this_location_info.empty() || this_location_info == last_location_info) {
      continue;
    }

    last_location_info = this_location_info;
    oss << "# " << index++ << " " << this_location_info;
  }

  stack.clear();
  MS_LOG(INFO) << "Get graph analysis information *end*";
}

// Trace the graph evaluator stack
thread_local TraceGraphEvalStack graph_infer_stack;
// Trace the cnode infer debug info
thread_local TraceCNodeEvalStack cnode_debug_stack{};

void TraceGraphEvalEnter(const abstract::AnalysisContextPtr &context, const abstract::AnfNodeConfigPtr &node) {
  if (context == nullptr) {
    MS_LOG(EXCEPTION) << "GraphInferEnter got null context";
  }
  (void)graph_infer_stack.push_back(std::pair<abstract::AnalysisContextPtr, abstract::AnfNodeConfigPtr>(context, node));
}

void TraceGraphEvalLeave(const abstract::AnalysisContextPtr &context) {
  if (context == nullptr || graph_infer_stack.empty()) {
    MS_LOG(EXCEPTION) << "The context is null, or call stack is empty.";
  }
  if (context != graph_infer_stack.back().first) {
    MS_LOG(EXCEPTION) << "Different context: " << context->func_graph()->ToString() << ", "
                      << graph_infer_stack.back().first->func_graph()->ToString();
  }
  graph_infer_stack.pop_back();
}

void TraceGraphEvalStackPrepare(const TraceGraphEvalStack &graph_evals) {
  (void)graph_infer_stack.insert(graph_infer_stack.end(), graph_evals.begin(), graph_evals.end());
}

void TraceEvalCNodeStackPrepare(const TraceCNodeEvalStack &cnode_evals) {
  (void)cnode_debug_stack.insert(cnode_debug_stack.end(), cnode_evals.begin(), cnode_evals.end());
}

void TraceEvalCNodeEnter(const abstract::AnfNodeConfigPtr &node_config) { cnode_debug_stack.push_back(node_config); }

void TraceEvalCNodeLeave() { cnode_debug_stack.pop_back(); }

TraceCNodeEvalStack &GetCNodeDebugStack() { return cnode_debug_stack; }

TraceGraphEvalStack &GetCurrentGraphEvalStack() { return graph_infer_stack; }

void ClearTraceStack() {
  while (!graph_infer_stack.empty()) {
    graph_infer_stack.pop_back();
  }
  cnode_debug_stack.clear();
}

void PrintMessage(std::ostringstream &oss, const std::string &content, bool add_title) {
  if (add_title) {
    const std::string &message = oss.str();
    size_t length = message.length();
    if ((length != 0) && (message[length - 1] != '\n')) {
      oss << "\n";
    }

    oss << "\n----------------------------------------------------\n"
        << "- The Traceback of Net Construct Code:"
        << "\n----------------------------------------------------";
  }
  oss << content;
}

void GetTraceStackInfo(std::ostringstream &oss, bool add_title) {
  TraceGraphEval();
  std::ostringstream trace_info;
  GetEvalStackInfo(trace_info);
  if (trace_info.str().empty()) {
    DebugInfoPtr debug_info = TraceManager::record_debug_info();
    if (debug_info != nullptr && TraceManager::record_debug_info_flag() == true) {
      auto debug_str = trace::GetDebugInfo(debug_info);
      if (!debug_str.empty()) {
        std::ostringstream content;
        content << "\n\n# " << debug_str;
        PrintMessage(oss, content.str(), add_title);
      }
    }
  } else {
    PrintMessage(oss, trace_info.str(), add_title);
  }
}

// Register trace provider to LogWriter.
struct TraceProviderRegister {
  TraceProviderRegister() noexcept { LogWriter::SetTraceProvider(GetTraceStackInfo); }
  ~TraceProviderRegister() = default;
} trace_provider_register;

// Register trace cnode provider to AbstractBase.
struct TraceNodeProviderRegister {
  TraceNodeProviderRegister() noexcept {
    abstract::AbstractBase::set_trace_node_provider([](AnfNodePtr *node) {
      auto stack = GetCNodeDebugStack();
      if (!stack.empty()) {
        auto conf = stack.back();
        *node = conf->node();
      }
    });
  }
  ~TraceNodeProviderRegister() = default;
} trace_node_provider_register;
}  // namespace trace
}  // namespace mindspore

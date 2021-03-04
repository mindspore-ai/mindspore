/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "ir/graph_utils.h"
#include "frontend/operator/composite/composite.h"
#include "ir/tensor.h"
#include "debug/anf_ir_utils.h"
#include "pipeline/jit/static_analysis/evaluator.h"
#include "utils/log_adapter.h"

namespace mindspore {
// namespace to support debug trace information
namespace trace {
using abstract::AbstractBasePtr;
using abstract::AnalysisContextPtr;
using abstract::AnalysisEnginePtr;
using abstract::AnfNodeConfigPtr;

std::string GetAbstractStr(const abstract::AbstractBasePtr &abs) {
  if (abs == nullptr) {
    return "NullAbstract";
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
  auto &infer_stack = GetCurrenGraphEvalStack();
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

void TraceGraphEval() {
  auto &infer_stack = GetCurrenGraphEvalStack();
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

  void ExportOneFuncGraph(std::ofstream &ofs, const FuncGraphPtr &func_graph);
  void OutputCNodes(std::ofstream &ofs, const std::vector<AnfNodePtr> &nodes, const FuncGraphPtr &func_graph);
  void OutputCNode(std::ofstream &ofs, const CNodePtr &cnode, const FuncGraphPtr &func_graph, int *idx,
                   std::map<AnfNodePtr, int> *const apply_map);

 private:
  std::string GetNodeType(const AnfNodePtr &nd) override;
  AbstractBasePtr GetNodeAbstract(const AnfNodePtr &nd);
  AnfNodeConfigPtr GetFordwardConfigPtr(const AnfNodeConfigPtr &cfg);
  std::vector<AnalysisContextPtr> ProcessFuncGraphCall(const CNodePtr &node, std::string *const op_comment);
  void OutputStatementComment(std::ofstream &ofs, const CNodePtr &node, const std::vector<AnalysisContextPtr> &ctxs);

  // key: context, val: whether the context has already been printed
  std::unordered_map<AnalysisContextPtr, bool> context_map_;
  std::vector<AnalysisContextPtr> context_vec_;

  AnalysisContextPtr cur_ctx_ = nullptr;
  AnalysisEnginePtr engine_ = nullptr;
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
  if (cur_ctx_ == nullptr) {
    return AnfExporter::GetNodeType(node);
  }

  MS_EXCEPTION_IF_NULL(engine_);
  auto cfg = engine_->MakeConfig(node, cur_ctx_);
  auto ret = engine_->analysis_cache().GetValue(cfg);
  if (ret == nullptr) {
    return "Undefined";
  }
  return GetAbstractStr(ret->abstract());
}

AbstractBasePtr AnalyzedFuncGraphExporter::GetNodeAbstract(const AnfNodePtr &node) {
  if (cur_ctx_ == nullptr) {
    return nullptr;
  }
  MS_EXCEPTION_IF_NULL(engine_);
  auto cfg = engine_->MakeConfig(node, cur_ctx_);
  auto ret = engine_->analysis_cache().GetValue(cfg);
  return ret == nullptr ? nullptr : ret->abstract();
}

AnfNodeConfigPtr AnalyzedFuncGraphExporter::GetFordwardConfigPtr(const AnfNodeConfigPtr &cfg) {
  AnfNodeConfigPtr cur_cfg = cfg;
  auto iter = engine_->anfnode_config_map().find(cur_cfg);
  while (iter != engine_->anfnode_config_map().end()) {
    auto node = cur_cfg->node();
    cur_cfg = iter->second;
    MS_LOG(DEBUG) << "Get forword node: " << node.get() << "[" << node->ToString() << "] --> " << cur_cfg->node().get()
                  << "[" << cur_cfg->node()->ToString() << "]";
    iter = engine_->anfnode_config_map().find(cur_cfg);
  }
  return cur_cfg;
}

std::vector<AnalysisContextPtr> AnalyzedFuncGraphExporter::ProcessFuncGraphCall(const CNodePtr &node,
                                                                                std::string *const op_comment) {
  std::vector<AnalysisContextPtr> ret_contexts;
  if (node == nullptr) {
    return ret_contexts;
  }
  auto cfg = engine_->MakeConfig(node, cur_ctx_);
  cfg = GetFordwardConfigPtr(cfg);
  auto cnode = dyn_cast<CNode>(cfg->node());
  if (cnode == nullptr) {
    MS_LOG(DEBUG) << "CNode is nullptr";
    return ret_contexts;
  }

  ret_contexts.resize(cnode->size());

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
      continue;
    }

    auto evaluator = engine_->GetEvaluatorFor(dyn_cast<abstract::AbstractFunction>(op_abs));
    if (!evaluator->isa<abstract::BaseFuncGraphEvaluator>()) {
      MS_LOG(DEBUG) << "Evaluator for inputs[" << i << "] of cnode " << cnode->ToString() << " is of type "
                    << evaluator->type_name() << ", not BaseFuncGraphEvaluator, ignore it.";
      continue;
    }

    auto base_fg_evaluator = dyn_cast<abstract::BaseFuncGraphEvaluator>(evaluator);
    auto ctx = base_fg_evaluator->graph_context();
    if (ctx != nullptr && context_map_.insert({ctx, false}).second) {
      MS_LOG(DEBUG) << "Add new context, ctx.addr = " << ctx.get() << "ctx = " << ctx->ToString();
      context_vec_.push_back(ctx);
    }
    ret_contexts[i] = ctx;
  }
  return ret_contexts;
}

void AnalyzedFuncGraphExporter::OutputStatementComment(std::ofstream &ofs, const CNodePtr &node,
                                                       const std::vector<AnalysisContextPtr> &ctxs) {
  if (node == nullptr) {
    return;
  }

  // output type of each input argument
  auto &inputs = node->inputs();
  if (inputs.size() > 1) {
    ofs << "    #(";
    for (size_t i = 1; i < inputs.size(); ++i) {
      if (i != 1) {
        ofs << ", ";
      }
      AnfNodePtr arg = inputs[i];
      ofs << GetNodeType(arg);
    }
    ofs << ")";
  }
  // output other comment, map the graph name to original representation(containing unicode character)
  std::ostringstream comment;
  comment << "    #";
  bool has_comment = false;
  for (size_t i = 0; i < inputs.size(); ++i) {
    AnfNodePtr arg = inputs[i];
    if (!IsValueNode<FuncGraph>(arg)) {
      continue;
    }
    if (!has_comment) {
      has_comment = true;
    } else {
      comment << ",";
    }
    FuncGraphPtr fg = GetValueNode<FuncGraphPtr>(arg);
    std::string func_graph_id = fg->debug_info()->get_id();
    comment << " fg_" << func_graph_id << "=" << fg->ToString() << "." << func_graph_id;
    if (ctxs.size() > i && ctxs[i] != nullptr) {
      comment << "(@ctx.addr=" << ctxs[i].get() << ")";
    }
  }
  if (has_comment) {
    ofs << comment.str();
  }
  ofs << " #scope: " << node->scope()->name();
}

void AnalyzedFuncGraphExporter::OutputCNode(std::ofstream &ofs, const CNodePtr &cnode, const FuncGraphPtr &func_graph,
                                            int *idx, std::map<AnfNodePtr, int> *const apply_map) {
  auto &inputs = cnode->inputs();
  std::string op_text = GetAnfNodeText(func_graph, inputs[0], *apply_map);
  // non-return node
  if (cnode != func_graph->get_return()) {
    int apply_idx = (*idx)++;
    (*apply_map)[cnode] = apply_idx;
    std::string type_info = GetNodeType(cnode);
    if (type_info == "Undefined") {
      ofs << "    %" << apply_idx << " = " << op_text << "(";
    } else {
      ofs << "    %" << apply_idx << " : " << type_info << " = " << op_text << "(";
    }
  } else {
    ofs << "    " << op_text << "(";
  }

  for (size_t i = 1; i < inputs.size(); ++i) {
    if (i != 1) {
      ofs << ", ";
    }
    AnfNodePtr arg = inputs[i];
    ofs << GetAnfNodeText(func_graph, arg, *apply_map);
  }
  ofs << ")";

  // process function graph call
  std::string op_comment;
  auto contexts = ProcessFuncGraphCall(cnode, &op_comment);
  AnalysisContextPtr ctx = contexts.empty() ? nullptr : contexts[0];

  if (!op_comment.empty()) {
    ofs << "    #" << GetAnfNodeText(func_graph, inputs[0], *apply_map) << ".prototype = " << op_comment;
  }
  // output comment
  OutputStatementComment(ofs, cnode, contexts);
  if (ctx != nullptr) {
    ofs << " @ctx.addr=" << ctx.get();
  }
  ofs << "\n";

  if (label_manage::GetGlobalTraceLabelType() == label_manage::TraceLabelType::kWithUniqueId) {
    ofs << trace::GetDebugInfo(cnode->debug_info(), "      # ", kSourceLineTipDiscard) << "#"
        << label_manage::Label(cnode->debug_info()) << "\n";
  } else {
    ofs << trace::GetDebugInfo(cnode->debug_info(), "      # ", kSourceLineTipDiscard) << "\n";
  }
}

void AnalyzedFuncGraphExporter::OutputCNodes(std::ofstream &ofs, const std::vector<AnfNodePtr> &nodes,
                                             const FuncGraphPtr &func_graph) {
  if (func_graph == nullptr) {
    return;
  }

  int idx = 1;
  std::map<AnfNodePtr, int> apply_map;
  for (const AnfNodePtr &node : nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      continue;
    }

    auto iter = tagged_cnodes_.find(node);
    if (iter != tagged_cnodes_.end()) {
      ofs << "\n#------------------------> " << iter->second << "\n";
    }

    auto cnode = node->cast<CNodePtr>();
    OutputCNode(ofs, cnode, func_graph, &idx, &apply_map);
  }
}

void AnalyzedFuncGraphExporter::ExportOneFuncGraph(std::ofstream &ofs, const FuncGraphPtr &func_graph) {
  if (func_graph == nullptr) {
    return;
  }

  std::vector<AnfNodePtr> nodes = TopoSort(func_graph->get_return(), SuccIncoming, AlwaysInclude);
  std::vector<AnfNodePtr> parameters = func_graph->parameters();
  OrderedMap<AnfNodePtr, int, ParamPtrHasher, ParamPtrEqual> param_map;

  ofs << "# [No." << (exported.size() + 1) << "] " << func_graph->DumpText() << "."
      << func_graph->debug_info()->get_id();
  if (cur_ctx_ != nullptr) {
    ofs << " @ctx.addr=" << cur_ctx_.get();
  }
  ofs << "\n";
  if (label_manage::GetGlobalTraceLabelType() == label_manage::TraceLabelType::kWithUniqueId) {
    ofs << trace::GetDebugInfo(func_graph->debug_info(), "# ", kSourceLineTipDiscard) << "#"
        << label_manage::Label(func_graph->debug_info()) << "\n";
  } else {
    ofs << trace::GetDebugInfo(func_graph->debug_info(), "# ", kSourceLineTipDiscard) << "\n";
  }
  ofs << "funcgraph fg_" << func_graph->debug_info()->get_id();
  // output name of parent of graph if exists
  if (func_graph->parent() != nullptr) {
    ofs << "[fg_" << func_graph->parent()->debug_info()->get_id() << "]";
  }
  ofs << "(\n";

  OutputParameters(ofs, parameters, &param_map);

  exported[func_graph] = param_map;
  ofs << (!parameters.empty() ? "    " : "") << ") {\n";

  OutputCNodes(ofs, nodes, func_graph);

  ofs << "}\n";
}

void AnalyzedFuncGraphExporter::ExportFuncGraph(const std::string &filename,
                                                const std::vector<abstract::AnfNodeConfigPtr> &node_cfgs) {
  if (node_cfgs.empty()) {
    MS_LOG(DEBUG) << "Node configs is empty";
    return;
  }

  context_map_.clear();
  context_vec_.clear();

  std::ofstream ofs(filename);
  if (!ofs.is_open()) {
    MS_LOG(ERROR) << "Open file '" << filename << "' failed!";
    return;
  }

  param_index = 1;
  auto tagged_func_graphs = CalcTaggedFuncGraphs();

  // 1. Output graph on the analysis stack
  for (const auto &node_cfg : node_cfgs) {
    auto ctx = node_cfg->context();
    if (engine_ == nullptr) {
      engine_ = node_cfg->engine();
    }
    if (context_map_.insert({ctx, false}).second) {
      context_vec_.push_back(ctx);
    }
    // If the graph has already been printed
    if (context_map_[ctx]) {
      continue;
    }
    context_map_[ctx] = true;

    auto fg = ctx->func_graph();

    // Set current context
    cur_ctx_ = ctx;
    tagged_cnodes_ = tagged_func_graphs[fg];
    ExportOneFuncGraph(ofs, fg);
    ofs << "\n\n";
  }

  tagged_cnodes_.clear();

  // Print separator between function graphs on analyzed graph call stack and others
  ofs << "#===============================================================================\n\n\n";

  // 2. Output other graphs
  size_t ctx_idx = 0;
  while (ctx_idx < context_vec_.size()) {
    auto ctx = context_vec_[ctx_idx++];
    if (context_map_[ctx]) {
      continue;
    }
    context_map_[ctx] = true;
    cur_ctx_ = ctx;
    ExportOneFuncGraph(ofs, ctx->func_graph());
    ofs << "\n\n";
  }

  ofs << "# num of total function graphs: " << context_map_.size() << "\n";

  ofs.close();
}

void GetEvalStackInfo(std::ostringstream &oss) {
  MS_LOG(INFO) << "Get graph analysis information begin";
  auto stack = GetCNodeDebugStack();
  if (stack.empty()) {
    MS_LOG(INFO) << "Length of analysis information stack is empty.";
    return;
  }

  OutputAnalyzedGraphWithType();
  oss << "\nThe function call stack (See file 'analyze_fail.dat' for details):\n";

  int index = 0;
  std::string last_location_info = "";
  for (size_t i = 0; i < stack.size(); ++i) {
    auto node_cfg = stack[i];

    auto cnode = dyn_cast<CNode>(node_cfg->node());
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

// trace the graph evaluator stack
static std::stack<std::pair<abstract::EvaluatorPtr, abstract::AnfNodeConfigPtr>> graph_infer_stack;
// trace the cnode infer debug info
static std::vector<abstract::AnfNodeConfigPtr> cnode_debug_stack{};

void TraceGraphEvalEnter(const abstract::EvaluatorPtr &eval, const abstract::AnfNodeConfigPtr &node) {
  if (eval == nullptr) {
    MS_LOG(EXCEPTION) << "GraphInferEnter got null eval";
  }
  if (eval->isa<abstract::FuncGraphEvaluator>() || eval->isa<abstract::MetaFuncGraphEvaluator>()) {
    graph_infer_stack.emplace(std::pair<abstract::EvaluatorPtr, abstract::AnfNodeConfigPtr>(eval, node));
  }
}

void TraceGraphEvalLeave(const abstract::EvaluatorPtr &eval) {
  if (eval == nullptr) {
    MS_LOG(EXCEPTION) << "GraphInferEnter got null eval";
  }
  if (eval->isa<abstract::FuncGraphEvaluator>() || eval->isa<abstract::MetaFuncGraphEvaluator>()) {
    graph_infer_stack.pop();
  }
}

void TraceEvalCNodeEnter(const abstract::AnfNodeConfigPtr &node_cfg) { cnode_debug_stack.push_back(node_cfg); }

void TraceEvalCNodeLeave() { cnode_debug_stack.pop_back(); }

std::vector<abstract::AnfNodeConfigPtr> &GetCNodeDebugStack() { return cnode_debug_stack; }

std::stack<std::pair<abstract::EvaluatorPtr, abstract::AnfNodeConfigPtr>> &GetCurrenGraphEvalStack() {
  return graph_infer_stack;
}

void ClearTraceStack() {
  while (!graph_infer_stack.empty()) {
    graph_infer_stack.pop();
  }
  cnode_debug_stack.clear();
}

// Register trace provider to LogWriter.
struct TraceProviderRegister {
  TraceProviderRegister() {
    LogWriter::set_trace_provider([](std::ostringstream &oss) {
      TraceGraphEval();
      std::ostringstream trace_info;
      GetEvalStackInfo(trace_info);
      if (trace_info.str().empty()) {
        DebugInfoPtr debug_info = TraceManager::GetParseOrResolveDebugInfo();
        if (debug_info != nullptr) {
          oss << "\n\n# " << trace::GetDebugInfo(debug_info);
        }
      } else {
        oss << trace_info.str();
      }
    });
  }
  ~TraceProviderRegister() = default;
} trace_provider_regsiter;
}  // namespace trace
}  // namespace mindspore

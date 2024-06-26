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

#include "pipeline/jit/ps/debug/trace.h"

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
#include "include/common/debug/common.h"
#include "include/common/debug/anf_ir_dump.h"
#include "pipeline/jit/ps/static_analysis/evaluator.h"
#include "pipeline/jit/ps/static_analysis/async_eval_result.h"
#include "utils/log_adapter.h"
#include "include/common/utils/comm_manager.h"
#include "abstract/abstract_value.h"
#include "utils/file_utils.h"
#include "utils/ms_exception.h"

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

std::string GetGraphParamString(const FuncGraphPtr &graph, const abstract::AbstractBasePtrList &args_abs_list) {
  MS_EXCEPTION_IF_NULL(graph);
  std::ostringstream oss;
  oss << "graph:" << graph->ToString() << " with args[";
  auto params = graph->parameters();
  if (params.size() < args_abs_list.size()) {
    MS_INTERNAL_EXCEPTION(TypeError) << "The size of parameters less than args_abs_list's size.";
  }
  for (size_t i = 0; i < args_abs_list.size(); i++) {
    auto parameter = params[i];
    MS_EXCEPTION_IF_NULL(parameter);
    oss << parameter->ToString() << ":<" << GetAbstractStr(args_abs_list[i]) << ">,";
  }
  oss << "]";
  oss << GetDebugInfoStr(graph->debug_info(), "", kSourceLineTipDiscard);
  return oss.str();
}

void DumpInferStack(std::ostringstream &oss) {
  auto graph_stack = GetCurrentGraphEvalStack();
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
      MS_LOG(INTERNAL_EXCEPTION) << "DumpInferStack failed, got null graph context";
    }
    auto graph = context->func_graph();
    if (graph == nullptr) {  // Top context.
      continue;
    }
    const auto &args_abs_list = context->args_abs_list();
    if (graph->parameters().size() < args_abs_list.size()) {
      continue;
    }
    oss << "    #" << index++ << " " << GetGraphParamString(graph, args_abs_list) << "\n";
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

class AnalyzeFailExporter {
 public:
  AnalyzeFailExporter() {
    is_top_graph_ = false;
    exported_.clear();
  }
  ~AnalyzeFailExporter() {}

  bool ExportFuncGraph(const std::string &filename, const TraceCNodeEvalStack &node_config_stack);

 protected:
  void OutputCNode(const CNodePtr &node, const FuncGraphPtr &sub_graph, const OrderedMap<AnfNodePtr, int32_t> &para_map,
                   const std::shared_ptr<SubGraphIRInfo> &gsub);
  AbstractBasePtr GetNodeAbstract(const AnfNodePtr &node);
  AnfNodeConfigPtr GetForwardConfig(const AnfNodeConfigPtr &cfg);
  void ProcessFuncGraphCall(const CNodePtr &node, std::string *const op_comment);
  mindspore::HashMap<FuncGraphPtr, TaggedNodeMap> CreateTaggedNodeMap(
    const std::vector<abstract::AnfNodeConfigPtr> &node_config_stack);
  void ExportOneFuncGraph(const FuncGraphPtr &func_graph, const TaggedNodeMap &tagged_cnodes_map,
                          std::ostringstream &oss, OrderedMap<AnfNodePtr, int32_t> *para_map, int32_t total_para);
  void OutputStatementComment(const CNodePtr &node, const FuncGraphPtr &func_graph, std::ostringstream &oss);
  std::string OuputIrStyleCNodes(const FuncGraphPtr &func_graph, const std::vector<AnfNodePtr> &nodes,
                                 const TaggedNodeMap &tagged_cnodes_map, int32_t total_para,
                                 OrderedMap<AnfNodePtr, int32_t> *para_map);
  std::string GetNodeType(const AnfNodePtr &node);

 private:
  AnalysisContextPtr current_context_ = nullptr;
  AnalysisEnginePtr engine_ = nullptr;
  OrderedMap<FuncGraphPtr, ParamIndexMap> exported_;
  bool is_top_graph_;
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

void PrintTupleNodeUsedFlagsDat(const abstract::AbstractSequencePtr &sequence_abs, std::ostringstream &buffer) {
  if (sequence_abs == nullptr || sequence_abs->sequence_nodes() == nullptr || sequence_abs->sequence_nodes()->empty()) {
    return;
  }

  buffer << ", sequence_nodes={";
  for (size_t i = 0; i < sequence_abs->sequence_nodes()->size(); ++i) {
    auto node = (*sequence_abs->sequence_nodes())[i].lock();
    if (node == nullptr) {
      MS_LOG(DEBUG) << "The node in sequence_nodes is free.";
      buffer << "node={<freed node>}";
    } else {
      buffer << "node={" << node->DebugString();
      auto flags = GetSequenceNodeElementsUseFlags(node);
      if (flags != nullptr) {
        buffer << ", elements_use_flags: {ptr: " << flags << ", value: " << (*flags) << "}";
      }
      buffer << "}";
    }
    if (i != sequence_abs->sequence_nodes()->size() - 1) {
      buffer << ", ";
    }
  }
  buffer << "}";
}

std::string GetNodeTypeOrigin(const AnfNodePtr &nd) {
  MS_EXCEPTION_IF_NULL(nd);
  ValuePtr tensor_value = nullptr;
  StringImmPtr ref_key = nullptr;
  abstract::AbstractSequencePtr sequence_abs = nullptr;
  auto abstract = nd->abstract();
  if (abstract != nullptr) {
    if (abstract->isa<abstract::AbstractTensor>()) {
      tensor_value = abstract->BuildValue();
    }
    if (auto ref_tensor = abstract->cast_ptr<abstract::AbstractRefTensor>(); ref_tensor != nullptr) {
      ref_key = dyn_cast<StringImm>(ref_tensor->ref_key_value());
    } else if (auto map_tensor = abstract->cast_ptr<abstract::AbstractMapTensor>(); map_tensor != nullptr) {
      ref_key = dyn_cast<StringImm>(map_tensor->ref_key_value());
    }
    sequence_abs = dyn_cast<abstract::AbstractSequence>(abstract);
  }
  abstract::BaseShapePtr shape = nd->Shape() == nullptr ? nullptr : dyn_cast<abstract::BaseShape>(nd->Shape());
  TypePtr type = dyn_cast<Type>(nd->Type());
  std::ostringstream oss;
  if ((shape != nullptr) && (type != nullptr)) {
    oss << "<" << type << ", " << shape->ToString();
    if (tensor_value != nullptr && tensor_value != kValueAny) {
      oss << ", value=...";
    }
    if (ref_key != nullptr) {
      oss << ", ref_key=" << ref_key->value();
    }
    PrintTupleNodeUsedFlagsDat(sequence_abs, oss);
    oss << ">";
  } else if (type != nullptr) {
    oss << "<" << type;
    if (tensor_value != nullptr && tensor_value != kValueAny) {
      oss << ", value=...";
    }
    if (ref_key != nullptr) {
      oss << ", ref_key=" << ref_key->value();
    }
    PrintTupleNodeUsedFlagsDat(sequence_abs, oss);
    oss << ">";
  } else {
    oss << "<null>";
  }
  return oss.str();
}

std::string AnalyzeFailExporter::GetNodeType(const AnfNodePtr &node) {
  if (current_context_ == nullptr) {
    return GetNodeTypeOrigin(node);
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
        for (const auto &arg : func->args_abs_list()) {
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

void AnalyzeFailExporter::OutputStatementComment(const CNodePtr &node, const FuncGraphPtr &func_graph,
                                                 std::ostringstream &oss) {
  if (node == nullptr) {
    return;
  }
  // Output type of each input argument
  auto &inputs = node->inputs();
  if (node != func_graph->get_return()) {
    if (inputs.size() > 1) {
      oss << "\n      : (";
      for (size_t i = 1; i < inputs.size(); ++i) {
        if (i != 1) {
          oss << ", ";
        }
        AnfNodePtr arg = inputs[i];
        oss << GetNodeType(arg);
      }
      oss << ")"
          << " -> "
          << "(" << GetNodeType(node) << ")";
    }
  } else {
    oss << "\n      : (" << GetNodeType(node) << ")";
  }
  // Output other comment, map the graph name to original representation(containing unicode character)
  oss << "\n";
  oss << "      #scope: (" << node->scope()->name() << ")";
}

void AnalyzeFailExporter::OutputCNode(const CNodePtr &node, const FuncGraphPtr &sub_graph,
                                      const OrderedMap<AnfNodePtr, int32_t> &para_map,
                                      const std::shared_ptr<SubGraphIRInfo> &gsub) {
  if (node != sub_graph->get_return()) {
    gsub->buffer << "  %" << gsub->local_var << "(" << node->ToString() << ")"
                 << " = ";
    gsub->local_var_map[node] = gsub->local_var++;
  } else {
    gsub->buffer << "  ";
  }

  if (node->inputs().empty()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Input of CNode is empty";
  }

  // Print operator
  DumpOperator(node, gsub);

  // Print operands
  DumpOperands(node, para_map, gsub);

  // Print operator attrs
  AnfNodePtr op = node->input(0);
  DumpOperateAttrs(op, gsub);

  // Print cnode attrs
  DumpCNodeAttrs(node, gsub);

  // Print cnode primal attrs
  DumpCNodePrimalAttrs(node, gsub);

  // Print parallel info
  DumpParallelInfo(node, gsub);

  // Process function graph call
  std::string op_comment;
  ProcessFuncGraphCall(node, &op_comment);
  if (!op_comment.empty()) {
    auto &inputs = node->inputs();
    std::shared_ptr<SubGraphIRInfo> input_gsub = std::make_shared<SubGraphIRInfo>();
    input_gsub->local_var = 0;
    DumpOperator(inputs[0], input_gsub);
    gsub->buffer << "    #" << input_gsub->buffer.str() << ".prototype = " << op_comment;
  }
  // Output comment
  OutputStatementComment(node, sub_graph, gsub->buffer);
  gsub->buffer << "\n";
}

std::string AnalyzeFailExporter::OuputIrStyleCNodes(const FuncGraphPtr &func_graph,
                                                    const std::vector<AnfNodePtr> &nodes,
                                                    const TaggedNodeMap &tagged_cnodes_map, int32_t total_para,
                                                    OrderedMap<AnfNodePtr, int32_t> *para_map) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto &parameters = func_graph->parameters();
  std::shared_ptr<SubGraphIRInfo> gsub = std::make_shared<SubGraphIRInfo>();
  gsub->local_var = 0;
  if (!is_top_graph_) {
    if (parameters.size() == 1) {
      MS_EXCEPTION_IF_NULL(parameters[0]);
      gsub->buffer << "%para" << (*para_map)[parameters[0]] << "_" << parameters[0]->ToString();
    } else if (parameters.size() > 1) {
      for (size_t idx = 0; idx < parameters.size() - 1; idx++) {
        MS_EXCEPTION_IF_NULL(parameters[idx]);
        gsub->buffer << "%para" << (*para_map)[parameters[idx]] << "_" << parameters[idx]->ToString();
        gsub->buffer << ", ";
      }
      MS_EXCEPTION_IF_NULL(parameters[parameters.size() - 1]);
      gsub->buffer << "%para" << (*para_map)[parameters[parameters.size() - 1]] << "_"
                   << parameters[parameters.size() - 1]->ToString();
    }
  } else {
    is_top_graph_ = false;
  }
  gsub->buffer << ") {\n";
  ParamIndexMap param_map;
  exported_[func_graph] = param_map;
  gsub->local_var = 0;
  for (size_t idx = 0; idx < parameters.size(); idx++) {
    MS_EXCEPTION_IF_NULL(parameters[idx]);
    if ((*para_map).count(parameters[idx]) == 0) {
      (*para_map)[parameters[idx]] = total_para++;
    }
  }
  for (const AnfNodePtr &node : nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (!tagged_cnodes_map.empty()) {
      auto iter = tagged_cnodes_map.find(node);
      if (iter != tagged_cnodes_map.end()) {
        gsub->buffer << "\n#------------------------> " << iter->second << "\n";
      }
    }
    OutputCNode(cnode, func_graph, *para_map, gsub);
    if (trace::GetGlobalTraceLabelType() == trace::TraceLabelType::kWithUniqueId) {
      gsub->buffer << trace::GetDebugInfoStr(cnode->debug_info(), "      # ", kSourceLineTipDiscard) << "#"
                   << trace::Label(cnode->debug_info()) << "\n";
    } else {
      std::string dgi = trace::GetDebugInfoStr(cnode->debug_info(), "      # ", kSourceLineTipDiscard);
      if (dgi != "") {
        gsub->buffer << trace::GetDebugInfoStr(cnode->debug_info(), "      # ", kSourceLineTipDiscard) << "\n";
      }
    }
  }
  return gsub->buffer.str();
}

void AnalyzeFailExporter::ExportOneFuncGraph(const FuncGraphPtr &func_graph, const TaggedNodeMap &tagged_cnodes_map,
                                             std::ostringstream &oss, OrderedMap<AnfNodePtr, int32_t> *para_map,
                                             int32_t total_para) {
  if (func_graph == nullptr) {
    return;
  }

  std::vector<AnfNodePtr> nodes = TopoSort(func_graph->get_return(), SuccIncoming, AlwaysInclude);

  if (*(func_graph->indirect())) {
    oss << "indirect: " << *(func_graph->indirect()) << "\n";
  }
  oss << "subgraph attr:" << std::endl;
  for (const auto &attr : func_graph->attrs()) {
    oss << attr.first << ": ";
    MS_EXCEPTION_IF_NULL(attr.second);
    if (attr.second->isa<BoolImm>()) {
      oss << GetValue<bool>(attr.second);
    } else if (attr.second->isa<StringImm>()) {
      oss << (GetValue<std::string>(attr.second));
    }
    oss << std::endl;
  }
  oss << "subgraph instance: " << func_graph->ToString() << " : " << func_graph.get() << std::endl;
  // Dump side effect info.
  auto effect_info = func_graph->GetEffectInfo();
  if (effect_info.HasEffect()) {
    oss << "# " << effect_info.ToString() << '\n';
  }
  if (trace::GetGlobalTraceLabelType() == trace::TraceLabelType::kWithUniqueId) {
    oss << trace::GetDebugInfoStr(func_graph->debug_info(), "# ", kSourceLineTipDiscard) << "#"
        << trace::Label(func_graph->debug_info()) << "\n";
  } else {
    oss << trace::GetDebugInfoStr(func_graph->debug_info(), "# ", kSourceLineTipDiscard) << "\n";
  }
  oss << "subgraph @" << func_graph->ToString();
  if (func_graph->parent() != nullptr) {
    oss << " parent: [subgraph @" << func_graph->parent()->ToString() << "]";
  }
  oss << "(";
  oss << OuputIrStyleCNodes(func_graph, nodes, tagged_cnodes_map, total_para, para_map);

  oss << "}\n";

  OutputOrderList(func_graph, oss);
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
  ofs << "# ===============================================================================================\n"
      << "# The following shows the last analyze fail log message.\n"
      << "# ===============================================================================================\n\n"
      << "----------------------------------------------------\n"
      << "- Caught exception:\n"
      << "----------------------------------------------------\n"
      << StaticAnalysisException::Instance().msg();

  ofs << "# ===============================================================================================\n"
      << "# The following shows the IR when the function graphs evaluation fails to help locate the problem.\n"
      << "# You can search the last ------------------------> to the node which is evaluated failure.\n"
      << "# Refer to https://www.mindspore.cn/search?inputValue=analyze_fail.ir to get more instructions.\n"
      << "# ===============================================================================================\n\n";

  if (engine_ == nullptr) {
    engine_ = node_config_stack.front()->engine();
  }
  auto top_func = node_config_stack.front()->func_graph();
  std::ostringstream head_buffer;
  MS_EXCEPTION_IF_NULL(top_func);
  is_top_graph_ = true;
  auto sub_graphs = top_func->func_graphs_used_total();
  std::ostringstream oss;
  DumpGlobalInfoEntry(top_func, oss, sub_graphs.size());
  OrderedMap<AnfNodePtr, int32_t> para_map;
  int32_t total_para = DumpParams(top_func, oss, &para_map);
  ofs << oss.str();
  ofs << std::endl;
  ofs << head_buffer.str();
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
    ExportOneFuncGraph(fg, tagged_func_graphs[fg], buffer, &para_map, total_para);
    ofs << buffer.str() << "\n\n";
  }
  current_context_ = nullptr;

  ofs << "# ===============================================================================================\n";
  ofs << "# The total of function graphs in evaluation stack: ";
  auto ignored_num = (node_config_stack.size() - printed_func_graphs.size());
  if (ignored_num == 0) {
    ofs << node_config_stack.size() << "\n";
  } else {
    ofs << printed_func_graphs.size() << "/" << node_config_stack.size() << " (Ignored " << ignored_num
        << " internal frames).\n";
  }
  ofs << "# ===============================================================================================\n";

  ofs << "\n\n# ===============================================================================================\n";
  ofs << "# The rest function graphs are the following:\n";
  ofs << "# ===============================================================================================\n";

  bool has_rest_fg = false;
  TaggedNodeMap empty_map;
  for (const auto &fg : top_func->func_graphs_used_total()) {
    if (!printed_func_graphs.emplace(fg).second) {
      continue;
    }
    std::ostringstream buffer;
    ExportOneFuncGraph(fg, empty_map, buffer, &para_map, total_para);
    ofs << buffer.str() << "\n\n";
    has_rest_fg = true;
  }
  if (!has_rest_fg) {
    ofs << "No more function graphs.\n\n";
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

  oss << "\n";
  int index = 0;
  int file_oss_index = 0;
  std::string last_location_info = "";
  const size_t max_stack_depth = 50;
  bool print_ellipsis = false;
  const size_t half = 2;
  if (stack.size() > max_stack_depth) {
    oss << "The depth of the current stack is " << stack.size() << ".\n"
        << "You can get more call stack information in analyze_fail.ir.\n\n";
  }
  std::ostringstream file_oss;
  file_oss << "\n";

  for (size_t i = 0; i < stack.size(); ++i) {
    auto node_config = stack[i];
    MS_EXCEPTION_IF_NULL(node_config);
    auto cnode = dyn_cast<CNode>(node_config->node());
    if (cnode == nullptr) {
      MS_LOG(DEBUG) << "CNode of elements[" << i << "] is nullptr.";
      continue;
    }

    auto debug_info = cnode->debug_info();
    auto this_location_info = trace::GetDebugInfoStr(debug_info);
    if (this_location_info.empty() || this_location_info == last_location_info) {
      continue;
    }
    last_location_info = this_location_info;
    file_oss << "# " << file_oss_index++ << " " << this_location_info;
    if ((i <= max_stack_depth / half) || (i >= (stack.size() - max_stack_depth / half))) {
      oss << "# " << index++ << " " << this_location_info;
    } else {
      if (!print_ellipsis) {
        print_ellipsis = true;
        oss << "......\n\n";
      }
      index++;
    }
  }
  bool empty_stack_info = oss.str() == "\n";

#ifndef ENABLE_SECURITY
  std::string msg =
    "\n----------------------------------------------------\n"
    "- The Traceback of Net Construct Code:\n"
    "----------------------------------------------------" +
    file_oss.str() + "\n";
  StaticAnalysisException::Instance().AppendMsg(msg);
  std::string file_name = GetEvalFailDatPath();
  auto ret = OutputAnalyzedGraphWithType(file_name);
  if (ret) {
    oss << " (See file '" << file_name
        << "' for more details. Get instructions about `analyze_fail.ir` at "
           "https://www.mindspore.cn/search?inputValue=analyze_fail.ir)";
  }
#endif
  stack.clear();
  MS_LOG(INFO) << "Get graph analysis information end";
  if (empty_stack_info) {
    oss.str("");
  }
}

// Trace the graph evaluator stack
thread_local TraceGraphEvalStack graph_infer_stack;
// Trace the cnode infer debug info
thread_local TraceCNodeEvalStack cnode_debug_stack{};

void TraceGraphEvalEnter(const abstract::AnalysisContextPtr &context, const abstract::AnfNodeConfigPtr &node) {
  if (context == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "GraphInferEnter got null context";
  }
  (void)graph_infer_stack.push_back(std::pair<abstract::AnalysisContextPtr, abstract::AnfNodeConfigPtr>(context, node));
}

void TraceGraphEvalLeave(const abstract::AnalysisContextPtr &context) {
  if (context == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "The context is null.";
  }
  if (graph_infer_stack.empty()) {
    MS_LOG(INTERNAL_EXCEPTION) << "The call stack is empty.";
  }
  if (context != graph_infer_stack.back().first) {
    MS_LOG(INTERNAL_EXCEPTION) << "Different context: " << context->func_graph()->ToString() << ", "
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
  static bool running = false;
  // Avoid to trace recursively
  if (running) {
    return;
  }
  running = true;
  try {
    TraceGraphEval();
    std::ostringstream trace_info;
    StaticAnalysisException::Instance().AppendMsg(oss.str());
    GetEvalStackInfo(trace_info);
    if (trace_info.str().empty()) {
      const DebugInfoPtr &debug_info = TraceManager::parser_debug_info();
      if (debug_info != nullptr && TraceManager::parser_debug_info_flag() == true) {
        auto debug_str = trace::GetTracedDebugInfoStr(debug_info);
        if (!debug_str.empty()) {
          std::ostringstream content;
          content << "\n\n" << debug_str;
          PrintMessage(oss, content.str(), add_title);
        }
      }
    } else {
      PrintMessage(oss, trace_info.str(), add_title);
    }
  } catch (...) {
    MS_LOG(INFO) << " Print trace information exception.";
  }
  running = false;
}

std::string GetTraceStackInfoStr(const AnfNodePtr &node, bool add_title) {
  if (node == nullptr) {
    return std::string();
  }
  if (node->debug_info() == nullptr) {
    return std::string();
  }
  auto debug_str = trace::GetTracedDebugInfoStr(node->debug_info());
  if (debug_str.empty()) {
    return std::string();
  }
  std::ostringstream oss;
  std::ostringstream content;
  content << "\n\n" << debug_str;
  PrintMessage(oss, content.str(), add_title);
  return oss.str();
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

// Register trace node stack provider.
struct GetTraceStrProviderRegister {
  GetTraceStrProviderRegister() noexcept { LogWriter::SetGetTraceStrProvider(GetTraceStackInfoStr); }
  ~GetTraceStrProviderRegister() = default;
} get_trace_str_provider_register;
}  // namespace trace
}  // namespace mindspore

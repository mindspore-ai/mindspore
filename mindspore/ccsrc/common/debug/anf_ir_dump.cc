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
#include "include/common/debug/anf_ir_dump.h"
#if defined(_WIN32) || defined(_WIN64)
#include <stdlib.h>
#endif
#include <fstream>
#include <iomanip>
#include "utils/label.h"
#include "utils/hash_map.h"
#include "ir/primitive.h"
#include "ir/func_graph.h"
#include "ir/graph_utils.h"
#include "utils/trace_base.h"
#include "utils/anf_utils.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/debug/anf_dump_utils.h"
#include "mindspore/core/utils/file_utils.h"

namespace mindspore {
void PrintTupleNodeUsedFlags(std::ostringstream &buffer, const abstract::AbstractSequencePtr &sequence_abs) {
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

void PrintNodeOutputType(std::ostringstream &buffer, const AnfNodePtr &node) {
  if (node == nullptr) {
    return;
  }

  ValuePtr tensor_value = nullptr;
  StringImmPtr ref_key = nullptr;
  abstract::AbstractSequencePtr sequence_abs = nullptr;
  auto abstract = node->abstract();
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

  abstract::BaseShapePtr shape = dyn_cast<abstract::BaseShape>(node->Shape());
  TypePtr type = dyn_cast<Type>(node->Type());
  if ((shape != nullptr) && (type != nullptr)) {
    buffer << "<" << type << ", " << shape->ToString();
    if (tensor_value != nullptr && tensor_value != kValueAny) {
      buffer << ", value=...";
    }
    if (ref_key != nullptr) {
      buffer << ", ref_key=:" << ref_key->value();
    }
    PrintTupleNodeUsedFlags(buffer, sequence_abs);
    buffer << ">";
  } else if (type != nullptr) {
    buffer << "<" << type;
    if (tensor_value != nullptr && tensor_value != kValueAny) {
      buffer << ", value=...";
    }
    if (ref_key != nullptr) {
      buffer << ", ref_key=:" << ref_key->value();
    }
    PrintTupleNodeUsedFlags(buffer, sequence_abs);
    buffer << ">";
  } else {
    buffer << "<null>";
  }
}

void PrintNodeInputType(std::ostringstream &buffer, const AnfNodePtr &node) {
  if (node == nullptr) {
    return;
  }

  const auto &inputs = GetInputs(node);
  size_t len = inputs.size();
  if (len > 1) {
    // Skip inputs[0] which is Primitive value node
    for (size_t i = 1; i < len; ++i) {
      AnfNodePtr in = inputs[i];
      if (i != 1) {
        buffer << ", ";
      }
      PrintNodeOutputType(buffer, in);
    }
  }
}

void GatherInputAndOutputInferType(std::ostringstream &buffer, const AnfNodePtr &node) {
  buffer << "      : (";
  PrintNodeInputType(buffer, node);
  buffer << ") -> (";
  PrintNodeOutputType(buffer, node);
  buffer << ")";
}

void DumpGlobalInfoEntry(const FuncGraphPtr &graph, std::ostringstream &buffer,
                         const OrderedMap<FuncGraphPtr, std::shared_ptr<SubGraphIRInfo>> &sub_graphs) {
  if (graph == nullptr) {
    return;
  }

  buffer << "#IR entry      : @" << graph->ToString() << std::endl;
  buffer << "#Total subgraph: " << sub_graphs.size() << std::endl;
  buffer << std::endl;
  buffer << "#attrs         :" << std::endl;
  for (const auto &attr : graph->attrs()) {
    buffer << attr.first << " : ";
    if (attr.second->isa<BoolImm>()) {
      buffer << GetValue<bool>(attr.second);
    } else if (attr.second->isa<StringImm>()) {
      buffer << GetValue<std::string>(attr.second);
    }
    buffer << std::endl;
  }
}

void DumpKernelObjectType(const CNodePtr &node, const std::shared_ptr<SubGraphIRInfo> &gsub) {
  auto inputs_str = AnfDumpHandler::PrintInputKernelObjectTypes(node);
  auto outputs_str = AnfDumpHandler::PrintOutputKernelObjectTypes(node);
  if (inputs_str.empty() && outputs_str.empty()) {
    return;
  }
  gsub->buffer << "      : (" << inputs_str << ") -> (" << outputs_str << ")" << std::endl;
}

void DumpKernelInfo(const CNodePtr &node, const std::shared_ptr<SubGraphIRInfo> &gsub) {
  if (node == nullptr || gsub == nullptr) {
    return;
  }
  auto kernel_info = node->kernel_info();
  if (kernel_info == nullptr || !kernel_info->has_build_info()) {
    return;
  }
  if (!AnfUtils::IsRealKernel(node)) {
    DumpKernelObjectType(node, gsub);
    return;
  }

  gsub->buffer << "      : (";
  gsub->buffer << AnfDumpHandler::PrintInputTypeShapeFormat(node);
  gsub->buffer << ") -> (";
  gsub->buffer << AnfDumpHandler::PrintOutputTypeShapeFormat(node);
  gsub->buffer << ")";
  gsub->buffer << std::endl;
  DumpKernelObjectType(node, gsub);
}

int32_t DumpParams(const FuncGraphPtr &graph, std::ostringstream &buffer, OrderedMap<AnfNodePtr, int32_t> *para_map) {
  if (graph == nullptr) {
    MS_LOG(INFO) << "Parameter \'graph\' should not be null.";
    return 0;
  }
  std::vector<AnfNodePtr> parameters = graph->parameters();
  buffer << "#Total params  : " << parameters.size() << std::endl;
  buffer << std::endl;

  // Dump parameters
  int32_t para_num = 1;
  for (const auto &param : parameters) {
    if (param == nullptr) {
      continue;
    }
    auto parameter_ptr = param->cast<ParameterPtr>();
    if (parameter_ptr == nullptr) {
      MS_LOG(EXCEPTION) << "param cannot cast to ParameterPtr";
    }
    buffer << "%para" << para_num << "_" << parameter_ptr->name() << " : ";
    // Print parameters' type and shape
    PrintNodeOutputType(buffer, param);
    if (parameter_ptr->has_default()) {
      buffer << "  :  has_default";
    }
    auto kernel_info = param->kernel_info();
    if (kernel_info != nullptr && kernel_info->has_build_info()) {
      buffer << "  :  ";
      buffer << AnfDumpHandler::PrintOutputTypeShapeFormat(param);
      buffer << "  :  IsWeight: " << std::boolalpha << common::AnfAlgo::IsParameterWeight(parameter_ptr);
    }
    buffer << std::endl;

    if (para_map != nullptr) {
      (*para_map)[param] = para_num++;
    }
    MS_LOG(DEBUG) << "Record param: " << param->ToString() << " graph belong : " << param->func_graph()->ToString();
  }
  return para_num;
}

void DumpOperator(const AnfNodePtr &node, const std::shared_ptr<SubGraphIRInfo> &gsub) {
  if (gsub == nullptr) {
    MS_LOG(INFO) << "Parameter \'gsub\' should not be null.";
    return;
  }
  auto cnode = dyn_cast<CNode>(node);
  if (cnode == nullptr) {
    MS_LOG(EXCEPTION) << "Parameter \'node\' should be a CNode";
  }
  AnfNodePtr op = cnode->input(0);
  MS_EXCEPTION_IF_NULL(op);
  if (IsValueNode<FuncGraph>(op)) {
    FuncGraphPtr fg = GetValueNode<FuncGraphPtr>(op);
    if (fg != nullptr) {
      gsub->buffer << "call @" << fg->ToString();
    }
  } else if (op->isa<CNode>()) {
    std::string func_str = GetNodeFuncStr(op);
    if (gsub->local_var_map.find(op) != gsub->local_var_map.end()) {
      gsub->buffer << "%" << gsub->local_var_map[op];
    } else {
      auto input = op->cast<CNodePtr>();
      auto fg = input->func_graph();
      gsub->buffer << "$(@" << fg->ToString() << ":" << input->ToString() << ")";
    }
    if (!func_str.empty()) {
      gsub->buffer << "[@" << func_str << "]";
    }
  } else if (op->isa<ValueNode>()) {
    auto value = GetValueNode(op);
    if (value != nullptr) {
      gsub->buffer << value->ToString();
    }
  } else {
    // It's Parameter.
    if (op->func_graph() != nullptr && op->func_graph() != node->func_graph()) {
      gsub->buffer << "$(@" << op->func_graph()->ToString() << ":";
    }
    gsub->buffer << op->ToString();
    if (op->func_graph() != nullptr && op->func_graph() != node->func_graph()) {
      gsub->buffer << ")";
    }
    std::string func_str = GetNodeFuncStr(op);
    if (!func_str.empty()) {
      gsub->buffer << "[@" << func_str << "]";
    }
  }
}

void DumpParamterInOperand(const AnfNodePtr &node, const AnfNodePtr &in,
                           const OrderedMap<AnfNodePtr, int32_t> &para_map,
                           const std::shared_ptr<SubGraphIRInfo> &gsub) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(node->func_graph());
  MS_EXCEPTION_IF_NULL(in);
  MS_EXCEPTION_IF_NULL(gsub);
  if (in->func_graph() == nullptr) {
    MS_LOG(ERROR) << "Parameter should belong to a func graph. Check func graph: " << node->func_graph();
  }
  if (in->func_graph() != nullptr && in->func_graph() != node->func_graph()) {
    gsub->buffer << "$(@" << in->func_graph()->ToString() << ":";
  } else {
    gsub->buffer << "%";
  }
  auto iter = para_map.find(in);
  if (iter == para_map.end()) {
    gsub->buffer << "para_" << in->ToString();
  } else {
    gsub->buffer << "para" << iter->second << "_" << in->ToString();
  }
  if (in->func_graph() != nullptr && in->func_graph() != node->func_graph()) {
    gsub->buffer << ")";
  }
}

void DumpOperands(const AnfNodePtr &node, const OrderedMap<AnfNodePtr, int32_t> &para_map,
                  const std::shared_ptr<SubGraphIRInfo> &gsub) {
  if (node == nullptr || gsub == nullptr) {
    return;
  }

  gsub->buffer << "(";
  const auto &inputs = GetInputs(node);
  size_t len = inputs.size();
  if (len > 1) {
    // Skip inputs[0] which is Primitive valuenode
    for (size_t i = 1; i < len; ++i) {
      AnfNodePtr in = inputs[i];
      MS_EXCEPTION_IF_NULL(in);
      if (i != 1) {
        gsub->buffer << ", ";
      }
      if (in->isa<Parameter>()) {
        DumpParamterInOperand(node, in, para_map, gsub);
      } else if (in->isa<CNode>()) {
        auto iter = gsub->local_var_map.find(in);
        if (iter != gsub->local_var_map.end()) {
          gsub->buffer << "%" << iter->second;
        } else {
          auto input = in->cast<CNodePtr>();
          auto fg = input->func_graph();
          gsub->buffer << "$(@" << fg->ToString() << ":" << input->ToString() << ")";
        }
      } else if (in->isa<ValueNode>() && !IsValueNode<FuncGraph>(in)) {
        // ValueNode except FuncGraph.
        gsub->buffer << GetValueNode(in)->ToString();
      } else if (IsValueNode<FuncGraph>(in)) {
        FuncGraphPtr fg = GetValueNode<FuncGraphPtr>(in);
        gsub->buffer << "@" << fg->ToString();
      } else if (AnfUtils::IsCustomActorNode(in)) {
        gsub->buffer << "%" << AnfUtils::GetCustomActorName(in);
      } else {
        gsub->buffer << in->ToString();
      }
    }
  }
  gsub->buffer << ")";
}

void DumpParallelInfo(const CNodePtr &node, const std::shared_ptr<SubGraphIRInfo> &gsub) {
  if ((node == nullptr) || (gsub == nullptr)) {
    return;
  }

  ValuePtr in_tmp = AnfDumpHandler::InStrategyValue(node);
  if (in_tmp == nullptr) {
    return;
  }
  gsub->buffer << " {in_strategy: ";
  gsub->buffer << in_tmp->ToString();

  ValuePtr out_tmp = AnfDumpHandler::OutStrategyValue(node);
  if (out_tmp != nullptr) {
    gsub->buffer << ", out_strategy: ";
    gsub->buffer << out_tmp->ToString();
  }

  gsub->buffer << "}";
}

void DumpAttrs(const mindspore::HashMap<std::string, ValuePtr> &attrs, const std::shared_ptr<SubGraphIRInfo> &gsub,
               bool check_strategy = false) {
  int i = 0;
  for (const auto &attr : attrs) {
    if (check_strategy && attr.first == PARALLEL_STRATEGY) {
      continue;  // Skip the strategy
    }
    if (i++ != 0) {
      gsub->buffer << ", ";
    }
    gsub->buffer << attr.first << ": ";
    if (attr.second == nullptr) {
      gsub->buffer << "null";
    } else {
      gsub->buffer << attr.second->ToString();
    }
  }
}

void DumpOperateAttrs(const AnfNodePtr &op, const std::shared_ptr<SubGraphIRInfo> &gsub) {
  if (op == nullptr || gsub == nullptr) {
    return;
  }

  if (IsValueNode<Primitive>(op)) {
    PrimitivePtr primitive = GetValueNode<PrimitivePtr>(op);
    if (!primitive->instance_name().empty()) {
      gsub->buffer << " {";
      gsub->buffer << "instance name"
                   << ": ";
      gsub->buffer << primitive->instance_name();
      gsub->buffer << "}";
    }
    auto attrs = primitive->attrs();
    if (!attrs.empty()) {
      gsub->buffer << " primitive_attrs: {";
      DumpAttrs(attrs, gsub, true);
      gsub->buffer << "}";
    }
  }
}

void DumpCNodeAttrs(const CNodePtr &op, const std::shared_ptr<SubGraphIRInfo> &gsub) {
  if (op == nullptr || gsub == nullptr) {
    return;
  }
  if (op->attrs().empty()) {
    return;
  }

  auto attrs = op->attrs();
  gsub->buffer << " cnode_attrs: {";
  DumpAttrs(attrs, gsub);
  gsub->buffer << "}";
}

void DumpCNodePrimalAttrs(const CNodePtr &op, const std::shared_ptr<SubGraphIRInfo> &gsub) {
  if (op == nullptr || gsub == nullptr) {
    return;
  }
  if (op->primal_attrs().empty()) {
    return;
  }
  auto primal_attrs = op->primal_attrs();
  gsub->buffer << " cnode_primal_attrs: {";
  DumpAttrs(primal_attrs, gsub);
  gsub->buffer << "}";
}

void DumpShape(const AnfNodePtr &node, const FuncGraphPtr &sub_graph, const std::shared_ptr<SubGraphIRInfo> &gsub) {
  if (node == nullptr || sub_graph == nullptr || gsub == nullptr) {
    return;
  }

  gsub->buffer << std::endl;
  if (node != sub_graph->get_return()) {
    gsub->buffer << "      : (";
    PrintNodeInputType(gsub->buffer, node);
    gsub->buffer << ") -> (";
    PrintNodeOutputType(gsub->buffer, node);
    gsub->buffer << ")";
  } else {
    gsub->buffer << "      : (";
    PrintNodeInputType(gsub->buffer, node);
    gsub->buffer << ")";
  }

  gsub->buffer << std::endl;
}

void DumpLocationInCurrentScope(const DebugInfoPtr &debug_info, const std::shared_ptr<SubGraphIRInfo> &gsub) {
  auto dump_debug_info = debug_info;
  std::list<DebugInfoPtr> need_dump_debug_infos;
  while (dump_debug_info != nullptr) {
    need_dump_debug_infos.push_front(dump_debug_info);
    if (dump_debug_info->trace_info() == nullptr) {
      break;
    }
    dump_debug_info = dump_debug_info->trace_info()->debug_info();
  }
  HashSet<std::string> visited_locations;
  for (const auto &cur_debug_info : need_dump_debug_infos) {
    if (cur_debug_info->location() != nullptr) {
      auto prefix = cur_debug_info->inlined() ? "      # inlined:" : "      # ";
      auto debug_info_str = trace::GetDebugInfo(cur_debug_info, "", kSourceLineTipDiscard);
      if (visited_locations.find(debug_info_str) == visited_locations.cend()) {
        gsub->buffer << prefix << debug_info_str << "\n";
        (void)visited_locations.insert(debug_info_str);
      }
    }
  }
}

void DumpPrimalDebugInfos(const CNodePtr &node, const std::shared_ptr<SubGraphIRInfo> &gsub) {
  MS_EXCEPTION_IF_NULL(node);
  auto primal_debug_infos = node->primal_debug_infos();
  if (!primal_debug_infos.empty()) {
    for (const auto &primal_debug_info : primal_debug_infos) {
      std::string lines;
      auto debug_info_str = trace::GetDebugInfo(primal_debug_info, "      # ", kSourceLineTipDiscard);
      if (!debug_info_str.empty()) {
        lines += debug_info_str + "\n";
      }
      gsub->buffer << "      # Corresponding forward node candidate:\n";
      if (!lines.empty()) {
        gsub->buffer << lines;
      }
    }
  }
}

void DumpDebugInfo(const CNodePtr &node, const std::shared_ptr<SubGraphIRInfo> &gsub,
                   const LocDumpMode &dump_location) {
  MS_EXCEPTION_IF_NULL(node);
  if (dump_location == kTopStack) {
    auto fused_debug_infos = node->fused_debug_infos();
    if (!fused_debug_infos.empty()) {
      for (const auto &debug_info : fused_debug_infos) {
        std::string lines;
        gsub->buffer << "      # Corresponding code candidate:\n";
        auto debug_info_str = trace::GetDebugInfo(debug_info, "      # ", kSourceLineTipDiscard);
        if (!debug_info_str.empty()) {
          lines += debug_info_str + "\n";
        }
        if (!lines.empty()) {
          gsub->buffer << lines;
        }
      }
    } else {
      auto debug_info_str = trace::GetDebugInfo(node->debug_info(), "      # ", kSourceLineTipDiscard);
      if (!debug_info_str.empty()) {
        gsub->buffer << debug_info_str << "\n";
      }
    }
    DumpPrimalDebugInfos(node, gsub);
  } else if (dump_location == kWholeStack) {
    auto fused_debug_infos = node->fused_debug_infos();
    if (!fused_debug_infos.empty()) {
      for (const auto &debug_info : fused_debug_infos) {
        gsub->buffer << "      # Corresponding code candidate:\n";
        DumpLocationInCurrentScope(debug_info, gsub);
      }
    } else {
      DumpLocationInCurrentScope(node->debug_info(), gsub);
    }
    // Print whole stack primal infos
    auto primal_debug_infos = node->primal_debug_infos();
    if (!primal_debug_infos.empty()) {
      for (const auto &primal_debug_info : primal_debug_infos) {
        gsub->buffer << "      # Corresponding forward node candidate:\n";
        DumpLocationInCurrentScope(primal_debug_info, gsub);
      }
    }
  }
}

void DumpCNode(const CNodePtr &node, const FuncGraphPtr &sub_graph, const OrderedMap<AnfNodePtr, int32_t> &para_map,
               const std::shared_ptr<SubGraphIRInfo> &gsub, bool dump_full_name, LocDumpMode dump_location) {
  if (node == nullptr || sub_graph == nullptr || gsub == nullptr) {
    return;
  }

  if (node != sub_graph->get_return()) {
    gsub->buffer << "  %" << gsub->local_var << "(" << node->ToString() << ")"
                 << " = ";
    gsub->local_var_map[node] = gsub->local_var++;
  } else {
    gsub->buffer << "  ";
  }

  if (node->inputs().empty()) {
    MS_LOG(EXCEPTION) << "Input of apply node is empty";
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

  // Print shape info
  DumpShape(node, sub_graph, gsub);

  // Print kernel info
  DumpKernelInfo(node, gsub);

  if (dump_full_name) {
    gsub->buffer << "      # fullname_with_scope: (" << node->fullname_with_scope() << ")" << std::endl;
  } else {
    gsub->buffer << "      # scope: (" << node->scope()->name() << ")" << std::endl;
  }

  // Print debug info
  DumpDebugInfo(node, gsub, dump_location);
}

void OutputOrderList(const FuncGraphPtr &sub_graph, std::ostringstream &oss) {
  auto &order_list = sub_graph->order_list();
  if (order_list.empty()) {
    return;
  }
  constexpr int width = 4;
  oss << "# order:\n";
  int i = 1;
  for (auto &node : order_list) {
    MS_EXCEPTION_IF_NULL(node);
    oss << '#' << std::setw(width) << i << ": " << node->DebugString() << '\n';
    ++i;
  }
}

void DumpIRInSubgraph(const std::vector<AnfNodePtr> &nodes, OrderedMap<AnfNodePtr, int32_t> *para_map,
                      OrderedMap<FuncGraphPtr, std::shared_ptr<SubGraphIRInfo>> *const sub_graphs, int32_t total_para,
                      bool dump_full_name = false, LocDumpMode dump_location = kOff) {
  if (para_map == nullptr || sub_graphs == nullptr) {
    return;
  }

  for (const auto &node : nodes) {
    MS_EXCEPTION_IF_NULL(node);
    FuncGraphPtr sub_graph = node->func_graph();
    if (sub_graph == nullptr) {
      MS_LOG(DEBUG) << "Node[" << node->ToString() << "] belongs to no graph!";
      continue;
    }
    std::shared_ptr<SubGraphIRInfo> gsub = (*sub_graphs)[sub_graph];
    if (gsub == nullptr) {
      gsub = std::make_shared<SubGraphIRInfo>();
      gsub->local_var = 0;
      (*sub_graphs)[sub_graph] = gsub;
    }
    std::vector<AnfNodePtr> parameters = sub_graph->parameters();
    for (size_t idx = 0; idx < parameters.size(); idx++) {
      MS_EXCEPTION_IF_NULL(parameters[idx]);
      if ((*para_map).count(parameters[idx]) == 0) {
        (*para_map)[parameters[idx]] = total_para++;
      }
    }
    if (!node->isa<Parameter>()) {
      if (node->isa<CNode>()) {
        // Print and record output of operator if it is not 'Return'
        DumpCNode(node->cast<CNodePtr>(), sub_graph, *para_map, gsub, dump_full_name, dump_location);
      } else if (AnfUtils::IsCustomActorNode(node)) {
        continue;
      } else {
        gsub->buffer << "  " << node->ToString() << std::endl;
      }
    }
  }
}

void DumpSubgraph(const OrderedMap<FuncGraphPtr, std::shared_ptr<SubGraphIRInfo>> *sub_graphs,
                  const FuncGraphPtr &graph, OrderedMap<AnfNodePtr, int32_t> *para_map, std::ostringstream &oss) {
  if (sub_graphs == nullptr || graph == nullptr) {
    return;
  }
  for (const auto &sg : *sub_graphs) {
    MS_EXCEPTION_IF_NULL(sg.first);
    if (*(sg.first->switch_input())) {
      oss << "switch_input: " << *(sg.first->switch_input()) << "\n";
    }
    if (*(sg.first->switch_layer_input())) {
      oss << "switch_layer_input: " << *(sg.first->switch_layer_input()) << "\n";
    }
    oss << "subgraph attr:" << std::endl;
    for (const auto &attr : sg.first->attrs()) {
      oss << attr.first << " : ";
      if (attr.second->isa<BoolImm>()) {
        oss << GetValue<bool>(attr.second);
      } else if (attr.second->isa<StringImm>()) {
        oss << (GetValue<std::string>(attr.second));
      }
      oss << std::endl;
    }
    oss << "subgraph instance: " << sg.first->ToString() << " : " << sg.first.get() << std::endl;
    if (label_manage::GetGlobalTraceLabelType() == label_manage::TraceLabelType::kWithUniqueId) {
      oss << trace::GetDebugInfo(sg.first->debug_info(), "# ", kSourceLineTipDiscard) << "#"
          << label_manage::Label(sg.first->debug_info()) << "\n";
    } else {
      oss << trace::GetDebugInfo(sg.first->debug_info(), "# ", kSourceLineTipDiscard) << "\n";
    }
    oss << "subgraph @" << sg.first->ToString();
    if (sg.first->manager() != nullptr && sg.first->parent() != nullptr) {
      oss << " parent: [subgraph @" << sg.first->parent()->ToString() << "]";
    }
    oss << "(";
    if (sg.first != graph) {
      std::vector<AnfNodePtr> parameters = sg.first->parameters();
      if (parameters.size() == 1) {
        MS_EXCEPTION_IF_NULL(parameters[0]);
        oss << "%para" << (*para_map)[parameters[0]] << "_" << parameters[0]->ToString();
      } else if (parameters.size() > 1) {
        for (size_t idx = 0; idx < parameters.size() - 1; idx++) {
          MS_EXCEPTION_IF_NULL(parameters[idx]);
          oss << "%para" << (*para_map)[parameters[idx]] << "_" << parameters[idx]->ToString();
          oss << ", ";
        }
        MS_EXCEPTION_IF_NULL(parameters[parameters.size() - 1]);
        oss << "%para" << (*para_map)[parameters[parameters.size() - 1]] << "_"
            << parameters[parameters.size() - 1]->ToString();
      }
    }
    oss << ") {" << std::endl;
    MS_EXCEPTION_IF_NULL(sg.second);
    oss << sg.second->buffer.str();
    oss << "}" << std::endl;
    OutputOrderList(sg.first, oss);
    oss << std::endl;
    oss << std::endl;
  }
}

void SetDumpConfigByString(const std::string &str, DumpConfig *dump_config) {
  MS_LOG(INFO) << "Set dump config:" << str;
  static mindspore::HashMap<std::string, enum LocDumpMode> dump_level_map = {
    {kDumpConfigLineLevel0, kOff}, {kDumpConfigLineLevel1, kTopStack}, {kDumpConfigLineLevel2, kWholeStack}};
  auto it = dump_level_map.find(str);
  if (it != dump_level_map.end()) {
    dump_config->dump_line_level = it->second;
    return;
  }
  if (str == kDumpConfigDisableBackend) {
    dump_config->disable_backend_dump = true;
    return;
  }
  if (str == kDumpConfigEnablePassIR) {
    dump_config->enable_dump_pass_ir = true;
    return;
  }
}

std::shared_ptr<OrderedSet<std::string>> GetAllConfigStrings(const std::string &config_full_string) {
  size_t start_pos = 0;
  auto config_strings = std::make_shared<OrderedSet<std::string>>();
  // if '#' is the last char of str, the str is legal, so we use '<=' but not '<'.
  while (start_pos <= config_full_string.size()) {
    auto pos = config_full_string.find('#', start_pos);
    if (pos == std::string::npos) {
      pos = config_full_string.size();
    }
    auto substr = config_full_string.substr(start_pos, pos - start_pos);
    // Skip the '#'
    start_pos = pos + 1;
    if (substr.empty()) {
      continue;
    }
    (void)config_strings->insert(substr);
  }
  return config_strings;
}

bool ConfigsAreLegal(const std::shared_ptr<OrderedSet<std::string>> &config_strings) {
  // Value 'int' is used to mark config group id
  HashMap<std::string, int> config_white_list = {{kDumpConfigLineLevel0, 0},
                                                 {kDumpConfigLineLevel1, 0},
                                                 {kDumpConfigLineLevel2, 0},
                                                 {kDumpConfigDisableBackend, 1},
                                                 {kDumpConfigEnablePassIR, 2}};
  // Key 'int' is config group id, value is the config.
  HashMap<int, std::string> config_groups;
  for (const auto &config_string : *config_strings) {
    auto config_white_list_it = config_white_list.find(config_string);
    if (config_white_list_it == config_white_list.end()) {
      std::ostringstream buffer;
      buffer << "Support configs:\n"
             << "[0]: " << kDumpConfigLineLevel0 << "\n"
             << "[1]: " << kDumpConfigLineLevel1 << "\n"
             << "[2]: " << kDumpConfigLineLevel2 << "\n"
             << "[3]: " << kDumpConfigDisableBackend << "\n"
             << "[4]: " << kDumpConfigEnablePassIR;
      MS_LOG(WARNING) << "Illegal dump config:\n" << config_string << "\n" << buffer.str();
      return false;
    }
    auto group_id = config_white_list_it->second;
    // Check conflict configs.
    auto config_groups_it = config_groups.find(group_id);
    if (config_groups_it != config_groups.end()) {
      const auto &record_config = config_groups_it->second;
      MS_LOG(WARNING) << "Dump configs are conflict. Conflict configs: [" << record_config << "] and [" << config_string
                      << "].\n"
                      << "Please keep only one of them.";
      return false;
    }
    config_groups[group_id] = config_string;
  }
  return true;
}

DumpConfig GetDumpConfig() {
  static DumpConfig dump_config = DumpConfig();
  static bool parsed = false;
  if (parsed) {
    return dump_config;
  }
  parsed = true;
  // Start parse config.
  std::string str(common::GetEnv("MS_DEV_DUMP_IR_CONFIG"));
  auto constexpr max_string_len = 100;
  if (str.size() > max_string_len) {
    MS_LOG(WARNING) << "Dump ir config length exceed max length: " << max_string_len;
    return dump_config;
  }
  if (str.empty()) {
    return dump_config;
  }
  auto config_strings = GetAllConfigStrings(str);
  if (!ConfigsAreLegal(config_strings)) {
    return dump_config;
  }
  for (const auto &config : *config_strings) {
    SetDumpConfigByString(config, &dump_config);
  }
  return dump_config;
}

void GetEnvDumpIrLineLevel(LocDumpMode *dump_location) {
  const auto &config = GetDumpConfig();
  if (config.dump_line_level != kInValid) {
    *dump_location = config.dump_line_level;
  }
}

#ifdef ENABLE_DUMP_IR
void DumpIR(const std::string &filename, const FuncGraphPtr &graph, bool dump_full_name, LocDumpMode dump_location,
            const std::string &target_file) {
  GetEnvDumpIrLineLevel(&dump_location);
  if (graph == nullptr) {
    return;
  }
  auto path = GetSaveGraphsPathName(Common::AddId(filename, ".ir"));
  if (!target_file.empty()) {
    path = target_file;
  }
  auto realpath = Common::CreatePrefixPath(path);
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Get real path failed, path=" << path;
    return;
  }

  ChangeFileMode(realpath.value(), S_IWUSR);
  std::ofstream fout(realpath.value());
  std::ostringstream oss;
  std::ostringstream buffer;
  if (!fout.is_open()) {
    MS_LOG(ERROR) << "Open dump file '" << realpath.value() << "' failed!" << ErrnoToString(errno);
    return;
  }

  auto nodes = TopoSort(graph->get_return(), SuccDeeperSimple, AlwaysInclude);
  OrderedMap<AnfNodePtr, int32_t> para_map;
  // Dump global info
  int32_t total_para = DumpParams(graph, oss, &para_map);

  OrderedMap<FuncGraphPtr, std::shared_ptr<SubGraphIRInfo>> sub_graphs;
  // Dump ir in each sub graph
  DumpIRInSubgraph(nodes, &para_map, &sub_graphs, total_para, dump_full_name, dump_location);

  DumpGlobalInfoEntry(graph, buffer, sub_graphs);
  buffer << oss.str();
  // Output global info
  fout << buffer.str() << std::endl;
  buffer.str(std::string());
  buffer.clear();

  // Output each sub graph
  DumpSubgraph(&sub_graphs, graph, &para_map, buffer);
  fout << buffer.str();

  fout.close();
  // Set file mode to read only by user
  ChangeFileMode(realpath.value(), S_IRUSR);
}

void DumpIR(std::ostringstream &graph_buffer, const FuncGraphPtr &graph, bool dump_full_name,
            LocDumpMode dump_location) {
  GetEnvDumpIrLineLevel(&dump_location);
  if (graph == nullptr) {
    return;
  }
  std::ostringstream oss;
  auto nodes = TopoSort(graph->get_return(), SuccDeeperSimple, AlwaysInclude);
  OrderedMap<AnfNodePtr, int32_t> para_map;
  int32_t total_para = DumpParams(graph, oss, &para_map);

  graph_buffer << "\n";

  OrderedMap<FuncGraphPtr, std::shared_ptr<SubGraphIRInfo>> sub_graphs;
  // Dump ir in each sub graph
  DumpIRInSubgraph(nodes, &para_map, &sub_graphs, total_para, dump_full_name, dump_location);

  // Dump global info
  DumpGlobalInfoEntry(graph, graph_buffer, sub_graphs);
  graph_buffer << oss.str();
  // Output each sub graph
  DumpSubgraph(&sub_graphs, graph, &para_map, graph_buffer);
}

void DumpIRForRDR(const std::string &filename, const FuncGraphPtr &graph, bool dump_full_name,
                  LocDumpMode dump_location) {
  GetEnvDumpIrLineLevel(&dump_location);
  if (graph == nullptr) {
    return;
  }
  auto path = Common::AddId(filename, ".ir");
  auto realpath = Common::CreatePrefixPath(path);
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Get real path failed. path=" << path;
    return;
  }

  ChangeFileMode(realpath.value(), S_IWUSR);
  std::ofstream fout(realpath.value());
  std::ostringstream buffer;
  if (!fout.is_open()) {
    MS_LOG(ERROR) << "Open dump file '" << realpath.value() << "' failed!" << ErrnoToString(errno);
    return;
  }

  auto nodes = TopoSort(graph->get_return(), SuccDeeperSimple, AlwaysInclude);
  OrderedMap<AnfNodePtr, int32_t> para_map;
  int32_t total_para = DumpParams(graph, buffer, &para_map);
  OrderedMap<FuncGraphPtr, std::shared_ptr<SubGraphIRInfo>> sub_graphs;
  // Dump ir in each sub graph
  DumpIRInSubgraph(nodes, &para_map, &sub_graphs, total_para, dump_full_name, dump_location);
  // Dump global info
  DumpGlobalInfoEntry(graph, buffer, sub_graphs);
  // Output global info
  fout << buffer.str() << std::endl;
  buffer.str(std::string());
  buffer.clear();

  // Output each sub graph
  DumpSubgraph(&sub_graphs, graph, &para_map, buffer);
  fout << buffer.str();

  fout.close();
  // Set file mode to read only by user
  ChangeFileMode(realpath.value(), S_IRUSR);
}

#else
void DumpIR(const std::string &, const FuncGraphPtr &, bool, LocDumpMode, const std::string &) {
  static bool already_printed = false;
  if (already_printed) {
    return;
  }
  already_printed = true;
  MS_LOG(WARNING) << "The functionality of dumping function graph IR is disabled, "
                  << "please recompile source to enable it. See help of building script.";
}

void DumpIR(std::ostringstream &, const FuncGraphPtr &, bool, LocDumpMode) {
  static bool already_printed = false;
  if (already_printed) {
    return;
  }
  already_printed = true;
  MS_LOG(WARNING) << "The functionality of dumping function graph IR is disabled, "
                  << "please recompile source to enable it. See help of building script.";
}

void DumpIRForRDR(const std::string &, const FuncGraphPtr &, bool, LocDumpMode) {
  static bool already_printed = false;
  if (already_printed) {
    return;
  }
  already_printed = true;
  MS_LOG(WARNING) << "The functionality of dumping function graph IR is disabled, "
                  << "please recompile source to enable it. See help of building script.";
}
#endif
}  // namespace mindspore

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

#include "pipeline/jit/debug/anf_ir_utils.h"

#include <fstream>
#include <map>
#include <memory>
#include <algorithm>
#include <iomanip>
#include "utils/hash_map.h"
#include "ir/graph_utils.h"
#include "utils/symbolic.h"
#include "ir/meta_func_graph.h"
#include "ir/param_info.h"
#include "pybind_api/ir/tensor_py.h"
#include "include/common/utils/python_adapter.h"
#include "pipeline/jit/parse/resolve.h"
#include "frontend/operator/composite/composite.h"
#include "frontend/operator/composite/vmap.h"
#include "frontend/operator/composite/map.h"
#include "frontend/operator/graph_bprop/bprop_meta_func_graph.h"
#include "utils/ordered_map.h"
#include "utils/ordered_set.h"
#include "include/common/utils/utils.h"
#include "utils/shape_utils.h"
#include "pipeline/jit/debug/trace.h"
#include "utils/label.h"
#include "utils/ms_context.h"
#include "frontend/operator/ops.h"
#include "pipeline/jit/base.h"
#include "include/common/debug/common.h"
#include "include/common/debug/anf_dump_utils.h"
#include "mindspore/core/utils/file_utils.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/debug/anf_ir_dump.h"

using mindspore::tensor::TensorPy;

namespace mindspore {
namespace {
struct AnfDumpHandlerRegister {
  AnfDumpHandlerRegister() noexcept {
    AnfDumpHandler::SetDumpDatHandler([](const std::string &realpath, const FuncGraphPtr &graph) {
      AnfExporter exporter("");
      std::string realpath_dat = realpath + ".dat";
      ChangeFileMode(realpath_dat, S_IRWXU);
      exporter.ExportFuncGraph(realpath_dat, graph);
      ChangeFileMode(realpath_dat, S_IRUSR);
    });
  }
} callback_register;
}  // namespace
// ============================================= MindSpore IR Exporter =============================================

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

std::string AnfExporter::GetNodeType(const AnfNodePtr &nd) {
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
      oss << ", ref_key=:" << ref_key->value();
    }
    PrintTupleNodeUsedFlagsDat(sequence_abs, oss);
    oss << ">";
  } else if (type != nullptr) {
    oss << "<" << type;
    if (tensor_value != nullptr && tensor_value != kValueAny) {
      oss << ", value=...";
    }
    if (ref_key != nullptr) {
      oss << ", ref_key=:" << ref_key->value();
    }
    PrintTupleNodeUsedFlagsDat(sequence_abs, oss);
    oss << ">";
  } else {
    oss << "<null>";
  }
  return oss.str();
}

int AnfExporter::GetParamIndex(const FuncGraphPtr &func_graph, const AnfNodePtr &param, bool throw_excp) {
  if (func_graph == nullptr || param == nullptr) {
    return -1;
  }

  FuncGraphPtr fg = func_graph;
  while (fg != nullptr) {
    if (exported.find(fg) == exported.end()) {
      if (!check_integrity_) {
        break;
      }
      MS_LOG(EXCEPTION) << "Can not find func graph '" << fg->DumpText() << "'";
    }
    auto param_map = exported[fg];
    if (param_map.find(param) != param_map.end()) {
      return param_map[param];
    }
    fg = fg->parent();
  }
  if (throw_excp) {
    MS_LOG(EXCEPTION) << "Can not find index for param '" << param->DumpText() << "' for func graph '"
                      << func_graph->DumpText() << "'";
  }
  return -1;
}

// Try to find index of parameter for SymbolicKeyInstance from all exported graphs
// NOTICE: Suppose name of all parameters in SymbolicKeyInstance are different
int AnfExporter::GetParamIndexFromExported(const AnfNodePtr &param) {
  if (param == nullptr) {
    return -1;
  }

  int ret = -1;
  for (const auto &item : exported) {
    auto pram_iter = item.second.find(param);
    if (pram_iter != item.second.end()) {
      return pram_iter->second;
    }
  }
  return ret;
}

std::string AnfExporter::GetValueNodeText(const FuncGraphPtr &func_graph, const ValueNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  return GetValueText(func_graph, node->value());
}

std::string AnfExporter::GetMultitypeFuncGraphText(const prim::MultitypeFuncGraphPtr &mt_func_graph) const {
  auto py_funcs = mt_func_graph->GetPyFunctions();
  if (py_funcs.empty()) {
    return "";
  }

  std::ostringstream oss;

  oss << "{";
  bool is_first = true;
  for (const auto &py_func : py_funcs) {
    if (is_first) {
      is_first = false;
    } else {
      oss << ", ";
    }
    oss << "(";
    for (size_t i = 0; i < py_func.first.size(); ++i) {
      if (i > 0) {
        oss << ", ";
      }
      oss << py_func.first[i]->DumpText();
    }
    oss << ")";
  }
  oss << "}";

  return oss.str();
}

inline bool Skip(const MetaFuncGraphPtr &meta_func_graph) {
  return meta_func_graph->isa<prim::Tail>() || meta_func_graph->isa<prim::MakeTupleGradient>() ||
         meta_func_graph->isa<prim::MakeListGradient>() || meta_func_graph->isa<prim::MakeDictGradient>() ||
         meta_func_graph->isa<prim::TupleAdd>() || meta_func_graph->isa<prim::SequenceSliceGetItem>() ||
         meta_func_graph->isa<prim::ListSliceSetItem>() || meta_func_graph->isa<prim::UnpackCall>() ||
         meta_func_graph->isa<prim::ZipOperation>() || meta_func_graph->isa<prim::ListAppend>() ||
         meta_func_graph->isa<prim::ListInsert>() || meta_func_graph->isa<prim::DoSignatureMetaFuncGraph>() ||
         meta_func_graph->isa<prim::VmapMatchOutAxis>() || meta_func_graph->isa<prim::VmapGeneralPreprocess>() ||
         meta_func_graph->isa<prim::GradAux>() || meta_func_graph->isa<prim::PyExecuteGradient>() ||
         meta_func_graph->isa<prim::MutableGradient>();
}

/* inherit relation of MetaFuncGraph
 *
 * MetaGraph
 * ├── MultitypeGraph
 * ├── HyperMap
 * │   └── HyperMapPy
 * ├── Map
 * │   └── MapPy
 * ├── Tail
 * ├── MakeTupleGradient
 * ├── MakeListGradient
 * ├── PyExecuteGradient
 * ├── MutableGradient
 * ├── GradOperation
 * ├── TupleAdd
 * └── SequenceSlice
 *     ├──  SequenceSliceGetItem
 *     └──  ListSliceSetItem
 */
std::string AnfExporter::GetMetaFuncGraphText(const MetaFuncGraphPtr &meta_func_graph) {
  if (meta_func_graph == nullptr) {
    return "";
  }

  std::ostringstream oss;
  oss << meta_func_graph->type_name() << "::" << meta_func_graph->name();

  if (meta_func_graph->isa<prim::MultitypeFuncGraph>()) {
    prim::MultitypeFuncGraphPtr mt_func_graph = meta_func_graph->cast<prim::MultitypeFuncGraphPtr>();
    oss << GetMultitypeFuncGraphText(mt_func_graph);
  } else if (meta_func_graph
               ->isa<prim::HyperMapPy>()) {  // This statement must before 'meta_graph->isa<prim::HyperMap>()'
    auto hyper_map = meta_func_graph->cast<prim::HyperMapPyPtr>();
    if (hyper_map->GetFnLeaf() != nullptr) {
      oss << "{fn_leaf=" << GetMetaFuncGraphText(hyper_map->GetFnLeaf()) << "}";
    }
  } else if (meta_func_graph->isa<prim::HyperMap>()) {
    auto hyper_map = meta_func_graph->cast<prim::HyperMapPtr>();
    if (hyper_map->GetFnLeaf() != nullptr) {
      oss << "{fn_leaf=" << GetMetaFuncGraphText(hyper_map->GetFnLeaf()) << "}";
    }
  } else if (meta_func_graph->isa<prim::MapPy>()) {  // This statement must before 'meta_graph->isa<prim::Map>()'
    auto map = meta_func_graph->cast<prim::MapPyPtr>();
    if (map->GetFnLeaf() != nullptr) {
      oss << "{fn_leaf=" << GetMetaFuncGraphText(map->GetFnLeaf()) << "}";
    }
  } else if (meta_func_graph->isa<prim::Map>()) {
    auto map = meta_func_graph->cast<prim::MapPtr>();
    if (map->GetFnLeaf() != nullptr) {
      oss << "{fn_leaf=" << GetMetaFuncGraphText(map->GetFnLeaf()) << "}";
    }
  } else if (meta_func_graph->isa<prim::GradOperation>()) {
    prim::GradOperationPtr grad_op = meta_func_graph->cast<prim::GradOperationPtr>();
    oss << "{get_all=" << grad_op->get_all_ << ", get_by_list=" << grad_op->get_by_list_
        << ", sens_param=" << grad_op->sens_param_ << "}";
  } else if (meta_func_graph->isa<prim::VmapGeneralRule>()) {
    prim::VmapGeneralRulePtr general_rule_fg = meta_func_graph->cast<prim::VmapGeneralRulePtr>();
    oss << "{prim=" << general_rule_fg->prim_name() << ", axis_size=" << general_rule_fg->axis_size() << "}";
  } else if (meta_func_graph->isa<graph_bprop::BpropMetaFuncGraph>()) {
    oss << "{" << meta_func_graph->name() << "}";
  } else if (Skip(meta_func_graph)) {
    // Do nothing.
  } else {
    MS_LOG(EXCEPTION) << "Unknown MetaFuncGraph type " << meta_func_graph->type_name();
  }

  return oss.str();
}

std::string AnfExporter::GetPrimitiveText(const PrimitivePtr &prim) const {
  std::ostringstream oss;
  if (prim == nullptr) {
    return oss.str();
  }
  oss << prim->type_name() << "::" << prim->name();
  // Output primitive type
  oss << "{prim_type=" << static_cast<int>(prim->prim_type()) << "}";
  // Output primitive attributes
  oss << prim->GetAttrsText();

  if (prim->isa<prim::DoSignaturePrimitive>()) {
    auto do_signature = dyn_cast<prim::DoSignaturePrimitive>(prim);
    auto &func = do_signature->function();
    if (func->isa<Primitive>()) {
      auto sig_prim = dyn_cast<Primitive>(func);
      oss << sig_prim->GetAttrsText();
    }
  }

  return oss.str();
}

std::string AnfExporter::GetNameSpaceText(const parse::NameSpacePtr &ns) const {
  std::ostringstream oss;
  if (ns == nullptr) {
    return oss.str();
  }

  // Dump related module information in Namespace
  oss << ns->type_name() << "::" << ns->module();

  return oss.str();
}

std::string AnfExporter::GetSymbolicKeyInstanceText(const FuncGraphPtr &func_graph,
                                                    const SymbolicKeyInstancePtr &sym_inst) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(sym_inst);
  AnfNodePtr sym_node = sym_inst->node();
  MS_EXCEPTION_IF_NULL(sym_node);
  std::ostringstream oss;
  if (sym_node->isa<Parameter>()) {
    int idx = GetParamIndex(func_graph, sym_node, false);
    // If can not find SymbolicKeyInstance related parameter from ancestors,
    // try to find from all exported graphs
    if (idx < 0) {
      idx = GetParamIndexFromExported(sym_node);
    }
    if (idx < 0) {
      ParameterPtr p = dyn_cast<Parameter>(sym_node);
      if (p == nullptr) {
        MS_LOG(EXCEPTION) << "Sym_inst's node could not cast to parameter";
      }
      MS_LOG(WARNING) << "Can not find SymbolicKeyInstance: " << p->name();
    }
    oss << "SymInst(%para" << idx << ")";
  } else {
    MS_LOG(WARNING) << "SymbolicKeyInstance does not embed a parameter: " << sym_node->ToString();
    oss << "SymInst(cnode_" << sym_node->ToString() << ")";
  }

  return oss.str();
}

std::string AnfExporter::GetSequenceText(const FuncGraphPtr &func_graph, const ValuePtr &value) {
  std::ostringstream oss;
  // Output ValueList, ValueTuple
  ValueSequencePtr seq = dyn_cast<ValueSequence>(value);
  MS_EXCEPTION_IF_NULL(seq);
  MS_EXCEPTION_IF_NULL(value);
  bool is_tuple = value->isa<ValueTuple>();
  oss << (is_tuple ? "(" : "[");
  bool first_flag = true;
  for (auto elem : seq->value()) {
    if (first_flag) {
      first_flag = false;
    } else {
      oss << ", ";
    }
    oss << GetValueText(func_graph, elem);
  }
  oss << (is_tuple ? ")" : "]");
  return oss.str();
}

std::string AnfExporter::GetDictText(const FuncGraphPtr &func_graph, const ValuePtr &value) {
  std::ostringstream oss;
  ValueDictionaryPtr dict = value->cast<ValueDictionaryPtr>();
  oss << "{";
  bool first_flag = true;
  for (const auto &elem : dict->value()) {
    if (first_flag) {
      first_flag = false;
    } else {
      oss << ", ";
    }
    oss << "\"" << elem.first << "\": " << GetValueText(func_graph, elem.second);
  }
  oss << "}";
  return oss.str();
}

std::string AnfExporter::GetOtherValueText(const ValuePtr &value) const {
  std::ostringstream oss;

  if (check_integrity_) {
    MS_LOG(EXCEPTION) << "Need to process type: " << value->type_name() << ", dump text: " << value->DumpText();
  }
  oss << value->type_name() << "[" << value->DumpText() << "]";

  return oss.str();
}

static bool CanUseDumpText(const ValuePtr &value) {
  return (value->isa<RefKey>() || value->isa<Scalar>() || value->isa<StringImm>() || value->isa<tensor::Tensor>() ||
          value->isa<parse::Symbol>() || value->isa<None>() || value->isa<Null>() || value->isa<ValueSlice>() ||
          value->isa<Type>() || value->isa<KeywordArg>());
}

std::string AnfExporter::GetValueText(const FuncGraphPtr &func_graph, const ValuePtr &value) {
  if (func_graph == nullptr || value == nullptr) {
    return "";
  }
  if (value->isa<Primitive>()) {
    return GetPrimitiveText(value->cast<PrimitivePtr>());
  }
  if (value->isa<MetaFuncGraph>()) {
    MetaFuncGraphPtr meta_func_graph = value->cast<MetaFuncGraphPtr>();
    return GetMetaFuncGraphText(meta_func_graph);
  }
  if (value->isa<SymbolicKeyInstance>()) {
    return GetSymbolicKeyInstanceText(func_graph, value->cast<SymbolicKeyInstancePtr>());
  }
  if (value->isa<ValueSequence>()) {
    return GetSequenceText(func_graph, value);
  }
  if (value->isa<ValueDictionary>()) {
    return GetDictText(func_graph, value);
  }
  if (value->isa<parse::NameSpace>()) {
    return GetNameSpaceText(value->cast<parse::NameSpacePtr>());
  }
  if (value->isa<parse::PyObjectWrapper>()) {
    return value->type_name();
  }
  if (CanUseDumpText(value)) {
    return value->DumpText();
  }
  return GetOtherValueText(value);
}

// This function is used to output node in CNode's inputs
std::string AnfExporter::GetAnfNodeText(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                        const std::map<AnfNodePtr, int> &apply_map) {
  std::ostringstream oss;
  if (func_graph == nullptr || node == nullptr) {
    return oss.str();
  }

  if (node->isa<CNode>()) {
    auto iter = apply_map.find(node);
    if (iter == apply_map.end()) {
      MS_LOG(EXCEPTION) << "Can not find node '" << node->DumpText() << "' in apply_map";
    }
    oss << "%" << iter->second;
  } else if (node->isa<Parameter>()) {
    // Parameter maybe a free variable, so check it in its own funcgraph.
    oss << "%para" << GetParamIndex(node->func_graph(), node, check_integrity_);
    oss << "_" << node->ToString();
  } else if (IsValueNode<FuncGraph>(node)) {
    FuncGraphPtr fg = GetValueNode<FuncGraphPtr>(node);
    if (fg != nullptr) {
      oss << "call @" << fg->ToString();
    }
    if (!func_graph_set.contains(fg) && exported.find(fg) == exported.end() && export_used_) {
      func_graph_set.add(fg);
    }
  } else if (node->isa<ValueNode>()) {
    oss << GetValueNodeText(func_graph, node->cast<ValueNodePtr>());
  } else {
    MS_LOG(EXCEPTION) << "Unknown node '" << node->DumpText() << "'";
  }

  return oss.str();
}

void AnfExporter::OutputParameters(std::ostringstream &oss, const std::vector<AnfNodePtr> &parameters,
                                   ParamIndexMap *param_map) {
  bool first_flag = true;
  for (const AnfNodePtr &param : parameters) {
    if (param == nullptr) {
      continue;
    }
    auto parameter_ptr = param->cast<ParameterPtr>();
    if (parameter_ptr == nullptr) {
      MS_LOG(EXCEPTION) << "param cannot cast to ParameterPtr";
    }
    if (first_flag) {
      first_flag = false;
    } else {
      oss << ", ";
    }
    (*param_map)[param] = param_index;
    std::string type_info = GetNodeType(param);
    // Output parameter
    oss << "%para" << param_index << "_" << parameter_ptr->name();
    param_index += 1;
  }
}

void AnfExporter::OutputStatementComment(const CNodePtr &node, const FuncGraphPtr &func_graph,
                                         std::ostringstream &oss) {
  if (node == nullptr) {
    return;
  }
  // Output type of each input argument
  auto &inputs = node->inputs();
  if (node != func_graph->get_return()) {
    if (inputs.size() > 1) {
      oss << "\n      :(";
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
    oss << "\n      :(" << GetNodeType(node) << ")";
  }
  // Output other comment, map the graph name to original representation(containing unicode character)
  oss << "\n";
  oss << "      #scope: (" << node->scope()->name() << ")";
}

void AnfExporter::OutputCNodeText(std::ostringstream &oss, const CNodePtr &cnode, const FuncGraphPtr &func_graph,
                                  int *idx, std::map<AnfNodePtr, int> *const apply_map) {
  if (cnode == nullptr || func_graph == nullptr || idx == nullptr || apply_map == nullptr) {
    return;
  }
  auto &inputs = cnode->inputs();
  if (inputs.empty()) {
    return;
  }
  std::string op_text = GetAnfNodeText(func_graph, inputs[0], *apply_map);
  std::string fv_text = (cnode->func_graph() != func_graph) ? ("$(" + cnode->func_graph()->ToString() + "):") : "";
  // Non-return node
  if (cnode != func_graph->get_return()) {
    int apply_idx = (*idx)++;
    (*apply_map)[cnode] = apply_idx;
    std::string func_str = GetNodeFuncStr(inputs[0]);
    oss << "  %" << apply_idx << "(" << cnode->ToString() << ")"
        << " = " << fv_text << op_text;
    if (!func_str.empty()) {
      oss << "[" << func_str << "]"
          << "(";
    } else {
      oss << "(";
    }
  } else {
    oss << "  " << fv_text << op_text << "(";
  }

  for (size_t i = 1; i < inputs.size(); ++i) {
    if (i != 1) {
      oss << ", ";
    }
    AnfNodePtr arg = inputs[i];
    oss << GetAnfNodeText(func_graph, arg, *apply_map);
  }
  oss << ")";
}

void AnfExporter::OutputCNode(std::ostringstream &oss, const CNodePtr &cnode, const FuncGraphPtr &func_graph, int *idx,
                              std::map<AnfNodePtr, int> *const apply_map) {
  OutputCNodeText(oss, cnode, func_graph, idx, apply_map);
  // Output comment
  OutputStatementComment(cnode, func_graph, oss);
  oss << "\n";
}

void AnfExporter::OutputCNodes(std::ostringstream &oss, const std::vector<AnfNodePtr> &nodes,
                               const FuncGraphPtr &func_graph, const TaggedNodeMap &tagged_cnodes_map) {
  if (func_graph == nullptr) {
    return;
  }
  MS_LOG_TRY_CATCH_SCOPE;
  int idx = 1;
  std::map<AnfNodePtr, int> apply_map;
  for (const AnfNodePtr &node : nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      continue;
    }

    if (!tagged_cnodes_map.empty()) {
      auto iter = tagged_cnodes_map.find(node);
      if (iter != tagged_cnodes_map.end()) {
        oss << "\n#------------------------> " << iter->second << "\n";
      }
    }

    auto cnode = node->cast<CNodePtr>();
    OutputCNode(oss, cnode, func_graph, &idx, &apply_map);
    if (label_manage::GetGlobalTraceLabelType() == label_manage::TraceLabelType::kWithUniqueId) {
      oss << trace::GetDebugInfo(cnode->debug_info(), "      # ", kSourceLineTipDiscard) << "#"
          << label_manage::Label(cnode->debug_info()) << "\n";
    } else {
      std::string dgi = trace::GetDebugInfo(cnode->debug_info(), "      # ", kSourceLineTipDiscard);
      if (dgi != "") {
        oss << trace::GetDebugInfo(cnode->debug_info(), "      # ", kSourceLineTipDiscard) << "\n";
      }
    }
  }
}

void AnfExporter::OuputIrStyleCNodes(const FuncGraphPtr &func_graph, const std::vector<AnfNodePtr> &nodes,
                                     int32_t total_para, std::ostringstream &oss,
                                     OrderedMap<AnfNodePtr, int32_t> *para_map) {
  auto &parameters = func_graph->parameters();
  std::shared_ptr<SubGraphIRInfo> gsub = std::make_shared<SubGraphIRInfo>();
  ParamIndexMap param_map;
  exported[func_graph] = param_map;
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
    auto &inputs = cnode->inputs();
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (IsValueNode<FuncGraph>(inputs[i])) {
        FuncGraphPtr fg = GetValueNode<FuncGraphPtr>(inputs[i]);
        if (!func_graph_set.contains(fg) && exported.find(fg) == exported.end() && export_used_) {
          func_graph_set.add(fg);
        }
      }
    }
    DumpCNode(cnode, func_graph, *para_map, gsub);
    if (label_manage::GetGlobalTraceLabelType() == label_manage::TraceLabelType::kWithUniqueId) {
      gsub->buffer << trace::GetDebugInfo(cnode->debug_info(), "      # ", kSourceLineTipDiscard) << "#"
                   << label_manage::Label(cnode->debug_info()) << "\n";
    } else {
      std::string dgi = trace::GetDebugInfo(cnode->debug_info(), "      # ", kSourceLineTipDiscard);
      if (dgi != "") {
        gsub->buffer << trace::GetDebugInfo(cnode->debug_info(), "      # ", kSourceLineTipDiscard) << "\n";
      }
    }
  }
  if (!is_top_graph) {
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
  } else {
    is_top_graph = false;
  }
  oss << ") {\n";
  oss << gsub->buffer.str();
}

void AnfExporter::ExportOneFuncGraph(const FuncGraphPtr &func_graph, const TaggedNodeMap &tagged_cnodes_map,
                                     std::ostringstream &oss, int32_t total_para,
                                     OrderedMap<AnfNodePtr, int32_t> *para_map) {
  if (func_graph == nullptr) {
    return;
  }

  std::vector<AnfNodePtr> nodes = TopoSort(func_graph->get_return(), SuccIncoming, AlwaysInclude);
  std::vector<AnfNodePtr> parameters = func_graph->parameters();

  if (*(func_graph->switch_input())) {
    oss << "switch_input: " << *(func_graph->switch_input()) << "\n";
  }
  if (*(func_graph->switch_layer_input())) {
    oss << "switch_layer_input: " << *(func_graph->switch_layer_input()) << "\n";
  }
  oss << "subgraph attr:" << std::endl;
  for (const auto &attr : func_graph->attrs()) {
    oss << attr.first << " : ";
    if (attr.second->isa<BoolImm>()) {
      oss << GetValue<bool>(attr.second);
    } else if (attr.second->isa<StringImm>()) {
      oss << (GetValue<std::string>(attr.second));
    }
    oss << std::endl;
  }
  oss << "subgraph instance: " << func_graph->ToString() << " : " << func_graph.get() << std::endl;
  if (label_manage::GetGlobalTraceLabelType() == label_manage::TraceLabelType::kWithUniqueId) {
    oss << trace::GetDebugInfo(func_graph->debug_info(), "# ", kSourceLineTipDiscard) << "#"
        << label_manage::Label(func_graph->debug_info()) << "\n";
  } else {
    oss << trace::GetDebugInfo(func_graph->debug_info(), "# ", kSourceLineTipDiscard) << "\n";
  }
  oss << "subgraph @" << func_graph->ToString();
  if (func_graph->parent() != nullptr) {
    oss << " parent: [subgraph @" << func_graph->parent()->ToString() << "]";
  }
  oss << "(";
  if (para_map != nullptr) {
    OuputIrStyleCNodes(func_graph, nodes, total_para, oss, para_map);
  } else {
    ParamIndexMap param_map;
    OutputParameters(oss, parameters, &param_map);
    exported[func_graph] = param_map;
    oss << ") {\n";
    OutputCNodes(oss, nodes, func_graph, tagged_cnodes_map);
  }

  oss << "}\n";

  OutputOrderList(func_graph, oss);
}

void ExportGlobalInfoEntry(const FuncGraphPtr &graph, std::ostringstream &buffer, int graph_size) {
  if (graph == nullptr) {
    return;
  }

  buffer << "#IR entry      : @" << graph->ToString() << std::endl;
  buffer << "#Total subgraph: " << graph_size;
  buffer << std::endl;
  buffer << std::endl;
  buffer << "#attrs         :" << std::endl;
  for (const auto &attr : graph->attrs()) {
    buffer << attr.first << " : ";
    if (attr.second->isa<BoolImm>()) {
      buffer << GetValue<bool>(attr.second);
    } else if (attr.second->isa<StringImm>()) {
      buffer << (GetValue<std::string>(attr.second));
    }
    buffer << std::endl;
  }
}

void AnfExporter::ExportFuncGraph(const std::string &filename, const FuncGraphPtr &func_graph) {
  if (func_graph == nullptr) {
    return;
  }

  std::ofstream ofs(filename);
  if (!ofs.is_open()) {
    MS_LOG(ERROR) << "Open file '" << filename << "' failed!" << ErrnoToString(errno);
    return;
  }

  param_index = 1;
  int graph_size = 0;
  std::ostringstream oss;
  std::ostringstream paramoss;
  TaggedNodeMap tagged_cnodes_map;
  OrderedMap<AnfNodePtr, int32_t> para_map;
  int32_t total_para = DumpParams(func_graph, paramoss, &para_map);
  func_graph_set.add(func_graph);
  is_top_graph = true;
  while (!func_graph_set.empty()) {
    FuncGraphPtr fg = *func_graph_set.cbegin();
    ExportOneFuncGraph(fg, tagged_cnodes_map, oss, total_para, &para_map);
    oss << "\n\n";
    (void)func_graph_set.erase(fg);
    graph_size++;
  }
  std::ostringstream buffer;
  ExportGlobalInfoEntry(func_graph, buffer, graph_size);
  ofs << buffer.str() << paramoss.str() << "\n" << oss.str();
  ofs.close();
}

#ifdef ENABLE_DUMP_IR
void ExportIR(const std::string &filename, const FuncGraphPtr &func_graph) {
  if (func_graph == nullptr) {
    return;
  }

  auto filepath = GetSaveGraphsPathName(Common::AddId(filename, ".ir"));
  auto real_filepath = Common::CreatePrefixPath(filepath);
  if (!real_filepath.has_value()) {
    MS_LOG(ERROR) << "The export ir path: " << filepath << " is not illegal.";
    return;
  }
  ChangeFileMode(real_filepath.value(), S_IWUSR);
  AnfExporter exporter;
  exporter.ExportFuncGraph(real_filepath.value(), func_graph);
  // Set file mode to read only by user
  ChangeFileMode(real_filepath.value(), S_IRUSR);
}
#else
void ExportIR(const std::string &, const FuncGraphPtr &) {
  static bool already_printed = false;
  if (already_printed) {
    return;
  }
  already_printed = true;
  MS_LOG(WARNING) << "The functionality of dumping function graph IR is disabled, "
                  << "please recompile to enable it. See help of building script.";
}
#endif
}  // namespace mindspore

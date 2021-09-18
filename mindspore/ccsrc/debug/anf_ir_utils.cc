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

#include "debug/anf_ir_utils.h"

#include <fstream>
#include <map>
#include <memory>
#include <unordered_map>
#include <algorithm>
#include <iomanip>
#include "ir/graph_utils.h"
#include "utils/symbolic.h"
#include "ir/meta_func_graph.h"
#include "ir/param_info.h"
#include "pybind_api/ir/tensor_py.h"
#include "pipeline/jit/parse/python_adapter.h"
#include "pipeline/jit/parse/resolve.h"
#include "frontend/operator/composite/composite.h"
#include "frontend/operator/composite/map.h"
#include "utils/ordered_map.h"
#include "utils/ordered_set.h"
#include "utils/utils.h"
#include "utils/shape_utils.h"
#include "debug/trace.h"
#include "utils/label.h"
#include "utils/ms_context.h"
#include "frontend/operator/ops.h"
#include "pipeline/jit/base.h"
#include "debug/common.h"

using mindspore::tensor::TensorPy;

namespace mindspore {
std::string GetKernelNodeName(const AnfNodePtr &anf_node) {
  std::string kernel_name = anf_node->fullname_with_scope();
  if (kernel_name.empty()) {
    kernel_name = anf_node->ToString();
  }
  MS_LOG(DEBUG) << "Full scope kernel name is " << kernel_name << ".";
  return kernel_name;
}

// ============================================= MindSpore IR Exporter =============================================

std::string AnfExporter::GetNodeType(const AnfNodePtr &nd) {
  ValuePtr tensor_value = nullptr;
  auto abstract = nd->abstract();
  if (abstract != nullptr && abstract->isa<abstract::AbstractTensor>()) {
    tensor_value = abstract->BuildValue();
  }
  abstract::ShapePtr shape = nd->Shape() == nullptr ? nullptr : dyn_cast<abstract::Shape>(nd->Shape());
  TypePtr type = dyn_cast<Type>(nd->Type());
  std::ostringstream oss;
  if ((shape != nullptr) && (type != nullptr)) {
    oss << type->DumpText() << shape->DumpText();
    if (tensor_value != nullptr && tensor_value != kAnyValue) {
      oss << "(...)";
    }
  } else if (type != nullptr) {
    oss << type->DumpText();
    if (tensor_value != nullptr && tensor_value != kAnyValue) {
      oss << "(...)";
    }
  } else {
    oss << "Undefined";
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

std::string AnfExporter::GetValueNodeText(const FuncGraphPtr &fg, const ValueNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  return GetValueText(fg, node->value());
}

std::string AnfExporter::GetMultitypeFuncGraphText(const prim::MultitypeFuncGraphPtr &mt_func_graph) {
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
         meta_func_graph->isa<prim::MakeListGradient>() || meta_func_graph->isa<prim::TupleAdd>() ||
         meta_func_graph->isa<prim::TupleSlice>() || meta_func_graph->isa<prim::UnpackCall>() ||
         meta_func_graph->isa<prim::ZipOperation>() || meta_func_graph->isa<prim::ListAppend>() ||
         meta_func_graph->isa<prim::DoSignatureMetaFuncGraph>();
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
 * ├── GradOperation
 * └── TupleAdd
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
  } else if (Skip(meta_func_graph)) {
    // Do nothing
  } else {
    MS_LOG(EXCEPTION) << "Unknown MetaFuncGraph type " << meta_func_graph->type_name();
  }

  return oss.str();
}

std::string AnfExporter::GetPrimitiveText(const PrimitivePtr &prim) {
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

std::string AnfExporter::GetNameSpaceText(const parse::NameSpacePtr &ns) {
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
  ValueSequeuePtr seq = dyn_cast<ValueSequeue>(value);
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

std::string AnfExporter::GetOtherValueText(const FuncGraphPtr &, const ValuePtr &value) {
  std::ostringstream oss;

  if (check_integrity_) {
    MS_LOG(EXCEPTION) << "Need to process type: " << value->type_name() << ", dump text: " << value->DumpText();
  }
  oss << value->type_name() << "[" << value->DumpText() << "]";

  return oss.str();
}

std::string AnfExporter::GetValueText(const FuncGraphPtr &func_graph, const ValuePtr &value) {
  std::ostringstream oss;
  bool is_null_ptr = (func_graph == nullptr || value == nullptr);
  if (is_null_ptr) {
    return oss.str();
  }

  if (value->isa<Primitive>()) {
    oss << GetPrimitiveText(value->cast<PrimitivePtr>());
  } else if (value->isa<MetaFuncGraph>()) {
    MetaFuncGraphPtr meta_func_graph = value->cast<MetaFuncGraphPtr>();
    oss << GetMetaFuncGraphText(meta_func_graph);
  } else if (value->isa<SymbolicKeyInstance>()) {
    oss << GetSymbolicKeyInstanceText(func_graph, value->cast<SymbolicKeyInstancePtr>());
  } else if (value->isa<RefKey>()) {
    oss << value->DumpText();
  } else if (value->isa<Scalar>() || value->isa<StringImm>()) {
    oss << value->DumpText();
  } else if (value->isa<tensor::Tensor>()) {
    oss << value->DumpText();
  } else if (value->isa<parse::Symbol>() || value->isa<None>() || value->isa<Null>()) {
    oss << value->DumpText();
  } else if (value->isa<ValueSequeue>()) {
    oss << GetSequenceText(func_graph, value);
  } else if (value->isa<ValueDictionary>()) {
    oss << GetDictText(func_graph, value);
  } else if (value->isa<ValueSlice>()) {
    ValueSlicePtr slice = value->cast<ValueSlicePtr>();
    oss << slice->DumpText();
  } else if (value->isa<Type>()) {
    oss << value->DumpText();
  } else if (value->isa<parse::NameSpace>()) {
    oss << GetNameSpaceText(value->cast<parse::NameSpacePtr>());
  } else if (value->isa<parse::PyObjectWrapper>()) {
    oss << value->type_name();
  } else if (value->isa<KeywordArg>()) {
    KeywordArgPtr keyword_arg = value->cast<KeywordArgPtr>();
    oss << keyword_arg->DumpText();
  } else {
    return GetOtherValueText(func_graph, value);
  }

  return oss.str();
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
  } else if (IsValueNode<FuncGraph>(node)) {
    FuncGraphPtr fg = GetValueNode<FuncGraphPtr>(node);
    oss << fg->type_name() << "::fg_" << fg->debug_info()->get_id();

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

void AnfExporter::OutputParameters(std::ofstream &ofs, const std::vector<AnfNodePtr> &parameters,
                                   OrderedMap<AnfNodePtr, int, ParamPtrHasher, ParamPtrEqual> *param_map) {
  bool first_flag = true;
  for (const AnfNodePtr &param : parameters) {
    if (first_flag) {
      first_flag = false;
      ofs << "        ";
    } else {
      ofs << "        , ";
    }
    (*param_map)[param] = param_index;
    std::string type_info = GetNodeType(param);
    // Output parameter and type
    if (type_info == "Undefined") {
      ofs << "%para" << param_index;
    } else {
      ofs << "%para" << param_index << " : " << type_info;
    }
    // Output comment
    ofs << "    # " << param->DumpText() << "\n";
    param_index += 1;
  }
}

void AnfExporter::OutputStatementComment(std::ofstream &ofs, const CNodePtr &node) {
  if (node == nullptr) {
    return;
  }

  // Output type of each input argument
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
  // Output other comment, map the graph name to original representation(containing unicode character)
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
    comment << " fg_" << func_graph_id << "=" << fg->ToString();
  }
  if (has_comment) {
    ofs << comment.str();
  }
  ofs << " #scope: " << node->scope()->name();
}

void AnfExporter::OutputCNodeText(std::ofstream &ofs, const CNodePtr &cnode, const FuncGraphPtr &func_graph, int *idx,
                                  std::map<AnfNodePtr, int> *const apply_map) {
  auto &inputs = cnode->inputs();
  std::string op_text = GetAnfNodeText(func_graph, inputs[0], *apply_map);
  std::string fv_text = (cnode->func_graph() != func_graph) ? ("$(" + cnode->func_graph()->ToString() + "):") : "";
  // Non-return node
  if (cnode != func_graph->get_return()) {
    int apply_idx = (*idx)++;
    (*apply_map)[cnode] = apply_idx;
    std::string type_info = GetNodeType(cnode);
    if (type_info == "Undefined") {
      ofs << "    %" << apply_idx << " = " << fv_text << op_text << "(";
    } else {
      ofs << "    %" << apply_idx << " : " << fv_text << type_info << " = " << op_text << "(";
    }
  } else {
    ofs << "    " << fv_text << op_text << "(";
  }

  for (size_t i = 1; i < inputs.size(); ++i) {
    if (i != 1) {
      ofs << ", ";
    }
    AnfNodePtr arg = inputs[i];
    ofs << GetAnfNodeText(func_graph, arg, *apply_map);
  }
  ofs << ")";
}

void AnfExporter::OutputCNode(std::ofstream &ofs, const CNodePtr &cnode, const FuncGraphPtr &func_graph, int *idx,
                              std::map<AnfNodePtr, int> *const apply_map) {
  OutputCNodeText(ofs, cnode, func_graph, idx, apply_map);
  // Output comment
  OutputStatementComment(ofs, cnode);
  ofs << "\n";
}

void AnfExporter::OutputCNodes(std::ofstream &ofs, const std::vector<AnfNodePtr> &nodes, const FuncGraphPtr &func_graph,
                               const TaggedNodeMap &tagged_cnodes_map) {
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

    if (!tagged_cnodes_map.empty()) {
      auto iter = tagged_cnodes_map.find(node);
      if (iter != tagged_cnodes_map.end()) {
        ofs << "\n#------------------------> " << iter->second << "\n";
      }
    }

    auto cnode = node->cast<CNodePtr>();
    OutputCNode(ofs, cnode, func_graph, &idx, &apply_map);
    if (label_manage::GetGlobalTraceLabelType() == label_manage::TraceLabelType::kWithUniqueId) {
      ofs << trace::GetDebugInfo(cnode->debug_info(), "      # ", kSourceLineTipDiscard) << "#"
          << label_manage::Label(cnode->debug_info()) << "\n";
    } else {
      ofs << trace::GetDebugInfo(cnode->debug_info(), "      # ", kSourceLineTipDiscard) << "#" << cnode->ToString()
          << "\n";
    }
  }
}

void AnfExporter::OutputOrderList(std::ofstream &ofs, const FuncGraphPtr &func_graph) {
  auto &order_list = func_graph->order_list();
  if (order_list.empty()) {
    return;
  }
  constexpr int width = 4;
  ofs << "# order:\n";
  int i = 1;
  for (auto &node : order_list) {
    ofs << '#' << std::setw(width) << i << ": " << node->DebugString() << '\n';
    ++i;
  }
}

void AnfExporter::ExportOneFuncGraph(std::ofstream &ofs, const FuncGraphPtr &func_graph,
                                     const TaggedNodeMap &tagged_cnodes_map) {
  if (func_graph == nullptr) {
    return;
  }

  std::vector<AnfNodePtr> nodes = TopoSort(func_graph->get_return(), SuccIncoming, AlwaysInclude);
  std::vector<AnfNodePtr> parameters = func_graph->parameters();
  OrderedMap<AnfNodePtr, int, ParamPtrHasher, ParamPtrEqual> param_map;

  if (*(func_graph->switch_input())) {
    ofs << "switch_input: " << *(func_graph->switch_input()) << "\n";
  }
  if (*(func_graph->switch_layer_input())) {
    ofs << "switch_layer_input: " << *(func_graph->switch_layer_input()) << "\n";
  }
  ofs << "# [No." << (exported.size() + 1) << "] " << func_graph->DumpText() << "\n";
  if (label_manage::GetGlobalTraceLabelType() == label_manage::TraceLabelType::kWithUniqueId) {
    ofs << trace::GetDebugInfo(func_graph->debug_info(), "# ", kSourceLineTipDiscard) << "#"
        << label_manage::Label(func_graph->debug_info()) << "\n";
  } else {
    ofs << trace::GetDebugInfo(func_graph->debug_info(), "# ", kSourceLineTipDiscard) << "\n";
  }
  ofs << "funcgraph fg_" << func_graph->debug_info()->get_id();
  // Output name of parent of graph if exists
  if (func_graph->parent() != nullptr) {
    ofs << "[fg_" << func_graph->parent()->debug_info()->get_id() << "]";
  }
  ofs << "(\n";

  OutputParameters(ofs, parameters, &param_map);

  exported[func_graph] = param_map;
  ofs << (!parameters.empty() ? "    " : "") << ") {\n";

  OutputCNodes(ofs, nodes, func_graph, tagged_cnodes_map);

  ofs << "}\n";

  OutputOrderList(ofs, func_graph);
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

  TaggedNodeMap tagged_cnodes_map;
  func_graph_set.add(func_graph);
  while (!func_graph_set.empty()) {
    FuncGraphPtr fg = *func_graph_set.begin();
    ExportOneFuncGraph(ofs, fg, tagged_cnodes_map);
    ofs << "\n\n";
    (void)func_graph_set.erase(fg);
  }
  ofs << "# num of total function graphs: " << exported.size();

  ofs.close();
}

#ifdef ENABLE_DUMP_IR
void ExportIR(const std::string &filename, const FuncGraphPtr &func_graph) {
  if (func_graph == nullptr) {
    return;
  }

  auto filepath = GetSaveGraphsPathName(Common::AddId(filename, ".dat"));
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
                  << "please recompile source to enable it. See help of building script.";
}

void ExportIR(const std::string &filename, const std::vector<TaggedGraph> &graphs) {
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

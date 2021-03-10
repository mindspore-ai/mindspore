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
// max number of elements in sequence
const int NUM_MAX_SEQUENCE_ELEMS = 0x00FFFFFF;

// ============================================== MindSpore IR Common ==============================================
// get MindSpore Intermediate Representation Path
std::string GetMsIrPath(void) {
  std::string path;
  const char *path_ptr = getenv("MS_IR_PATH");
  if (path_ptr != nullptr) {
    path = path_ptr;
    char real_path[PATH_MAX] = {0};
#if defined(_WIN32) || defined(_WIN64)
    if (path.size() > PATH_MAX || _fullpath(real_path, path.c_str(), PATH_MAX) == nullptr) {
      MS_LOG(EXCEPTION) << "MS IR Path error, " << path_ptr;
    }
#else
    if (path.size() > PATH_MAX || nullptr == realpath(path.c_str(), real_path)) {
      MS_LOG(EXCEPTION) << "MS IR path error, " << path_ptr;
    }
#endif
    path = real_path;
  }
  return path;
}

std::string dump_obj(const py::object &obj, const std::string &path) {
  py::module mod = parse::python_adapter::GetPyModule(parse::PYTHON_MOD_PARSE_MODULE);
  py::object name = parse::python_adapter::CallPyModFn(mod, "dump_obj", obj, py::str(path));
  return py::str(name);
}

py::object load_obj(const std::string &path) {
  py::module mod = parse::python_adapter::GetPyModule(parse::PYTHON_MOD_PARSE_MODULE);
  py::object obj = parse::python_adapter::CallPyModFn(mod, "load_obj", py::str(path));
  return obj;
}

// ============================================= MindSpore IR Exporter =============================================

std::string AnfExporter::GetNodeType(const AnfNodePtr &nd) {
  abstract::ShapePtr shape = nd->Shape() == nullptr ? nullptr : dyn_cast<abstract::Shape>(nd->Shape());
  TypePtr type = dyn_cast<Type>(nd->Type());
  std::ostringstream oss;
  if ((nullptr != shape) && (nullptr != type)) {
    oss << type->DumpText() << shape->DumpText();
  } else if (nullptr != type) {
    oss << type->DumpText();
  } else {
    oss << "Undefined";
  }
  return oss.str();
}

std::string AnfExporter::DumpObject(const py::object &obj, const std::string &category) const {
  std::string pkl_path = GetMsIrPath();
  // if not specified env 'MS_IR_PATH', do not create any files
  if (pkl_path.empty() || (getenv("MS_IR_FILE") != nullptr)) {
    return "null";
  }
  std::string file_prefix = id_ + "." + category;
  std::string file_name = dump_obj(obj, pkl_path + "/" + file_prefix);
  return file_prefix + file_name;
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
      MS_LOG(EXCEPTION) << "Can not find func graph '" << fg->DumpText() << "." << fg->debug_info()->get_id() << "'";
    }
    auto param_map = exported[fg];
    if (param_map.find(param) != param_map.end()) {
      return param_map[param];
    }
    fg = fg->parent();
  }
  if (throw_excp) {
    MS_LOG(EXCEPTION) << "Can not find index for param '" << param->DumpText() << "' for func graph '"
                      << func_graph->DumpText() << "." << func_graph->debug_info()->get_id() << "'";
  }
  return -1;
}

// try to find index of parameter for SymbolicKeyInstance from all exported graphs
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

    // dump Python Function object
    oss << "@" << DumpObject(py_func.second, "F");
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
               ->isa<prim::HyperMapPy>()) {  // this statement must before 'meta_graph->isa<prim::HyperMap>()'
    auto hyper_map = meta_func_graph->cast<prim::HyperMapPyPtr>();
    if (hyper_map->GetFnLeaf() != nullptr) {
      oss << "{fn_leaf=" << GetMetaFuncGraphText(hyper_map->GetFnLeaf()) << "}";
    }
  } else if (meta_func_graph->isa<prim::HyperMap>()) {
    auto hyper_map = meta_func_graph->cast<prim::HyperMapPtr>();
    if (hyper_map->GetFnLeaf() != nullptr) {
      oss << "{fn_leaf=" << GetMetaFuncGraphText(hyper_map->GetFnLeaf()) << "}";
    }
  } else if (meta_func_graph->isa<prim::MapPy>()) {  // this statement must before 'meta_graph->isa<prim::Map>()'
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
    // do nothing
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
  // need to serialize internal python function of PrimitivePy and record its prim_type
  if (prim->isa<PrimitivePy>()) {
    PrimitivePyPtr primpy = prim->cast<PrimitivePyPtr>();

    // dump related function in PrimitivePy
    oss << "@" << DumpObject(primpy->GetPyObj(), "P");

    // output primitive type
    oss << "{prim_type=" << static_cast<int>(prim->prim_type()) << "}";
  }

  // output primitive attributes
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

  // dump related module information in Namespace
  oss << ns->type_name() << "::" << ns->module() << "@" << DumpObject(ns->obj(), "N");

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
    // if can not find SymbolicKeyInstance related parameter from ancestors,
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
  // output ValueList, ValueTuple
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
    auto tensor_ptr = dyn_cast<tensor::Tensor>(value);
    oss << value->DumpText() << "@" << DumpObject(TensorPy::AsNumpy(*tensor_ptr), "T");
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

// this function is used to output node in CNode's inputs
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
    // output parameter and type
    if (type_info == "Undefined") {
      ofs << "%para" << param_index;
    } else {
      ofs << "%para" << param_index << " : " << type_info;
    }

    // dump Default value of parameter if exists
    const ParameterPtr param_ptr = dyn_cast<Parameter>(param);
    if (param_ptr == nullptr) {
      MS_LOG(EXCEPTION) << "Param could not cast to parameter";
    }
    if (param_ptr->has_default()) {
      auto param_value = param_ptr->default_param();
      ofs << " = @" << DumpObject(py::cast(param_value), "D");
    }

    // output comment
    ofs << "    # " << param->DumpText() << "\n";

    param_index += 1;
  }
}

void AnfExporter::OutputStatementComment(std::ofstream &ofs, const CNodePtr &node) {
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
  }
  if (has_comment) {
    ofs << comment.str();
  }
  ofs << " #scope: " << node->scope()->name();
}

void AnfExporter::OutputCNodes(std::ofstream &ofs, const std::vector<AnfNodePtr> &nodes,
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
    auto &inputs = cnode->inputs();
    std::string op_text = GetAnfNodeText(func_graph, inputs[0], apply_map);
    // non-return node
    if (node != func_graph->get_return()) {
      int apply_idx = idx++;
      apply_map[node] = apply_idx;
      std::string type_info = GetNodeType(node);
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
      ofs << GetAnfNodeText(func_graph, arg, apply_map);
    }
    ofs << ")";

    // output comment
    OutputStatementComment(ofs, cnode);
    ofs << "\n";
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

void AnfExporter::ExportOneFuncGraph(std::ofstream &ofs, const FuncGraphPtr &func_graph) {
  if (func_graph == nullptr) {
    return;
  }

  std::vector<AnfNodePtr> nodes = TopoSort(func_graph->get_return(), SuccIncoming, AlwaysInclude);
  std::vector<AnfNodePtr> parameters = func_graph->parameters();
  OrderedMap<AnfNodePtr, int, ParamPtrHasher, ParamPtrEqual> param_map;

  if (*(func_graph->switch_layer_input())) {
    ofs << "switch_layer_input: " << *(func_graph->switch_layer_input()) << "\n";
  }
  ofs << "# [No." << (exported.size() + 1) << "] " << func_graph->DumpText() << "."
      << func_graph->debug_info()->get_id() << "\n";
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

  OutputOrderList(ofs, func_graph);
}

void AnfExporter::ExportFuncGraph(const std::string &filename, const FuncGraphPtr &func_graph) {
  if (func_graph == nullptr) {
    return;
  }

  std::ofstream ofs(filename);
  if (!ofs.is_open()) {
    MS_LOG(ERROR) << "Open file '" << filename << "' failed!";
    return;
  }

  param_index = 1;

  func_graph_set.add(func_graph);
  while (!func_graph_set.empty()) {
    FuncGraphPtr fg = *func_graph_set.begin();
    ExportOneFuncGraph(ofs, fg);
    ofs << "\n\n";
    (void)func_graph_set.erase(fg);
  }
  ofs << "# num of total function graphs: " << exported.size();

  ofs.close();
}

void AnfExporter::ExportFuncGraph(const std::string &filename, const std::vector<TaggedGraph> &graphs) {
  if (graphs.empty()) {
    return;
  }

  std::ofstream ofs(filename);
  if (!ofs.is_open()) {
    MS_LOG(ERROR) << "Open file '" << filename << "' failed!";
    return;
  }

  param_index = 1;

  for (const auto &tagged_graph : graphs) {
    tagged_cnodes_ = tagged_graph.second;
    ExportOneFuncGraph(ofs, tagged_graph.first);
    tagged_cnodes_.clear();
    ofs << "\n\n";
  }

  ofs << "# num of total function graphs: " << graphs.size();

  ofs.close();
}

#ifdef ENABLE_DUMP_IR
void ExportIR(const std::string &filename, const std::string &id, const FuncGraphPtr &func_graph) {
  if (func_graph == nullptr) {
    return;
  }

  auto real_filename = pipeline::GetSaveGraphsPathName(Common::AddId(filename, ".dat"));
  AnfExporter exporter(id);
  ChangeFileMode(real_filename, S_IRWXU);
  exporter.ExportFuncGraph(real_filename, func_graph);
  // set file mode to read only by user
  ChangeFileMode(real_filename, S_IRUSR);
}

void ExportIR(const std::string &filename, const std::vector<TaggedGraph> &graphs) {
  auto real_filename = pipeline::GetSaveGraphsPathName(Common::AddId(filename, ".dat"));
  AnfExporter exporter("", false);
  ChangeFileMode(real_filename, S_IRWXU);
  exporter.ExportFuncGraph(real_filename, graphs);
  // set file mode to read only by user
  ChangeFileMode(real_filename, S_IRUSR);
}
#else
void ExportIR(const std::string &, const std::string &, const FuncGraphPtr &) {
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

// ============================================= MindSpore IR Importer =============================================

enum Token : int {
  TOK_INVALID = 0,   // invalid token
  TOK_LPARENTHESIS,  // ( left parenthesis
  TOK_RPARENTHESIS,  // ) right parenthesis
  TOK_LBRACKET,      // [ left bracket
  TOK_RBRACKET,      // ] right bracket
  TOK_LBRACE,        // { left brace
  TOK_RBRACE,        // } right brace
  TOK_COMMA,         // , comma
  TOK_EQUALITY,      // = equality
  TOK_COLON,         // : colon
  TOK_STAR,          // * star
  TOK_VARIABLE,      // variable
  TOK_AT_FILE,       // @filename
  TOK_PARAMETER,     // parameter
  TOK_IDENTIFIER,    // identifier
  TOK_FUNCGRAPH,     // keyword 'funcgraph'
  TOK_RETURN,        // id prim::return
  TOK_STRING,        // string
  TOK_NUMBER,        // number
  TOK_COMMENT,       // comment
  TOK_EOL,           // end of line
  TOK_EOF,           // end of file
  TOK_ERROR          // file read error
};

std::map<Token, const char *> token_text = {
  {TOK_INVALID, "invalid"},      // invalid token
  {TOK_LPARENTHESIS, "("},       // ( left parenthesis
  {TOK_RPARENTHESIS, ")"},       // ) right parenthesis
  {TOK_LBRACKET, "["},           // [ left bracket
  {TOK_RBRACKET, "]"},           // ] right bracket
  {TOK_LBRACE, "{"},             // { left brace
  {TOK_RBRACE, "}"},             // } right brace
  {TOK_COMMA, ","},              // , comma
  {TOK_EQUALITY, "="},           // = equality
  {TOK_COLON, ":"},              // : colon
  {TOK_STAR, "*"},               // * start
  {TOK_VARIABLE, nullptr},       // variable
  {TOK_AT_FILE, nullptr},        // @file
  {TOK_PARAMETER, nullptr},      // parameter
  {TOK_IDENTIFIER, nullptr},     // identifier
  {TOK_FUNCGRAPH, "funcgraph"},  // keyword 'funcgraph'
  {TOK_RETURN, nullptr},         // id prim::return
  {TOK_STRING, nullptr},         // string
  {TOK_NUMBER, nullptr},         // number
  {TOK_COMMENT, nullptr},        // comment
  {TOK_EOL, "\n"},               // end of line
  {TOK_EOF, ""},                 // end of file
  {TOK_ERROR, "error"}           // file read error
};

class Lexer {
 public:
  // filename is checked in ImportIR;
  explicit Lexer(const char *filename) : fin(filename) {}

  ~Lexer() {
    try {
      if (fin.is_open()) {
        fin.close();
      }
    } catch (const std::exception &e) {
      MS_LOG(ERROR) << "Exception when closing file";
    } catch (...) {
      std::string exName(abi::__cxa_current_exception_type()->name());
      MS_LOG(ERROR) << "Error occurred when closing file. Exception name: " << exName;
    }
  }

  bool IsSingleCharToken(char ch, Token *token_ptr) {
    // clang-format off
    std::unordered_map<char, Token> char_to_token = {
      {'(', TOK_LPARENTHESIS},
      {')', TOK_RPARENTHESIS},
      {'[', TOK_LBRACKET},
      {']', TOK_RBRACKET},
      {'{', TOK_LBRACE},
      {'}', TOK_RBRACE},
      {',', TOK_COMMA},
      {'=', TOK_EQUALITY},
      {':', TOK_COLON},
      {'*', TOK_STAR}};
    // clang-format on

    auto iter = char_to_token.find(ch);
    if (iter == char_to_token.end()) {
      return false;
    }

    if (token_ptr != nullptr) {
      *token_ptr = iter->second;
    }

    return true;
  }

  Token GetNextToken() {
#ifdef DEBUG
    Token token = GetNextTokenInner();
    const char *str = token_text[token];
    std::string text = (str == nullptr ? GetTokenText() : str);
    MS_LOG(DEBUG) << "------Parse token] " << text;
    return token;
  }

  Token GetNextTokenInner() {
#endif
    tok_idx = 0;
    Token tok = TOK_ERROR;
    char ch = SkipTabAndSpace();
    if (ch == CODE_EOF) {
      return TOK_EOF;
    } else if (ch == CODE_ERROR) {
      return TOK_ERROR;
    } else if (IsSingleCharToken(ch, &tok)) {
      return tok;
    } else if (ch == '\r') {
      char c = GetChar();
      if (c == '\n') {
        line_++;
        return TOK_EOL;
      }
      UnGetChar(c);
      line_++;
      return TOK_EOL;
    } else if (ch == '\n') {
      line_++;
      return TOK_EOL;
    } else if (ch == '#') {
      return ParseComment(ch);
    } else if (ch == '"') {
      return ParseString();
    } else if (ch == '%') {
      return ParseVariableOrParameter(ch);
    } else if (ch == '@') {
      return ParseAtFile();
    } else if (IsDigit(ch) || ch == '-') {
      return ParseNumber(ch);
    } else if (IsAlpha(ch) || ch == '_') {
      return ParseIdentifier(ch);
    } else {
      return TOK_ERROR;
    }
  }

  Token SkipWhiteToken() {
    Token tok = GetNextToken();
    while (tok == TOK_EOL || tok == TOK_COMMENT) {
      tok = GetNextToken();
    }
    return tok;
  }

  std::string GetTokenText() const { return std::string(tok_buf); }

  int GetLineNo() const { return line_; }

 private:
  Token ParseComment(char ch) {
    char c = GetChar();
    while (c != '\r' && c != '\n' && c != CODE_EOF) {
      c = GetChar();
    }
    if (ch != CODE_EOF) {
      UnGetChar(c);
    }
    tok_buf[0] = '#';
    tok_buf[1] = '\0';
    return TOK_COMMENT;
  }

  Token ParseString() {
    tok_idx = 0;
    char c = GetChar();
    while (c != '"') {
      if (tok_idx >= BUF_SIZE) {
        MS_LOG(EXCEPTION) << "Length of token which is " << tok_idx << " exceeds " << BUF_SIZE;
      }
      if (c == '\r' || c == '\n') {
        MS_LOG(EXCEPTION) << "Literal newline characters are not allowed within the quote at line " << line_;
      }
      if (c == CODE_EOF) {
        MS_LOG(EXCEPTION) << "Encounter EOF within the quote at line " << line_;
      }
      tok_buf[tok_idx++] = c;
      c = GetChar();
    }
    tok_buf[tok_idx] = '\0';
    return TOK_STRING;
  }

  Token ParseVariableOrParameter(char ch) {
    tok_idx = 0;
    tok_buf[tok_idx++] = ch;
    char c = GetChar();
    while (IsAlphaNumeric(c)) {
      if (tok_idx >= BUF_SIZE) {
        MS_LOG(EXCEPTION) << "Length of token which is " << tok_idx << " exceeds " << BUF_SIZE;
      }
      tok_buf[tok_idx++] = c;
      c = GetChar();
    }
    tok_buf[tok_idx] = '\0';
    UnGetChar(c);

    // judge parameter: %para[0-9]+
    tok_buf[tok_idx] = '\0';
    std::string param_key = "%para";
    if (strncmp(tok_buf, param_key.c_str(), param_key.size()) == 0) {
      if (tok_idx <= param_key.size()) {
        return TOK_ERROR;
      }
      for (auto i = static_cast<unsigned>(param_key.size()); i < tok_idx; ++i) {
        if (!IsDigit(tok_buf[i])) {
          return TOK_ERROR;
        }
      }
      return TOK_PARAMETER;
    }

    // judge local variable: %[0-9]+
    if (tok_idx == 1) {
      return TOK_ERROR;
    }
    for (unsigned i = 1; i < tok_idx; ++i) {
      if (!IsDigit(tok_buf[i])) {
        return TOK_ERROR;
      }
    }
    return TOK_VARIABLE;
  }

  Token ParseAtFile() {
    tok_idx = 0;
    char c = GetChar();
    while (IsAlphaNumeric(c) || c == '_' || c == '.') {
      if (tok_idx >= BUF_SIZE) {
        MS_LOG(EXCEPTION) << "Length of token which is " << tok_idx << " exceeds " << BUF_SIZE;
      }
      tok_buf[tok_idx++] = c;
      c = GetChar();
    }
    tok_buf[tok_idx] = '\0';
    UnGetChar(c);

    if (tok_idx == 0) {
      return TOK_ERROR;
    }

    return TOK_AT_FILE;
  }

  Token ParseNumber(char ch) {
    tok_buf[tok_idx++] = ch;
    char c = GetChar();
    // parse number, e.g. 10, 15.6, 1e-5
    while (IsDigit(c) || c == '.' || c == 'e' || c == '-') {
      if (tok_idx >= BUF_SIZE) {
        MS_LOG(EXCEPTION) << "Length of token which is " << tok_idx << " exceeds " << BUF_SIZE;
      }
      tok_buf[tok_idx++] = c;
      c = GetChar();
    }
    UnGetChar(c);
    tok_buf[tok_idx] = '\0';
    return TOK_NUMBER;
  }

  Token ParseIdentifier(char ch) {
    tok_idx = 0;
    tok_buf[tok_idx++] = ch;
    char c = GetChar();
    while (IsAlphaNumeric(c) || c == '.' || c == ':' || c == '_') {
      if (tok_idx >= BUF_SIZE) {
        MS_LOG(EXCEPTION) << "Length of token which is " << tok_idx << " exceeds " << BUF_SIZE;
      }
      tok_buf[tok_idx++] = c;
      c = GetChar();
    }
    UnGetChar(c);
    tok_buf[tok_idx] = '\0';

    if (strcmp(tok_buf, "funcgraph") == 0) {
      return TOK_FUNCGRAPH;
    }
    if (strcmp(tok_buf, "Primitive::return") == 0) {
      return TOK_RETURN;
    }
    return TOK_IDENTIFIER;
  }

  // Suppose the file only contain ASCII character
  char GetChar() {
    if (ungot_char != UNGOT_CHAR) {
      char ch = ungot_char;
      ungot_char = UNGOT_CHAR;
      return ch;
    }
    if (idx >= cnt) {
      if (fin.eof()) {
        return CODE_EOF;
      }
      cnt = fin.read(buffer, BUF_SIZE).gcount();
      if ((fin.bad() || fin.fail()) && !fin.eof()) {
        MS_LOG(EXCEPTION) << "Read file error!";
      }
      idx = 0;
    }
    return buffer[idx++];
  }

  void UnGetChar(char ch) {
    if (ungot_char == UNGOT_CHAR) {
      ungot_char = ch;
    }
  }

  static bool IsTabOrSpace(char ch) { return ch == ' ' || ch == '\t'; }

  static bool IsDigit(char ch) { return ch >= '0' && ch <= '9'; }

  static bool IsAlpha(char ch) { return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z'); }

  static bool IsAlphaNumeric(char ch) { return IsDigit(ch) || IsAlpha(ch); }

  // skip whitespace(including comment) to read a valid character
  char SkipTabAndSpace() {
    char ch = GetChar();
    while (IsTabOrSpace(ch)) {
      ch = GetChar();
    }
    return ch;
  }

  std::ifstream fin;

  static const unsigned BUF_SIZE = 4096;  // lexer buffer size
  char buffer[BUF_SIZE + 1] = {0};        // buffer for holding text read from text
  std::streamsize cnt = 0;                // number of valid characters in the buffer
  unsigned idx = 0;                       // index of next char the lexer to read from

  char tok_buf[BUF_SIZE + 1] = {0};  // token buffer
  unsigned tok_idx = 0;              // token buffer index

  char ungot_char = UNGOT_CHAR;  // store ungot char

  static const int CODE_EOF = -1;     // return code of GetChar
  static const int CODE_ERROR = -2;   // read file error
  static const char UNGOT_CHAR = -3;  // value of ungot char

  int line_ = 1;  // current line number
};

const unsigned Lexer::BUF_SIZE;

class IrParser {
 public:
  explicit IrParser(const char *filename) : lexer_(filename) {}

  ~IrParser() {}

  py::object LoadObject(const std::string &file_name) const {
    std::string pkl_path = GetMsIrPath();
    py::object default_obj = load_obj(pkl_path + "/" + file_name);
    return default_obj;
  }

  void ParseFile() {
    FuncGraphPtr func_graph = ParseFuncGraph();
    while (func_graph != nullptr) {
      func_graphs_.push_back(func_graph);
      func_graph = ParseFuncGraph();
    }
    if (error_flag_) {
      MS_LOG(EXCEPTION) << "Parse Error at line: " << lexer_.GetLineNo();
    }

    MS_LOG(INFO) << "Total graphs: " << func_graphs_.size();
  }

  Token ParseParent(FuncGraphPtr *const parent_ptr) {
    if (lexer_.GetNextToken() != TOK_IDENTIFIER) {
      return TOK_ERROR;
    }

    std::string parent_name = lexer_.GetTokenText();
    // NOTICE: require definition of parent graph must before child graph
    auto iter = func_graphs_map_.find(parent_name);
    if (iter == func_graphs_map_.end()) {
      MS_LOG(EXCEPTION) << "Can not find definition of parent func graph '" << parent_name << "' at line "
                        << lexer_.GetLineNo();
    }
    if (parent_ptr != nullptr) {
      *parent_ptr = iter->second;
    }

    if (lexer_.GetNextToken() != TOK_RBRACKET) {
      return TOK_ERROR;
    }

    return lexer_.GetNextToken();
  }

  FuncGraphPtr ParseFuncGraph() {
    cnodes_.clear();

    Token tok = lexer_.SkipWhiteToken();
    if (tok != TOK_FUNCGRAPH) {
      error_flag_ = tok != TOK_EOF;
      return nullptr;
    }

    if (lexer_.GetNextToken() != TOK_IDENTIFIER) {
      error_flag_ = true;
      return nullptr;
    }

    std::string func_graph_name = lexer_.GetTokenText();
    if (func_graphs_map_.find(func_graph_name) == func_graphs_map_.end()) {
      func_graphs_map_[func_graph_name] = std::make_shared<FuncGraph>();
    }
    FuncGraphPtr func_graph = func_graphs_map_[func_graph_name];
    MS_EXCEPTION_IF_NULL(func_graph);
    MS_EXCEPTION_IF_NULL(func_graph->debug_info());
    func_graph->debug_info()->set_name(func_graph_name);  // for debugging

    FuncGraphPtr parent = nullptr;
    tok = lexer_.GetNextToken();
    if (tok == TOK_LBRACKET) {
      tok = ParseParent(&parent);
      if (parent != nullptr) {
        parents_map_[func_graph] = parent;
      }
    }

    if (tok != TOK_LPARENTHESIS) {
      error_flag_ = true;
      return nullptr;
    }

    if (ParseParameters(func_graph) == nullptr) {
      error_flag_ = true;
      return nullptr;
    }

    if (lexer_.SkipWhiteToken() != TOK_LBRACE) {
      error_flag_ = true;
      return nullptr;
    }

    // parse statements
    if (ParseStatements(func_graph) == nullptr) {
      error_flag_ = true;
      return nullptr;
    }

    func_graphs_map_[func_graph_name] = func_graph;

    return func_graph;
  }

  FuncGraphPtr ParseStatements(const FuncGraphPtr &func_graph) {
    Token tok = lexer_.SkipWhiteToken();
    while (tok == TOK_VARIABLE) {
      if (ParseStatement(func_graph) == nullptr) {
        return nullptr;
      }
      tok = lexer_.SkipWhiteToken();
    }
    if (tok == TOK_RETURN) {
      return ParseReturn(func_graph);
    }
    return nullptr;
  }

  FuncGraphPtr ParseStatement(FuncGraphPtr func_graph) {
    std::string var_name = lexer_.GetTokenText();
    Token tok = lexer_.GetNextToken();
    AbstractBasePtr type = nullptr;
    if (tok == TOK_COLON) {
      tok = ParseType(func_graph, &type);
    }
    if (tok != TOK_EQUALITY) {
      return nullptr;
    }

    std::vector<AnfNodePtr> inputs;
    AnfNodePtr node = nullptr;
    ValuePtr val = nullptr;
    tok = ParseItem(func_graph, &node, &val);
    if (tok != TOK_LPARENTHESIS) {
      return nullptr;
    }
    inputs.push_back(node);

    int lineno = lexer_.GetLineNo();

    if (ParseArguments(func_graph, &inputs) == nullptr) {
      return nullptr;
    }

    tok = lexer_.GetNextToken();
    if (tok == TOK_COMMENT) {
      tok = lexer_.GetNextToken();
    }
    if (tok != TOK_EOL) {
      return nullptr;
    }

    MS_EXCEPTION_IF_NULL(func_graph);
    cnodes_[var_name] = func_graph->NewCNode(inputs);
    MS_EXCEPTION_IF_NULL(cnodes_[var_name]);
    cnodes_[var_name]->set_debug_info(std::make_shared<NodeDebugInfo>(var_name + "@" + std::to_string(lineno)));
    return func_graph;
  }

  FuncGraphPtr ParseReturn(FuncGraphPtr func_graph) {
    if (lexer_.GetNextToken() != TOK_LPARENTHESIS) {
      return nullptr;
    }

    AnfNodePtr input1 = nullptr;
    ValuePtr value = nullptr;
    Token tok = ParseItem(func_graph, &input1, &value, lexer_.GetNextToken());
    int lineno = lexer_.GetLineNo();

    if (tok != TOK_RPARENTHESIS) {
      return nullptr;
    }

    tok = lexer_.GetNextToken();
    if (tok == TOK_COMMENT) {
      tok = lexer_.GetNextToken();
    }
    if (tok != TOK_EOL) {
      return nullptr;
    }

    if (lexer_.SkipWhiteToken() != TOK_RBRACE) {
      return nullptr;
    }

    PrimitivePtr prim = std::make_shared<Primitive>("Return");
    ValueNodePtr input0 = std::make_shared<ValueNode>(prim);
    std::vector<AnfNodePtr> inputs;
    inputs.push_back(input0);
    inputs.push_back(input1);
    MS_EXCEPTION_IF_NULL(func_graph);
    CNodePtr ret = func_graph->NewCNode(inputs);
    MS_EXCEPTION_IF_NULL(ret);
    ret->set_debug_info(std::make_shared<NodeDebugInfo>(std::string("ret@") + std::to_string(lineno)));

    func_graph->set_return(ret);

    return func_graph;
  }

  void SetBasicType(TypePtr *ptr, const TypePtr &dtype) const {
    if (ptr == nullptr) {
      return;
    }
    *ptr = dtype;
  }

  void SetTupleType(TypePtr *ptr) {
    if (ptr == nullptr) {
      return;
    }
    *ptr = std::make_shared<Tuple>();
  }

  void SetTupleType(TypePtr *ptr, const TypePtrList &elems) {
    if (ptr == nullptr) {
      return;
    }
    *ptr = std::make_shared<Tuple>(elems);
  }

  void SetArrayType(TypePtr *const ptr, const TypePtr &elem_type, const ShapeVector &) {
    if (ptr == nullptr) {
      return;
    }
    *ptr = std::make_shared<TensorType>(elem_type);
  }

  void SetListType(TypePtr *ptr) {
    if (ptr == nullptr) {
      return;
    }
    *ptr = std::make_shared<List>();
  }

  void SetListType(TypePtr *ptr, const TypePtrList &elems) {
    if (ptr == nullptr) {
      return;
    }
    *ptr = std::make_shared<List>(elems);
  }

  void SetJTaggedType(TypePtr *ptr, const TypePtr &elem) {
    if (ptr == nullptr) {
      return;
    }
    *ptr = std::make_shared<JTagged>(elem);
  }

  void SetBasicType(AbstractBasePtr *ptr, const TypePtr &dtype) const {
    if (ptr == nullptr) {
      return;
    }
    *ptr = std::make_shared<abstract::AbstractScalar>(dtype);
  }

  // void SetBasicType(AbstractBasePtr *ptr, const SymbolicKeyTypePtr& dtype) {}
  void SetBasicType(AbstractBasePtr *const ptr, const TypeNonePtr &) const {
    if (ptr == nullptr) {
      return;
    }
    *ptr = std::make_shared<abstract::AbstractNone>();
  }

  void SetBasicType(AbstractBasePtr *, const FunctionPtr &) const {}
  void SetBasicType(AbstractBasePtr *, const TensorTypePtr &) const {}

  void SetTupleType(AbstractBasePtr *const ptr, const AbstractBasePtrList &elems) {
    if (ptr == nullptr) {
      return;
    }
    // if one of elems is nullptr, just return
    if (std::any_of(std::begin(elems), std::end(elems), [](const AbstractBasePtr &elem) { return elem == nullptr; })) {
      return;
    }
    *ptr = std::make_shared<abstract::AbstractTuple>(elems);
  }

  void SetArrayType(AbstractBasePtr *const ptr, const TypePtr &elem_type, const ShapeVector &shape) {
    if (ptr == nullptr) {
      return;
    }
    *ptr = std::make_shared<abstract::AbstractTensor>(elem_type, shape);
  }

  void SetListType(AbstractBasePtr *const ptr, const AbstractBasePtrList &elems) {
    if (ptr == nullptr) {
      return;
    }
    if (std::any_of(std::begin(elems), std::end(elems), [](const AbstractBasePtr &elem) { return elem == nullptr; })) {
      return;
    }
    *ptr = std::make_shared<abstract::AbstractList>(elems);
  }

  void SetJTaggedType(AbstractBasePtr *const ptr, const AbstractBasePtr &elem) {
    if (ptr == nullptr) {
      return;
    }
    *ptr = std::make_shared<abstract::AbstractJTagged>(elem);
  }

  template <typename T>
  Token ParseTypeVector(const FuncGraphPtr &func_graph, Token tok, const std::string &type, T *const ptr = nullptr) {
    if (tok != TOK_LBRACKET) {
      MS_LOG(EXCEPTION) << "Illegal case, , wrong token start symbol.";
      return tok;
    }

    bool first_flag = true;
    std::vector<T> elems;
    do {
      tok = lexer_.GetNextToken();
      if (first_flag) {
        if (tok == TOK_RBRACKET) {
          return lexer_.GetNextToken();
        }
        first_flag = false;
      }
      T elem = nullptr;
      tok = ParseOneType(func_graph, tok, &elem);
      elems.push_back(elem);
      if (tok == TOK_STAR) {
        if (lexer_.GetNextToken() != TOK_NUMBER) {
          return TOK_ERROR;
        }
        int num_elems = StringToScalar<int>(lexer_.GetTokenText());
        if (num_elems < 1 || num_elems > NUM_MAX_SEQUENCE_ELEMS) {
          MS_LOG(EXCEPTION) << "Number of elements " << num_elems << " is out of range [1, " << NUM_MAX_SEQUENCE_ELEMS
                            << "]";
        }
        for (int i = 0; i < num_elems - 1; ++i) {
          elems.push_back(elem);
        }
        tok = lexer_.GetNextToken();
      }
    } while (tok == TOK_COMMA);
    if (tok != TOK_RBRACKET) {
      return TOK_ERROR;
    }
    if (type == "Tuple") {
      SetTupleType(ptr, elems);
    } else if (type == "List") {
      SetListType(ptr, elems);
    } else {
      MS_LOG(EXCEPTION) << "This method does not support " << type << " parse.";
    }
    return lexer_.GetNextToken();
  }

  template <typename T>
  Token ParseTypeArray(const FuncGraphPtr &func_graph, Token tok, T *const ptr = nullptr) {
    if (tok != TOK_LPARENTHESIS) {
      if (ptr != nullptr) {
        SetBasicType(ptr, std::make_shared<TensorType>());
      }
      return tok;
    }
    // process Array element type
    TypePtr elem_type = nullptr;
    ShapeVector shape;
    tok = ParseOneType(func_graph, lexer_.GetNextToken(), &elem_type);
    if (tok != TOK_RPARENTHESIS) {
      return TOK_ERROR;
    }
    tok = lexer_.GetNextToken();
    if (tok != TOK_LBRACKET) {
      // NOTICE: if shape.size == 0, is this ok?
      SetArrayType(ptr, elem_type, shape);
      return tok;
    }
    // process Array shape
    do {
      tok = lexer_.GetNextToken();
      // case: Array(I32)[]
      if (tok != TOK_NUMBER) {
        break;
      }
      shape.push_back(StringToScalar<int>(lexer_.GetTokenText()));
      tok = lexer_.GetNextToken();
    } while (tok == TOK_COMMA);
    if (tok != TOK_RBRACKET) {
      return TOK_ERROR;
    }

    SetArrayType(ptr, elem_type, shape);

    return lexer_.GetNextToken();
  }

  bool IsNumberType(const std::string &type, TypeId *typeid_ptr) {
    // clang-format off
    static std::unordered_map<std::string, TypeId> basic_types = {
      {"Bool", kNumberTypeBool},
      {"I8", kNumberTypeInt8},
      {"I16", kNumberTypeInt16},
      {"I32", kNumberTypeInt32},
      {"I64", kNumberTypeInt64},
      {"U8", kNumberTypeUInt8},
      {"U16", kNumberTypeUInt16},
      {"U32", kNumberTypeUInt32},
      {"U64", kNumberTypeUInt64},
      {"F16", kNumberTypeFloat16},
      {"F32", kNumberTypeFloat32},
      {"F64", kNumberTypeFloat64},
      {"Int", kNumberTypeInt},
      {"UInt", kNumberTypeUInt},
      {"Float", kNumberTypeFloat},
      {"Number", kObjectTypeNumber}};
    // clang-format on

    auto iter = basic_types.find(type);
    if (iter == basic_types.end()) {
      return false;
    }
    if (typeid_ptr != nullptr) {
      *typeid_ptr = iter->second;
    }
    return true;
  }

  template <typename T>
  void ParseNumberType(const std::string &type, TypeId typeId, T *const ptr = nullptr) {
    TypePtr dtype = nullptr;

    std::unordered_map<int, TypePtr> type_map = {
      {static_cast<int>(kNumberTypeBool), std::make_shared<Bool>()},        // Bool
      {static_cast<int>(kNumberTypeInt8), std::make_shared<Int>(8)},        // Int8
      {static_cast<int>(kNumberTypeInt16), std::make_shared<Int>(16)},      // Int16
      {static_cast<int>(kNumberTypeInt32), std::make_shared<Int>(32)},      // Int32
      {static_cast<int>(kNumberTypeInt64), std::make_shared<Int>(64)},      // Int64
      {static_cast<int>(kNumberTypeUInt8), std::make_shared<UInt>(8)},      // UInt8
      {static_cast<int>(kNumberTypeUInt16), std::make_shared<UInt>(16)},    // UInt16
      {static_cast<int>(kNumberTypeUInt32), std::make_shared<UInt>(32)},    // UInt32
      {static_cast<int>(kNumberTypeUInt64), std::make_shared<UInt>(64)},    // UInt64
      {static_cast<int>(kNumberTypeFloat16), std::make_shared<Float>(16)},  // Float16
      {static_cast<int>(kNumberTypeFloat32), std::make_shared<Float>(32)},  // Float32
      {static_cast<int>(kNumberTypeFloat64), std::make_shared<Float>(64)},  // Float64
      {static_cast<int>(kNumberTypeInt), std::make_shared<Int>()},          // Int
      {static_cast<int>(kNumberTypeUInt), std::make_shared<UInt>()},        // UInt
      {static_cast<int>(kNumberTypeFloat), std::make_shared<Float>()},      // Float
      {static_cast<int>(kObjectTypeNumber), std::make_shared<Number>()},    // Number
    };

    auto iter = type_map.find(static_cast<int>(typeId));
    if (iter != type_map.end()) {
      dtype = iter->second;
    } else {
      MS_LOG(EXCEPTION) << "Unknown number type " << type;
    }

    SetBasicType(ptr, dtype);
  }

  template <typename T>
  Token ParseTrivalType(const std::string &type, T *const ptr = nullptr) {
    if (type == "NoneType") {
      SetBasicType(ptr, std::make_shared<TypeNone>());
      return lexer_.GetNextToken();
    } else if (type == "ProblemType") {
      SetBasicType(ptr, std::make_shared<Problem>());
      return lexer_.GetNextToken();
    } else if (type == "ExternalType") {
      SetBasicType(ptr, std::make_shared<External>());
      return lexer_.GetNextToken();
    } else if (type == "AnythingType") {
      SetBasicType(ptr, kAnyType);
      return lexer_.GetNextToken();
    } else if (type == "TypeType") {
      SetBasicType(ptr, std::make_shared<TypeType>());
      return lexer_.GetNextToken();
    } else {
      MS_LOG(EXCEPTION) << "Unknown type error at line " << lexer_.GetLineNo();
    }
  }

  template <typename T>
  Token ParseOneType(const FuncGraphPtr &func_graph, Token tok, T *const ptr = nullptr) {
    if (tok != TOK_IDENTIFIER) {
      return TOK_ERROR;
    }
    std::string type = lexer_.GetTokenText();
    TypeId typeId = kTypeUnknown;
    if (IsNumberType(type, &typeId)) {
      ParseNumberType(type, typeId, ptr);
      return lexer_.GetNextToken();
    } else if (type == "Tuple") {
      return ParseTypeVector(func_graph, lexer_.GetNextToken(), type, ptr);
    } else if (type == "Tensor") {
      return ParseTypeArray(func_graph, lexer_.GetNextToken(), ptr);
    } else if (type == "List") {
      return ParseTypeVector(func_graph, lexer_.GetNextToken(), type, ptr);
    } else if (type == "Func") {
      tok = lexer_.GetNextToken();
      if (tok != TOK_LBRACKET) {
        SetBasicType(ptr, std::make_shared<Function>());
        return tok;
      }
      MS_LOG(EXCEPTION) << "Need to process function parameter types at line " << lexer_.GetLineNo();
    } else if (type == "JT") {
      tok = lexer_.GetNextToken();
      if (tok != TOK_LBRACKET) {
        return tok;
      }
      T elem = nullptr;
      tok = ParseOneType(func_graph, lexer_.GetNextToken(), &elem);
      SetJTaggedType(ptr, elem);
      if (tok != TOK_RBRACKET) {
        return TOK_ERROR;
      }
      return lexer_.GetNextToken();
    } else if (type == "SymType") {
      SetBasicType(ptr, std::make_shared<SymbolicKeyType>());
      return lexer_.GetNextToken();
    } else if (type == "EnvType") {
      SetBasicType(ptr, std::make_shared<EnvType>());
      return lexer_.GetNextToken();
    } else if (Match(type, "Cls.")) {
      MS_LOG(EXCEPTION) << "Need to do class type at line " << lexer_.GetLineNo();
    } else {
      return ParseTrivalType(type, ptr);
    }
  }

  Token ParseType(const FuncGraphPtr &func_graph, AbstractBasePtr *const abstract = nullptr) {
    return ParseOneType(func_graph, lexer_.GetNextToken(), abstract);
  }

  Token ParseAttributes(const FuncGraphPtr &func_graph, const PrimitivePtr &prim) {
    Token tok = ParseAttribute(func_graph, prim);
    while (tok == TOK_COMMA) {
      tok = ParseAttribute(func_graph, prim);
    }
    if (tok != TOK_RBRACKET) {
      return TOK_ERROR;
    }
    return lexer_.GetNextToken();
  }

  Token ParseAttribute(const FuncGraphPtr &func_graph, const PrimitivePtr &prim) {
    Token tok = lexer_.GetNextToken();
    if (tok != TOK_IDENTIFIER) {
      return TOK_ERROR;
    }
    std::string attr_name = lexer_.GetTokenText();

    if (lexer_.GetNextToken() != TOK_EQUALITY) {
      return TOK_ERROR;
    }

    ValuePtr value = nullptr;
    tok = ParseValue(func_graph, lexer_.GetNextToken(), &value);

    if (prim != nullptr) {
      prim->set_attr(attr_name, value);
    } else {
      MS_LOG(EXCEPTION) << "Non primitive obj has attributes";
    }

    return tok;
  }

  FuncGraphPtr ParseParameters(FuncGraphPtr func_graph) {
    Token tok = lexer_.SkipWhiteToken();
    while (tok == TOK_PARAMETER) {
      ParameterPtr param = std::make_shared<Parameter>(func_graph);
      param->set_name(lexer_.GetTokenText());
      param_nodes_[lexer_.GetTokenText()] = param;
      int lineno = lexer_.GetLineNo();
      param->set_debug_info(std::make_shared<NodeDebugInfo>(lexer_.GetTokenText() + "@" + std::to_string(lineno)));
      func_graph->add_parameter(param);

      tok = lexer_.GetNextToken();
      // parse type
      if (tok == TOK_COLON) {
        AbstractBasePtr type = nullptr;
        tok = ParseType(func_graph, &type);
      }
      // parse default value
      if (tok == TOK_EQUALITY) {
        if (lexer_.GetNextToken() != TOK_AT_FILE) {
          MS_LOG(EXCEPTION) << "Expect @file at line " << lexer_.GetLineNo();
        }

        // load parameter default value from serialized file
        py::object default_obj = LoadObject(lexer_.GetTokenText());
        auto param_value_new = py::cast<tensor::TensorPtr>(default_obj);
        param->set_default_param(param_value_new);

        tok = lexer_.GetNextToken();
      }
      if (tok == TOK_COMMENT || tok == TOK_EOL) {
        tok = lexer_.SkipWhiteToken();
      }

      Token next = tok;
      if (next == TOK_RPARENTHESIS) {
        return func_graph;
      } else if (next == TOK_COMMA) {
        tok = lexer_.SkipWhiteToken();
      } else {
        return nullptr;
      }
    }
    return tok == TOK_RPARENTHESIS ? func_graph : nullptr;
  }

  FuncGraphPtr ParseArguments(FuncGraphPtr func_graph, std::vector<AnfNodePtr> *const inputs_ptr) {
    Token tok = ParseArgument(func_graph, inputs_ptr);
    while (tok == TOK_COMMA) {
      tok = ParseArgument(func_graph, inputs_ptr);
    }
    if (tok != TOK_RPARENTHESIS) {
      return nullptr;
    }
    return func_graph;
  }

  AnfNodePtr FindParameter(FuncGraphPtr func_graph, const std::string &param_name) {
    while (func_graph != nullptr) {
      for (auto &ptr : func_graph->parameters()) {
        MS_EXCEPTION_IF_NULL(ptr);
        ParameterPtr param = ptr->cast<ParameterPtr>();
        MS_EXCEPTION_IF_NULL(param);
        if (param->name() == param_name) {
          return ptr;
        }
      }
      auto iter = parents_map_.find(func_graph);
      if (iter == parents_map_.end()) {
        break;
      }
      func_graph = iter->second;
    }

    return nullptr;
  }

  bool Match(const std::string &str, const std::string &pattern) const {
    return strncmp(str.c_str(), pattern.c_str(), pattern.length()) == 0;
  }

  template <typename T, typename V>
  Token ParseScalar(ValuePtr *const val_ptr) {
    if (lexer_.GetNextToken() != TOK_NUMBER) {
      return TOK_ERROR;
    }
    std::stringstream ss;
    ss << lexer_.GetTokenText();

    if (lexer_.GetNextToken() != TOK_RPARENTHESIS) {
      return TOK_ERROR;
    }

    V val;
    ss >> val;
    *val_ptr = std::make_shared<T>(val);

    return lexer_.GetNextToken();
  }

  template <typename VT, typename V, typename T>
  Token ParseScalar(ValuePtr *const val_ptr, Token tok) {
    if (tok != TOK_LPARENTHESIS) {
      *val_ptr = std::make_shared<T>();
      return tok;
    }

    return ParseScalar<VT, V>(val_ptr);
  }

  template <typename VT, typename V, typename T, const unsigned nbits>
  Token ParseScalar(ValuePtr *const val_ptr, Token tok) {
    if (tok != TOK_LPARENTHESIS) {
      *val_ptr = std::make_shared<T>(nbits);
      return tok;
    }

    return ParseScalar<VT, V>(val_ptr);
  }

  template <typename T>
  T StringToScalar(const std::string &text) {
    std::stringstream ss;
    T value;
    ss << text;
    ss >> value;
    return value;
  }

  Token ParseTensor(ValuePtr *const val_ptr) {
    // parse type
    TypeId type;
    if (lexer_.GetNextToken() != TOK_LPARENTHESIS) {
      return TOK_ERROR;
    }
    if (lexer_.GetNextToken() != TOK_NUMBER) {
      return TOK_ERROR;
    }
    type = static_cast<TypeId>(StringToScalar<int>(lexer_.GetTokenText()));
    if (lexer_.GetNextToken() != TOK_RPARENTHESIS) {
      return TOK_ERROR;
    }

    // parse shape
    ShapeVector shape;
    Token tok = lexer_.GetNextToken();
    if (tok != TOK_LBRACKET) {
      return TOK_ERROR;
    }

    do {
      tok = lexer_.GetNextToken();
      // consider case: Tensor(23)[]
      if (tok != TOK_NUMBER) {
        break;
      }
      shape.push_back(StringToScalar<int>(lexer_.GetTokenText()));

      tok = lexer_.GetNextToken();
    } while (tok == TOK_COMMA);

    if (tok != TOK_RBRACKET) {
      return TOK_ERROR;
    }

    if (lexer_.GetNextToken() != TOK_AT_FILE) {
      return TOK_ERROR;
    }

    py::object tensor_obj = LoadObject(lexer_.GetTokenText());
    py::array tensor_data = py::cast<py::array>(tensor_obj);
    if (!tensor_data) {
      return TOK_ERROR;
    }
    *val_ptr = TensorPy::MakeTensor(tensor_data, TypeIdToType(type));

    return lexer_.GetNextToken();
  }

  Token ParsePrimType(Token tok, PrimType *prim_type_ptr) {
    if (tok != TOK_LBRACE) {
      return tok;
    }
    if (lexer_.GetNextToken() != TOK_IDENTIFIER) {
      return TOK_ERROR;
    }
    if (lexer_.GetTokenText() != "prim_type") {
      return TOK_ERROR;
    }
    if (lexer_.GetNextToken() != TOK_EQUALITY) {
      return TOK_ERROR;
    }
    if (lexer_.GetNextToken() != TOK_NUMBER) {
      return TOK_ERROR;
    }
    int val = 0;
    std::stringstream ss;
    ss << lexer_.GetTokenText();
    ss >> val;
    *prim_type_ptr = PrimType(val);
    if (lexer_.GetNextToken() != TOK_RBRACE) {
      return TOK_ERROR;
    }
    return lexer_.GetNextToken();
  }

  Token ParseMultitypeFuncGraphItem(const prim::MultitypeFuncGraphPtr &mt_func_graph, Token tok) {
    if (tok != TOK_LPARENTHESIS) {
      return TOK_ERROR;
    }
    TypePtrList type_list;
    do {
      TypePtr type = nullptr;
      tok = ParseOneType(nullptr, lexer_.GetNextToken(), &type);
      type_list.push_back(type);
    } while (tok == TOK_COMMA);
    if (tok != TOK_RPARENTHESIS) {
      return TOK_ERROR;
    }
    if (lexer_.GetNextToken() != TOK_AT_FILE) {
      return TOK_ERROR;
    }

    // load Python function from serialized file
    py::object py_func = LoadObject(lexer_.GetTokenText());
    MS_EXCEPTION_IF_NULL(mt_func_graph);
    mt_func_graph->Register(type_list, py::function(py_func));

    return lexer_.GetNextToken();
  }

  Token ParseMultitypeFuncGraph(const prim::MultitypeFuncGraphPtr &mt_func_graph, Token tok) {
    if (tok != TOK_LBRACE) {
      return tok;
    }
    do {
      tok = ParseMultitypeFuncGraphItem(mt_func_graph, lexer_.GetNextToken());
    } while (tok == TOK_COMMA);
    if (tok != TOK_RBRACE) {
      return TOK_ERROR;
    }
    return lexer_.GetNextToken();
  }

  Token ParseBoolValue(const std::string &key, bool *val_ptr) {
    if (lexer_.GetNextToken() != TOK_IDENTIFIER || lexer_.GetTokenText() != key) {
      return TOK_ERROR;
    }
    if (lexer_.GetNextToken() != TOK_EQUALITY) {
      return TOK_ERROR;
    }
    if (lexer_.GetNextToken() != TOK_NUMBER) {
      return TOK_ERROR;
    }
    bool value = false;
    {
      std::stringstream ss;
      ss << lexer_.GetTokenText();
      ss >> value;
    }

    if (val_ptr != nullptr) {
      *val_ptr = value;
    }

    return lexer_.GetNextToken();
  }

  Token ParseValueGradOperation(const std::string &name, ValuePtr *const val_ptr) {
    if (lexer_.GetNextToken() != TOK_LBRACE) {
      return TOK_ERROR;
    }
    // get_all=0, get_by_list=1, sens_param=1
    bool get_all = false;
    Token tok = ParseBoolValue("get_all", &get_all);
    if (tok != TOK_COMMA) {
      return TOK_ERROR;
    }

    bool get_by_list = false;
    tok = ParseBoolValue("get_by_list", &get_by_list);
    if (tok != TOK_COMMA) {
      return TOK_ERROR;
    }

    bool sens_param = false;
    tok = ParseBoolValue("sens_param", &sens_param);
    if (tok != TOK_RBRACE) {
      return TOK_ERROR;
    }

    *val_ptr = std::make_shared<prim::GradOperation>(name, get_all, get_by_list, sens_param);

    return lexer_.GetNextToken();
  }

  Token ParseSymbolicKeyInstance(const FuncGraphPtr &func_graph, AnfNodePtr *const node_ptr = nullptr) {
    if (lexer_.GetNextToken() != TOK_LPARENTHESIS) {
      return TOK_ERROR;
    }
    if (lexer_.GetNextToken() != TOK_PARAMETER) {
      return TOK_ERROR;
    }
    std::string param_name = lexer_.GetTokenText();
    if (lexer_.GetNextToken() != TOK_RPARENTHESIS) {
      return TOK_ERROR;
    }
    auto iter = param_nodes_.find(param_name);
    if (iter == param_nodes_.end()) {
      MS_LOG(EXCEPTION) << "Can not find param '" << param_name << "' for SymbolicKeyInstance at line "
                        << lexer_.GetLineNo();
    }

    PrimitivePtr embed = std::make_shared<Primitive>("embed");
    std::vector<AnfNodePtr> inputs;
    inputs.push_back(std::make_shared<ValueNode>(embed));
    inputs.push_back(iter->second);
    if (node_ptr != nullptr) {
      MS_EXCEPTION_IF_NULL(func_graph);
      *node_ptr = func_graph->NewCNode(inputs);
    } else {
      MS_LOG(EXCEPTION) << "Not processed SymbolicKeyInstance '" << param_name << "' at line " << lexer_.GetLineNo()
                        << ".";
    }
    return lexer_.GetNextToken();
  }

  Token ParsePrimitivePy(const FuncGraphPtr &func_graph, const std::string &id, ValuePtr *const val_ptr) {
    if (lexer_.GetNextToken() != TOK_AT_FILE) {
      return TOK_ERROR;
    }

    // restore python function of PrimitivePy from serialized file
    py::object py_obj = LoadObject(lexer_.GetTokenText());
    PrimitivePyPtr ptr = nullptr;
    if (py::hasattr(py_obj, "__setattr_flag__") && py::hasattr(py_obj, "_clone")) {
      auto clone_fn = py_obj.attr("_clone");
      py::object new_obj = clone_fn();
      ptr = new_obj.cast<PrimitivePyPtr>();
      if (ptr == nullptr) {
        MS_LOG(EXCEPTION) << "Cast to type 'PrimitivePyPtr' error";
      }
    } else {
      auto len = strlen("PrimitivePy::");
      if (id.size() < len) {
        return TOK_ERROR;
      }
      ptr = std::make_shared<PrimitivePy>(id.substr(len), py_obj);
    }
    *val_ptr = ptr;

    PrimType prim_type = kPrimTypeUnknown;
    Token next = ParsePrimType(lexer_.GetNextToken(), &prim_type);
    if (prim_type != kPrimTypeUnknown) {
      ptr->set_prim_type(prim_type);
    }
    if (next != TOK_LBRACKET) {
      return next;
    }
    // parse attributes
    next = ParseAttributes(func_graph, ptr);
    return next;
  }

  Token ParseValueGraphAndNamespace(const std::string &id, ValuePtr *const val_ptr) {
    if (Match(id, "MultitypeFuncGraph::")) {
      std::string name = id.substr(strlen("MultitypeFuncGraph::"));
      auto mt_func_graph = std::make_shared<prim::MultitypeFuncGraph>(name);
      *val_ptr = mt_func_graph;
      Token next = ParseMultitypeFuncGraph(mt_func_graph, lexer_.GetNextToken());
      return next;
    } else if (Match(id, "HyperMapPy::")) {
      *val_ptr = std::make_shared<prim::HyperMapPy>();
      Token next = lexer_.GetNextToken();
      // process case: fn_leaf is not null
      if (next == TOK_LBRACE) {
        MS_LOG(EXCEPTION) << "Need to process fn_leaf at line " << lexer_.GetLineNo();
      }
      return next;
    } else if (Match(id, "FuncGraph::")) {
      std::string func_graph_name = id.substr(strlen("FuncGraph::"));
      // if the graph does not exist, create a null graph, then fill the graph when encounter the definition
      // of the graph
      if (func_graphs_map_.find(func_graph_name) == func_graphs_map_.end()) {
        func_graphs_map_[func_graph_name] = std::make_shared<FuncGraph>();
      }
      *val_ptr = func_graphs_map_[func_graph_name];
      return lexer_.GetNextToken();
    } else if (Match(id, "NameSpace::")) {
      std::string module_name = id.substr(strlen("NameSpace::"));
      if (lexer_.GetNextToken() != TOK_AT_FILE) {
        MS_LOG(ERROR) << "Expect TOK_AT_FILE at line " << lexer_.GetLineNo();
        return TOK_ERROR;
      }
      // load Python module information from serialized file
      py::object py_obj = LoadObject(lexer_.GetTokenText());
      *val_ptr = std::make_shared<parse::NameSpace>(module_name, py_obj);

      return lexer_.GetNextToken();
    } else {
      MS_LOG(EXCEPTION) << "Unknown id " << id << " at line " << lexer_.GetLineNo();
    }
  }

  Token ParseValueBasic(const FuncGraphPtr &func_graph, const std::string &id, ValuePtr *const val_ptr,
                        AnfNodePtr *const node_ptr = nullptr) {
    if (id == "None") {
      *val_ptr = std::make_shared<None>();
      return lexer_.GetNextToken();
    } else if (id == "Bool") {
      return ParseScalar<BoolImm, bool, Bool>(val_ptr, lexer_.GetNextToken());
    } else if (id == "I8") {
      return ParseScalar<Int8Imm, int8_t, Int, 8>(val_ptr, lexer_.GetNextToken());
    } else if (id == "I16") {
      return ParseScalar<Int16Imm, int16_t, Int, 16>(val_ptr, lexer_.GetNextToken());
    } else if (id == "I32") {
      return ParseScalar<Int32Imm, int32_t, Int, 32>(val_ptr, lexer_.GetNextToken());
    } else if (id == "I64") {
      return ParseScalar<Int64Imm, int64_t, Int, 64>(val_ptr, lexer_.GetNextToken());
    } else if (id == "U8") {
      return ParseScalar<UInt8Imm, uint8_t, UInt, 8>(val_ptr, lexer_.GetNextToken());
    } else if (id == "U16") {
      return ParseScalar<UInt16Imm, uint16_t, UInt, 16>(val_ptr, lexer_.GetNextToken());
    } else if (id == "U32") {
      return ParseScalar<UInt32Imm, uint32_t, UInt, 32>(val_ptr, lexer_.GetNextToken());
    } else if (id == "U64") {
      return ParseScalar<UInt64Imm, uint64_t, UInt, 64>(val_ptr, lexer_.GetNextToken());
    } else if (id == "F16") {
      // Notice: Since there is no basic data type for storing fp16, just use float instead
      return ParseScalar<FP32Imm, float, Float, 16>(val_ptr, lexer_.GetNextToken());
    } else if (id == "F32") {
      return ParseScalar<FP32Imm, float, Float, 32>(val_ptr, lexer_.GetNextToken());
    } else if (id == "F64") {
      return ParseScalar<FP64Imm, double, Float, 64>(val_ptr, lexer_.GetNextToken());
    } else if (id == "Tensor") {
      return ParseTensor(val_ptr);
    } else if (id == "SymInst") {
      return ParseSymbolicKeyInstance(func_graph, node_ptr);
    } else if (id == "Array") {
      TypePtr type = nullptr;
      Token ret = ParseTypeArray(func_graph, lexer_.GetNextToken(), &type);
      *val_ptr = type;
      return ret;
    } else if (Match(id, "PrimitivePy::")) {
      return ParsePrimitivePy(func_graph, id, val_ptr);
    } else if (Match(id, "Primitive::")) {
      *val_ptr = std::make_shared<Primitive>(id.substr(strlen("Primitive::")));
      return lexer_.GetNextToken();
    } else if (Match(id, "GradOperation::")) {
      return ParseValueGradOperation(id.substr(strlen("GradOperation::")), val_ptr);
    } else {
      return ParseValueGraphAndNamespace(id, val_ptr);
    }
  }

  Token SetListOrTupleValue(const FuncGraphPtr &func_graph, Token left_tok, Token next, bool node_is_valid,
                            const std::vector<ValuePtr> &elems, const std::vector<AnfNodePtr> &nodes,
                            ValuePtr *const val_ptr, AnfNodePtr *node_ptr) {
    if (left_tok == TOK_LPARENTHESIS && next == TOK_RPARENTHESIS) {
      if (node_is_valid && node_ptr != nullptr) {
        MS_EXCEPTION_IF_NULL(func_graph);
        *node_ptr = func_graph->NewCNode(nodes);
      } else {
        *val_ptr = std::make_shared<ValueTuple>(elems);
      }
      return lexer_.GetNextToken();
    } else if (left_tok == TOK_LBRACKET && next == TOK_RBRACKET) {
      if (node_is_valid && node_ptr != nullptr) {
        MS_LOG(EXCEPTION) << "Encounter valid node in value list";
      }
      *val_ptr = std::make_shared<ValueList>(elems);
      return lexer_.GetNextToken();
    } else {
      return TOK_ERROR;
    }
  }

  Token ParseListOrTupleValue(const FuncGraphPtr &func_graph, Token tok, ValuePtr *const val_ptr,
                              AnfNodePtr *node_ptr = nullptr) {
    Token left_tok = tok;

    std::vector<ValuePtr> elems;
    std::vector<AnfNodePtr> nodes;
    nodes.push_back(std::make_shared<ValueNode>(std::make_shared<Primitive>("MakeTuple")));
    ValuePtr elem = nullptr;
    AnfNodePtr node = nullptr;
    bool node_is_valid = false;
    bool first_flag = true;
    Token next = TOK_ERROR;
    do {
      next = lexer_.GetNextToken();
      if (first_flag) {
        first_flag = false;
        // case (), zero elements
        if ((left_tok == TOK_LPARENTHESIS && next == TOK_RPARENTHESIS) ||
            (left_tok == TOK_LBRACKET && next == TOK_RBRACKET)) {
          if (left_tok == TOK_LPARENTHESIS) {
            *val_ptr = std::make_shared<ValueTuple>(elems);
          } else {
            *val_ptr = std::make_shared<ValueList>(elems);
          }
          return lexer_.GetNextToken();
        }
      }
      node = nullptr;
      next = ParseValue(func_graph, next, &elem, &node);
      elems.push_back(elem);
      if (node != nullptr) {
        nodes.push_back(node);
        node_is_valid = true;
      } else {
        nodes.push_back(std::make_shared<ValueNode>(elem));
      }
    } while (next == TOK_COMMA);

    return SetListOrTupleValue(func_graph, left_tok, next, node_is_valid, elems, nodes, val_ptr, node_ptr);
  }

  Token ParseValue(const FuncGraphPtr &func_graph, Token tok, ValuePtr *const val_ptr, AnfNodePtr *node_ptr = nullptr) {
    // tuple or list
    if (tok == TOK_LPARENTHESIS || tok == TOK_LBRACKET) {
      return ParseListOrTupleValue(func_graph, tok, val_ptr, node_ptr);
    } else if (tok == TOK_IDENTIFIER) {
      return ParseValueBasic(func_graph, lexer_.GetTokenText(), val_ptr, node_ptr);
    } else if (tok == TOK_STRING) {
      *val_ptr = std::make_shared<StringImm>(lexer_.GetTokenText());
      return lexer_.GetNextToken();
    }
    MS_LOG(ERROR) << "Parse error!";
    return TOK_ERROR;
  }

  Token ParseItem(const FuncGraphPtr &func_graph, AnfNodePtr *node_ptr, ValuePtr *const val_ptr,
                  Token tok = TOK_INVALID) {
    if (tok == TOK_INVALID) {
      tok = lexer_.GetNextToken();
    }
    if (tok == TOK_VARIABLE) {
      auto iter = cnodes_.find(lexer_.GetTokenText());
      if (iter == cnodes_.end()) {
        MS_LOG(EXCEPTION) << "Can not find definition of '" << lexer_.GetTokenText() << "'";
      }
      *node_ptr = iter->second;
    } else if (tok == TOK_PARAMETER) {
      AnfNodePtr param = FindParameter(func_graph, lexer_.GetTokenText());
      if (param == nullptr) {
        MS_LOG(EXCEPTION) << "Can not find definition of '" << lexer_.GetTokenText() << "' at line "
                          << lexer_.GetLineNo();
      }
      *node_ptr = param;
    } else if (tok == TOK_IDENTIFIER || tok == TOK_LPARENTHESIS || tok == TOK_STRING) {
      ValuePtr value;
      AnfNodePtr node;
      tok = ParseValue(func_graph, tok, &value, &node);
      if (tok == TOK_ERROR) {
        MS_LOG(ERROR) << "Parse value error!";
        return tok;
      }
      if (node == nullptr) {
        *val_ptr = value;
        *node_ptr = std::make_shared<ValueNode>(value);
      } else {
        *node_ptr = node;
      }

      return tok;
    } else {
      MS_LOG(EXCEPTION) << "tok_type = " << tok;
    }

    return lexer_.GetNextToken();
  }

  Token ParseArgument(const FuncGraphPtr &func_graph, std::vector<AnfNodePtr> *const inputs_ptr) {
    Token tok = lexer_.GetNextToken();
    if (tok == TOK_RPARENTHESIS) {
      return tok;
    }
    AnfNodePtr node = nullptr;
    ValuePtr value = nullptr;
    tok = ParseItem(func_graph, &node, &value, tok);
    if (tok != TOK_ERROR) {
      MS_EXCEPTION_IF_NULL(inputs_ptr);
      inputs_ptr->push_back(node);
    }
    return tok;
  }

  const std::vector<FuncGraphPtr> &GetFuncGraphs() const { return func_graphs_; }

 private:
  Lexer lexer_;
  std::vector<FuncGraphPtr> func_graphs_;
  bool error_flag_ = false;

  // store all parsed graphs
  std::map<std::string, FuncGraphPtr> func_graphs_map_;
  // map from child to parent, consider adding a 'parent' field in class Graph
  std::map<FuncGraphPtr, FuncGraphPtr> parents_map_;

  // map for buffering cnodes when parsing a graph
  std::map<std::string, CNodePtr> cnodes_;

  std::map<std::string, ParameterPtr> param_nodes_;  // map parameter name to parameter
};

std::vector<FuncGraphPtr> ImportIR(const std::string &filename) {
  IrParser parser(filename.c_str());
  parser.ParseFile();
  return parser.GetFuncGraphs();
}
}  // namespace mindspore

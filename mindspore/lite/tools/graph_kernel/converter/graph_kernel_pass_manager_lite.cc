/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "tools/graph_kernel/converter/graph_kernel_pass_manager_lite.h"

#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <iomanip>
#include <optional>
#include "ir/graph_utils.h"
#include "src/common/file_utils.h"
#include "utils/file_utils.h"
#include "src/common/utils.h"

namespace mindspore::graphkernel {
namespace dumpir {
struct SubGraphIRInfo {
  int32_t local_var;
  std::ostringstream dumpbuf;
  OrderedMap<AnfNodePtr, int32_t> local_var_map;
};

void DumpGlobalInfoEntry(const FuncGraphPtr &graph, std::ostringstream &dumpbuf) {
  if (graph == nullptr) {
    return;
  }
  dumpbuf << "#IR entry      : @" << graph->ToString() << std::endl;
  dumpbuf << "#attrs         :" << std::endl;
  for (const auto &attr : graph->attrs()) {
    dumpbuf << attr.first << " : ";
    if (attr.second->isa<BoolImm>()) {
      dumpbuf << (GetValue<bool>(attr.second));
    } else if (attr.second->isa<StringImm>()) {
      dumpbuf << (GetValue<std::string>(attr.second));
    }
    dumpbuf << std::endl;
  }
}

void PrintNodeOutputType(std::ostringstream &dumpbuf, const AnfNodePtr &nd) {
  if (nd == nullptr) {
    return;
  }
  ValuePtr tensor_value = nullptr;
  auto abstract = nd->abstract();
  if (abstract != nullptr && abstract->isa<abstract::AbstractTensor>()) {
    tensor_value = abstract->BuildValue();
  }
  abstract::ShapePtr shape = dyn_cast<abstract::Shape>(nd->Shape());
  TypePtr type = dyn_cast<Type>(nd->Type());
  if ((shape != nullptr) && (type != nullptr)) {
    dumpbuf << "<" << type << ", " << shape->ToString();
    if (tensor_value != nullptr && tensor_value != kAnyValue) {
      dumpbuf << ", value=...";
    }
    dumpbuf << ">";
  } else if (type != nullptr) {
    dumpbuf << "<" << type;
    if (tensor_value != nullptr && tensor_value != kAnyValue) {
      dumpbuf << ", value=...";
    }
    dumpbuf << ">";
  } else {
    dumpbuf << "<null>";
  }
}

int32_t DumpParams(const FuncGraphPtr &graph, std::ostringstream &dumpbuf, OrderedMap<AnfNodePtr, int32_t> *para_map) {
  if (graph == nullptr) {
    MS_LOG(INFO) << "Param graph is nullptr.";
    return 0;
  }
  std::vector<AnfNodePtr> parameters = graph->parameters();
  dumpbuf << "#Total params  : " << parameters.size() << std::endl;
  dumpbuf << std::endl;

  // dump parameters
  int32_t para = 1;
  for (const auto &p : parameters) {
    if (p == nullptr) {
      continue;
    }
    auto parameter_ptr = p->cast<ParameterPtr>();
    if (parameter_ptr == nullptr) {
      MS_LOG(EXCEPTION) << "p cannot cast to ParameterPtr";
    }
    dumpbuf << "%para" << para << "_" << parameter_ptr->name() << " : ";
    // print parameters' type and shape
    PrintNodeOutputType(dumpbuf, p);
    dumpbuf << std::endl;

    if (para_map != nullptr) {
      (*para_map)[p] = para++;
    }
    MS_LOG(DEBUG) << "Record param: " << p->ToString() << " graph belong : " << p->func_graph()->ToString();
  }
  return para;
}

void DumpOperator(const AnfNodePtr &op, const std::shared_ptr<SubGraphIRInfo> &gsub) {
  if (op == nullptr || gsub == nullptr) {
    return;
  }

  if (IsValueNode<FuncGraph>(op)) {
    FuncGraphPtr fg = GetValueNode<FuncGraphPtr>(op);
    if (fg != nullptr) {
      gsub->dumpbuf << "call @" << fg->ToString();
    }
  } else if (op->isa<CNode>()) {
    if (gsub->local_var_map.find(op) != gsub->local_var_map.end()) {
      gsub->dumpbuf << "%" << gsub->local_var_map[op];
    } else {
      auto node = op->cast<CNodePtr>();
      auto fg = node->func_graph();
      gsub->dumpbuf << "$(" << fg->ToString() << ":" << node->ToString() << ")";
    }
  } else if (op->isa<ValueNode>()) {
    gsub->dumpbuf << GetValueNode(op)->ToString();
  } else {
    gsub->dumpbuf << op->ToString();
  }
}

void DumpOperands(const AnfNodePtr &nd, OrderedMap<AnfNodePtr, int32_t> *para_map,
                  const std::shared_ptr<SubGraphIRInfo> &gsub) {
  if (nd == nullptr || para_map == nullptr || gsub == nullptr) {
    return;
  }

  gsub->dumpbuf << "(";
  const auto &inputs = GetInputs(nd);
  size_t len = inputs.size();
  if (len > 1) {
    // skip inputs[0] which is Primitive valuenode
    for (size_t i = 1; i < len; ++i) {
      AnfNodePtr in = inputs[i];
      MS_EXCEPTION_IF_NULL(in);
      if (i != 1) {
        gsub->dumpbuf << ", ";
      }
      if (in->isa<Parameter>()) {
        if (!(*para_map)[in]) {
          gsub->dumpbuf << "%para_" << in->ToString();
        } else {
          gsub->dumpbuf << "%para" << (*para_map)[in] << "_" << in->ToString();
        }
      } else if (in->isa<CNode>()) {
        if (gsub->local_var_map.find(in) != gsub->local_var_map.end()) {
          gsub->dumpbuf << "%" << gsub->local_var_map[in];
        } else {
          auto node = in->cast<CNodePtr>();
          auto fg = node->func_graph();
          gsub->dumpbuf << "$(" << fg->ToString() << ":" << node->ToString() << ")";
        }
      } else if (in->isa<ValueNode>() && !IsValueNode<FuncGraph>(in)) {
        // non Primitive valuenode
        gsub->dumpbuf << GetValueNode(in)->ToString();
      } else if (IsValueNode<FuncGraph>(in)) {
        FuncGraphPtr fg = GetValueNode<FuncGraphPtr>(in);
        gsub->dumpbuf << "@" << fg->ToString();
      } else {
        gsub->dumpbuf << in->ToString();
      }
    }
  }
  gsub->dumpbuf << ")";
}

void DumpAttrs(const mindspore::HashMap<std::string, ValuePtr> &attrs, const std::shared_ptr<SubGraphIRInfo> &gsub,
               bool check_strategy = false) {
  int i = 0;
  for (const auto &attr : attrs) {
    if (i++ != 0) {
      gsub->dumpbuf << ", ";
    }
    gsub->dumpbuf << attr.first << ": ";
    if (attr.second == nullptr) {
      gsub->dumpbuf << "null";
    } else {
      gsub->dumpbuf << attr.second->ToString();
    }
  }
}

void DumpOperateAttrs(const AnfNodePtr &op, const std::shared_ptr<SubGraphIRInfo> &gsub) {
  if (op == nullptr || gsub == nullptr) {
    return;
  }

  if (IsValueNode<Primitive>(op)) {
    auto primitive = GetValueNode<PrimitivePtr>(op);
    if (!primitive->instance_name().empty()) {
      gsub->dumpbuf << " {";
      gsub->dumpbuf << "instance name"
                    << ": ";
      gsub->dumpbuf << primitive->instance_name();
      gsub->dumpbuf << "}";
    }
    auto attrs = primitive->attrs();
    if (!attrs.empty()) {
      gsub->dumpbuf << " primitive_attrs: {";
      DumpAttrs(attrs, gsub, true);
      gsub->dumpbuf << "}";
    }
  }
}

void DumpCNodeAttrs(const CNodePtr &op, const std::shared_ptr<SubGraphIRInfo> &gsub) {
  if (op == nullptr || gsub == nullptr) {
    return;
  }
  auto &attrs = op->attrs();
  if (attrs.empty()) {
    return;
  }

  gsub->dumpbuf << " cnode_attrs: {";
  DumpAttrs(attrs, gsub);
  gsub->dumpbuf << "}";
}

void DumpCNodePrimalAttrs(const CNodePtr &op, const std::shared_ptr<SubGraphIRInfo> &gsub) {
  if (op == nullptr || gsub == nullptr) {
    return;
  }
  if (op->primal_attrs().empty()) {
    gsub->dumpbuf << std::endl;
    return;
  }
  auto primal_attrs = op->primal_attrs();
  gsub->dumpbuf << " cnode_primal_attrs: {";
  DumpAttrs(primal_attrs, gsub);
  gsub->dumpbuf << "}";
  gsub->dumpbuf << std::endl;
}

void PrintNodeInputType(std::ostringstream &dumpbuf, const AnfNodePtr &nd) {
  if (nd == nullptr) {
    return;
  }
  const auto &inputs = GetInputs(nd);
  size_t len = inputs.size();
  if (len > 1) {
    // skip inputs[0] which is Primitive value node
    for (size_t i = 1; i < len; ++i) {
      AnfNodePtr in = inputs[i];
      if (i != 1) {
        dumpbuf << ", ";
      }
      PrintNodeOutputType(dumpbuf, in);
    }
  }
}

void DumpShape(const AnfNodePtr &nd, const FuncGraphPtr &sub_graph, const std::shared_ptr<SubGraphIRInfo> &gsub) {
  if (nd == nullptr || sub_graph == nullptr || gsub == nullptr) {
    return;
  }

  if (nd != sub_graph->get_return()) {
    gsub->dumpbuf << "      : (";
    PrintNodeInputType(gsub->dumpbuf, nd);
    gsub->dumpbuf << ") -> (";
    PrintNodeOutputType(gsub->dumpbuf, nd);
    gsub->dumpbuf << ")";
  } else {
    gsub->dumpbuf << "      : (";
    PrintNodeInputType(gsub->dumpbuf, nd);
    gsub->dumpbuf << ")";
  }

  gsub->dumpbuf << std::endl;
}

void DumpCNode(const CNodePtr &nd, const FuncGraphPtr &sub_graph, OrderedMap<AnfNodePtr, int32_t> *const para_map,
               const std::shared_ptr<SubGraphIRInfo> &gsub, bool dump_full_name = false) {
  if (nd == nullptr || sub_graph == nullptr || para_map == nullptr || gsub == nullptr) {
    return;
  }

  if (nd != sub_graph->get_return()) {
    gsub->dumpbuf << "  %" << gsub->local_var << "(" << nd->ToString() << ")"
                  << " = ";
    gsub->local_var_map[nd] = gsub->local_var++;
  } else {
    gsub->dumpbuf << "  ";
  }

  if (nd->inputs().empty()) {
    MS_LOG(EXCEPTION) << "Input of apply node is empty";
  }

  AnfNodePtr op = nd->input(0);
  DumpOperator(op, gsub);
  DumpOperands(nd, para_map, gsub);
  DumpOperateAttrs(op, gsub);
  DumpCNodeAttrs(nd, gsub);
  DumpCNodePrimalAttrs(nd, gsub);
  DumpShape(nd, sub_graph, gsub);
  if (dump_full_name) {
    gsub->dumpbuf << "      : (" << nd->fullname_with_scope() << ")" << std::endl;
  }
}

void DumpIRInSubgraph(const std::vector<AnfNodePtr> &nodes, OrderedMap<AnfNodePtr, int32_t> *para_map,
                      OrderedMap<FuncGraphPtr, std::shared_ptr<SubGraphIRInfo>> *const sub_graphs, int32_t total_para,
                      bool dump_full_name = false) {
  if (para_map == nullptr || sub_graphs == nullptr) {
    return;
  }

  for (const auto &nd : nodes) {
    MS_EXCEPTION_IF_NULL(nd);
    FuncGraphPtr sub_graph = nd->func_graph();
    if (sub_graph == nullptr) {
      MS_LOG(DEBUG) << "Node[" << nd->ToString() << "] belongs to no graph!";
      continue;
    }
    std::shared_ptr<SubGraphIRInfo> gsub = (*sub_graphs)[sub_graph];
    if (gsub == nullptr) {
      gsub = std::make_shared<SubGraphIRInfo>();
      gsub->local_var = 0;
      (*sub_graphs)[sub_graph] = gsub;
    }
    auto &param = sub_graph->parameters();
    for (size_t idx = 0; idx < param.size(); idx++) {
      MS_EXCEPTION_IF_NULL(param[idx]);
      if ((*para_map).count(param[idx]) == 0) {
        (*para_map)[param[idx]] = total_para++;
      }
    }
    if (!nd->isa<Parameter>()) {
      if (nd->isa<CNode>()) {
        // print and record output of operator if it is not 'Return'
        DumpCNode(nd->cast<CNodePtr>(), sub_graph, para_map, gsub, dump_full_name);
      } else {
        gsub->dumpbuf << "  " << nd->ToString() << std::endl;
      }
    }
  }
}

void DumpSubgraph(const OrderedMap<FuncGraphPtr, std::shared_ptr<SubGraphIRInfo>> *sub_graphs,
                  const FuncGraphPtr &graph, OrderedMap<AnfNodePtr, int32_t> *para_map, std::ofstream &fout) {
  if (sub_graphs == nullptr || graph == nullptr) {
    return;
  }

  fout << "#Total subgraph : " << sub_graphs->size() << std::endl;
  fout << std::endl;

  for (const auto &sg : *sub_graphs) {
    fout << "subgraph attr:" << std::endl;
    MS_EXCEPTION_IF_NULL(sg.first);
    for (const auto &attr : sg.first->attrs()) {
      fout << attr.first << " : ";
      if (attr.second->isa<BoolImm>()) {
        fout << GetValue<bool>(attr.second);
      } else if (attr.second->isa<StringImm>()) {
        fout << (GetValue<std::string>(attr.second));
      }
      fout << std::endl;
    }
    fout << "subgraph @" << sg.first->ToString() << "(";
    if (sg.first != graph) {
      std::vector<AnfNodePtr> parameters = sg.first->parameters();
      if (parameters.size() == 1) {
        MS_EXCEPTION_IF_NULL(parameters[0]);
        fout << "%para" << (*para_map)[parameters[0]] << "_" << parameters[0]->ToString();
      } else if (parameters.size() > 1) {
        for (size_t idx = 0; idx < parameters.size() - 1; idx++) {
          MS_EXCEPTION_IF_NULL(parameters[idx]);
          fout << "%para" << (*para_map)[parameters[idx]] << "_" << parameters[idx]->ToString();
          fout << ", ";
        }
        MS_EXCEPTION_IF_NULL(parameters[parameters.size() - 1]);
        fout << "%para" << (*para_map)[parameters[parameters.size() - 1]] << "_"
             << parameters[parameters.size() - 1]->ToString();
      }
    }
    fout << ") {" << std::endl;
    MS_EXCEPTION_IF_NULL(sg.second);
    fout << sg.second->dumpbuf.str();
    fout << "}" << std::endl;
    fout << std::endl;
  }
}

std::optional<std::string> CreatePrefixPath(const std::string &input_path) {
  std::optional<std::string> prefix_path;
  std::optional<std::string> file_name;
  FileUtils::SplitDirAndFileName(input_path, &prefix_path, &file_name);
  if (!file_name.has_value()) {
    MS_LOG(ERROR) << "Cannot get file_name from: " << input_path;
    return std::nullopt;
  }
  auto file_name_str = file_name.value();
  std::string prefix_path_str;
  if (prefix_path.has_value()) {
    auto create_prefix_path = FileUtils::CreateNotExistDirs(prefix_path.value(), true);
    if (!create_prefix_path.has_value()) {
      return std::nullopt;
    }
    prefix_path_str = create_prefix_path.value();
  } else {
    auto pwd_path = FileUtils::GetRealPath("./");
    if (!pwd_path.has_value()) {
      MS_LOG(ERROR) << "Can not get pwd path";
      return std::nullopt;
    }
    prefix_path_str = pwd_path.value();
  }
  return std::string(prefix_path_str + "/" + file_name_str);
}

void DumpIR(const std::string &filename, const FuncGraphPtr &graph, bool dump_full_name) {
  if (graph == nullptr) {
    return;
  }
  auto path = "./" + filename;
  auto realpath = CreatePrefixPath(path);
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Get real path failed, path=" << path;
    return;
  }

  std::ofstream fout(realpath.value());
  std::ostringstream dumpbuf;

  auto nodes = TopoSort(graph->get_return(), SuccDeeperSimple, AlwaysInclude);
  OrderedMap<AnfNodePtr, int32_t> para_map;
  // dump global info
  DumpGlobalInfoEntry(graph, dumpbuf);
  int32_t total_para = DumpParams(graph, dumpbuf, &para_map);

  OrderedMap<FuncGraphPtr, std::shared_ptr<SubGraphIRInfo>> sub_graphs;
  // dump ir in each sub graph
  DumpIRInSubgraph(nodes, &para_map, &sub_graphs, total_para, dump_full_name);

  // output global info
  fout << dumpbuf.str() << std::endl;

  // output each sub graph
  DumpSubgraph(&sub_graphs, graph, &para_map, fout);

  fout.close();
}
}  // namespace dumpir

void GraphKernelPassManagerLite::DumpPassIR(const FuncGraphPtr &func_graph, const std::string &pass_fullname) const {
  static bool dumpir = (common::GetEnv("MS_DEV_DUMP_GRAPH_KERNEL_IR") == "on");
  if (dumpir) {
    std::string filename = "verbose_ir_files/" + pass_fullname + ".ir";
    dumpir::DumpIR(filename, func_graph, true);
  }
}

// transplant this function from pass_manager_extends.cc because the implement was moved to PassManagerLite.
bool GraphKernelPassManagerLite::RunPass(const FuncGraphPtr &func_graph, size_t pass_id, const PassPtr &pass) const {
  bool changed = false;
  auto begin_time = lite::GetTimeUs();
  if (pass->Run(func_graph)) {
    changed = true;
  }
  auto end_time = lite::GetTimeUs();
  MS_LOG(INFO) << "Run pass " << GetPassFullname(pass_id, pass) << " in " << (end_time - begin_time) << " us.";
  return changed;
}
}  // namespace mindspore::graphkernel

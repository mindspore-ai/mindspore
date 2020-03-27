/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "debug/anf_ir_dump.h"
#include <fstream>
#include <map>
#include <memory>
#include <unordered_map>

#include "ir/primitive.h"
#include "ir/func_graph.h"
#include "device/kernel_info.h"
#include "utils/graph_utils.h"
#include "session/anf_runtime_algorithm.h"

namespace mindspore {
const std::string ToShortString(const TypeId &typeId) {
  std::string label = TypeIdLabel(typeId);
  std::string prefix = "kNumberType";
  if (prefix.length() > label.length()) {
    return label;
  }
  auto position = label.find(prefix);
  // position is 0 when label begins with prefix
  if (position != 0) {
    return label;
  }
  auto sub_position = position + prefix.length();
  if (sub_position >= label.length()) {
    return label;
  }
  return label.substr(sub_position);
}

void PrintKernelFormatAndType(std::ostringstream &buffer, const std::string &fmt, const TypeId &type,
                              const std::vector<size_t> &shape) {
  buffer << "<" << ToShortString(type);
  if (!fmt.empty()) {
    buffer << "x" << fmt << shape;
  }
  buffer << ">";
}

void PrintNodeOutputType(std::ostringstream &buffer, const AnfNodePtr &nd) {
  if (nd == nullptr) {
    return;
  }

  abstract::ShapePtr shape = dyn_cast<abstract::Shape>(nd->Shape());
  TypePtr type = dyn_cast<Type>(nd->Type());
  if ((nullptr != shape) && (nullptr != type)) {
    buffer << "<" << type << "x" << shape->shape() << ">";
  } else if (nullptr != type) {
    buffer << "<" << type << ">";
  } else {
    buffer << "<null>";
  }
}

void PrintNodeInputType(std::ostringstream &buffer, const AnfNodePtr &nd) {
  if (nd == nullptr) {
    return;
  }

  std::vector<AnfNodePtr> inputs = SuccIncoming(nd);
  size_t len = inputs.size();
  if (len > 1) {
    // skip inputs[0] which is Primitive value node
    for (size_t i = 1; i < len; ++i) {
      AnfNodePtr in = inputs[i];
      if (i != 1) {
        buffer << ", ";
      }
      PrintNodeOutputType(buffer, in);
    }
  }
}

struct SubGraphIRInfo {
  int32_t local_var;
  std::ostringstream buffer;
  OrderedMap<AnfNodePtr, int32_t> local_var_map;
};

void DumpGrobalInfoEntry(const FuncGraphPtr &graph, std::ostringstream &buffer) {
  if (graph == nullptr) {
    return;
  }

  buffer << "#IR entry      : @" << graph->ToString() << "." << graph->debug_info()->get_id() << std::endl;
  buffer << "#flags         :" << std::endl;
  for (const auto &flag : graph->flags()) {
    buffer << flag.first << " : " << flag.second << std::endl;
  }
}

void DumpKernelInfo(const CNodePtr &node, const std::shared_ptr<SubGraphIRInfo> &gsub) {
  if (node == nullptr || gsub == nullptr) {
    return;
  }
  auto kernel_info = node->kernel_info();
  if (kernel_info == nullptr || kernel_info->select_kernel_build_info() == nullptr) {
    return;
  }

  gsub->buffer << "      : (";
  for (size_t i = 0; i < AnfAlgo::GetInputTensorNum(node); ++i) {
    if (i != 0) {
      gsub->buffer << ", ";
    }
    auto format = AnfAlgo::GetInputFormat(node, i);
    auto type = AnfAlgo::GetInputDeviceDataType(node, i);
    auto shape = AnfAlgo::GetInputDeviceShape(node, i);
    PrintKernelFormatAndType(gsub->buffer, format, type, shape);
  }
  gsub->buffer << ") -> (";
  for (size_t i = 0; i < AnfAlgo::GetOutputTensorNum(node); ++i) {
    if (i != 0) {
      gsub->buffer << ", ";
    }
    auto format = AnfAlgo::GetOutputFormat(node, i);
    auto type = AnfAlgo::GetOutputDeviceDataType(node, i);
    auto shape = AnfAlgo::GetOutputDeviceShape(node, i);
    PrintKernelFormatAndType(gsub->buffer, format, type, shape);
  }
  gsub->buffer << ")";
  gsub->buffer << std::endl;
}

void DumpParams(const FuncGraphPtr &graph, std::ostringstream &buffer, OrderedMap<AnfNodePtr, int32_t> *para_map) {
  if (graph == nullptr) {
    MS_LOG(INFO) << "param graph is nullptr.";
    return;
  }
  std::vector<AnfNodePtr> parameters = graph->parameters();
  buffer << "#Total params  : " << parameters.size() << std::endl;
  buffer << std::endl;

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
    buffer << "%para" << para << " = " << parameter_ptr->name() << " : ";
    // print parameters' type and shape
    PrintNodeOutputType(buffer, p);
    auto kernel_info = p->kernel_info();
    if (kernel_info != nullptr && kernel_info->select_kernel_build_info() != nullptr) {
      buffer << "  :  ";
      auto type = AnfAlgo::GetOutputDeviceDataType(p, 0);
      auto format = AnfAlgo::GetOutputFormat(p, 0);
      auto shape = AnfAlgo::GetOutputDeviceShape(p, 0);
      PrintKernelFormatAndType(buffer, format, type, shape);
      buffer << "  :  IsWeight:" << std::boolalpha << AnfAlgo::IsParameterWeight(parameter_ptr);
    }
    buffer << std::endl;

    if (para_map != nullptr) {
      (*para_map)[p] = para++;
    }
    MS_LOG(DEBUG) << "record param: " << p->ToString() << " graph belong : " << p->func_graph()->ToString();
  }
}

void DumpOperator(const AnfNodePtr &op, const std::shared_ptr<SubGraphIRInfo> &gsub) {
  if (op == nullptr) {
    MS_LOG(INFO) << "param op is nullptr";
    return;
  }
  if (gsub == nullptr) {
    MS_LOG(INFO) << "param gsub is nullptr";
    return;
  }

  if (IsValueNode<FuncGraph>(op)) {
    FuncGraphPtr fg = GetValueNode<FuncGraphPtr>(op);
    if (fg != nullptr) {
      gsub->buffer << "call @" << fg->ToString() << "." << fg->debug_info()->get_id();
    }
  } else if (op->isa<CNode>()) {
    if (gsub->local_var_map.find(op) != gsub->local_var_map.end()) {
      gsub->buffer << "%" << gsub->local_var_map[op];
    }
  } else if (op->isa<ValueNode>()) {
    gsub->buffer << GetValueNode(op)->ToString();
  } else {
    gsub->buffer << op->ToString();
  }
}

void DumpOperands(const AnfNodePtr &nd, OrderedMap<AnfNodePtr, int32_t> *para_map,
                  const std::shared_ptr<SubGraphIRInfo> &gsub) {
  if (nd == nullptr || para_map == nullptr || gsub == nullptr) {
    return;
  }

  gsub->buffer << "(";
  std::vector<AnfNodePtr> inputs = SuccIncoming(nd);
  size_t len = inputs.size();
  if (len > 1) {
    // skip inputs[0] which is Primitive valuenode
    for (size_t i = 1; i < len; ++i) {
      AnfNodePtr in = inputs[i];
      MS_EXCEPTION_IF_NULL(in);
      if (i != 1) {
        gsub->buffer << ", ";
      }
      if (in->isa<Parameter>()) {
        if (!(*para_map)[in]) {
          gsub->buffer << "%arg";
        } else {
          gsub->buffer << "%para" << (*para_map)[in];
        }
      } else if (in->isa<CNode>()) {
        gsub->buffer << "%" << gsub->local_var_map[in];
      } else if (in->isa<ValueNode>() && !IsValueNode<FuncGraph>(in)) {
        // non Primitive valuenode
        gsub->buffer << GetValueNode(in)->ToString();
      } else if (IsValueNode<FuncGraph>(in)) {
        FuncGraphPtr fg = GetValueNode<FuncGraphPtr>(in);
        gsub->buffer << "@" << fg->ToString() << "." << fg->debug_info()->get_id();
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

  auto operator_info = node->operator_info();
  if (operator_info == nullptr) {
    return;
  }

  auto strategy = operator_info->strategy();
  if (strategy == nullptr) {
    return;
  }

  ValuePtr temp = MakeValue(strategy->GetInputDim());
  gsub->buffer << " { strategy: ";
  gsub->buffer << temp->ToString();
  gsub->buffer << " }";
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
      gsub->buffer << " {";
      int i = 0;
      for (const auto &attr : attrs) {
        if (attr.first == PARALLEL_STRATEGY) {
          continue;  // skip the strategy
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
      gsub->buffer << "}";
    }
  }
  gsub->buffer << std::endl;
}

void DumpShape(const AnfNodePtr &nd, const FuncGraphPtr &sub_graph, const std::shared_ptr<SubGraphIRInfo> &gsub) {
  if (nd == nullptr || sub_graph == nullptr || gsub == nullptr) {
    return;
  }

  if (nd != sub_graph->get_return()) {
    gsub->buffer << "      : (";
    PrintNodeInputType(gsub->buffer, nd);
    gsub->buffer << ") -> (";
    PrintNodeOutputType(gsub->buffer, nd);
    gsub->buffer << ")";
  } else {
    gsub->buffer << "      : (";
    PrintNodeInputType(gsub->buffer, nd);
    gsub->buffer << ")";
  }

  gsub->buffer << std::endl;
}

void DumpCNode(const CNodePtr &nd, const FuncGraphPtr &sub_graph, OrderedMap<AnfNodePtr, int32_t> *const para_map,
               const std::shared_ptr<SubGraphIRInfo> &gsub, bool dump_full_name = false) {
  if (nd == nullptr || sub_graph == nullptr || para_map == nullptr || gsub == nullptr) {
    return;
  }

  if (nd != sub_graph->get_return()) {
    gsub->buffer << "  %" << gsub->local_var << "(" << nd->ToString() << ")"
                 << " = ";
    gsub->local_var_map[nd] = gsub->local_var++;
  } else {
    gsub->buffer << "  ";
  }

  if (nd->inputs().empty()) {
    MS_LOG(EXCEPTION) << "input of apply node is empty";
  }

  // print operator
  AnfNodePtr op = nd->input(0);
  DumpOperator(op, gsub);

  // print operands
  DumpOperands(nd, para_map, gsub);

  // print operator attrs
  DumpOperateAttrs(op, gsub);

  // print parallel info
  DumpParallelInfo(nd, gsub);

  // print shape info
  DumpShape(nd, sub_graph, gsub);

  // print kernel info
  DumpKernelInfo(nd, gsub);

  if (dump_full_name) {
    gsub->buffer << "      : (" << nd->fullname_with_scope() << ")" << std::endl;
  }
}

void DumpIRInSubgraph(const std::vector<AnfNodePtr> &nodes, OrderedMap<AnfNodePtr, int32_t> *para_map,
                      OrderedMap<FuncGraphPtr, std::shared_ptr<SubGraphIRInfo>> *const sub_graphs,
                      bool dump_full_name = false) {
  if (para_map == nullptr || sub_graphs == nullptr) {
    return;
  }

  for (const auto &nd : nodes) {
    MS_EXCEPTION_IF_NULL(nd);
    FuncGraphPtr sub_graph = nd->func_graph();
    if (sub_graph == nullptr) {
      MS_LOG(DEBUG) << "node[" << nd->ToString() << "] belongs to no graph!";
      continue;
    }
    std::shared_ptr<SubGraphIRInfo> gsub = (*sub_graphs)[sub_graph];
    if (gsub == nullptr) {
      gsub = std::make_shared<SubGraphIRInfo>();
      gsub->local_var = 0;
      (*sub_graphs)[sub_graph] = gsub;
    }
    if (!nd->isa<Parameter>()) {
      if (nd->isa<CNode>()) {
        // print and record output of operator if it is not 'Return'
        DumpCNode(nd->cast<CNodePtr>(), sub_graph, para_map, gsub, dump_full_name);
      } else {
        gsub->buffer << "  " << nd->ToString() << std::endl;
      }
    }
  }
}

void DumpSubgraph(const OrderedMap<FuncGraphPtr, std::shared_ptr<SubGraphIRInfo>> *sub_graphs,
                  const FuncGraphPtr &graph, std::ofstream &fout) {
  if (sub_graphs == nullptr || graph == nullptr) {
    return;
  }

  fout << "#Total subgraph : " << sub_graphs->size() << std::endl;
  fout << std::endl;

  for (const auto &sg : *sub_graphs) {
    fout << "subgraph flag:" << std::endl;
    MS_EXCEPTION_IF_NULL(sg.first);
    for (const auto &flag : sg.first->flags()) {
      fout << flag.first << " : " << flag.second << std::endl;
    }
    fout << "subgraph @" << sg.first->ToString() << ".";
    fout << sg.first->debug_info()->get_id() << "(";
    if (sg.first != graph) {
      fout << "%arg";
    }
    fout << ") {" << std::endl;
    MS_EXCEPTION_IF_NULL(sg.second);
    fout << sg.second->buffer.str();
    fout << "}" << std::endl;
    fout << std::endl;
  }
}

#ifdef ENABLE_DUMP_IR
void DumpIR(const std::string &filename, const FuncGraphPtr &graph, bool dump_full_name) {
  if (graph == nullptr) {
    return;
  }
  if (filename.size() > PATH_MAX) {
    MS_LOG(ERROR) << "file path " << filename << " is too long.";
    return;
  }
  char real_path[PATH_MAX] = {0};
  if (nullptr == realpath(filename.c_str(), real_path)) {
    MS_LOG(DEBUG) << "dir " << filename << " does not exit.";
  }

  OrderedMap<AnfNodePtr, int32_t> para_map;
  std::string path_string = real_path;
  ChangeFileMode(path_string, S_IRWXU);
  std::ofstream fout(real_path);
  std::ostringstream buffer;

  if (!fout.is_open()) {
    MS_LOG(ERROR) << "Open dump file '" << real_path << "' failed!";
    return;
  }

  auto nodes = TopoSort(graph->get_return(), SuccDeeperSimple, AlwaysInclude);

  // dump global info
  DumpGrobalInfoEntry(graph, buffer);
  DumpParams(graph, buffer, &para_map);

  OrderedMap<FuncGraphPtr, std::shared_ptr<SubGraphIRInfo>> sub_graphs;
  // dump ir in each sub graph
  DumpIRInSubgraph(nodes, &para_map, &sub_graphs, dump_full_name);

  // output global info
  fout << buffer.str() << std::endl;

  // output each sub graph
  DumpSubgraph(&sub_graphs, graph, fout);

  fout.close();
  // set file mode to read only by user
  ChangeFileMode(path_string, S_IRUSR);
}
#else
void DumpIR(const std::string &, const FuncGraphPtr &, bool) {
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

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

#include "frontend/parallel/graph_util/get_parallel_info.h"

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include "ir/func_graph.h"
#include "frontend/parallel/ops_info/operator_info.h"
#include "frontend/parallel/graph_util/graph_info.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/tensor_layout/tensor_layout.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "frontend/parallel/parameter_manager.h"

namespace mindspore {
namespace parallel {
namespace {
constexpr char INPUTS[] = "inputs";
constexpr char ATTRS[] = "attrs";
using FuncGraphNameMap = const std::unordered_map<FuncGraphPtr, std::string>;
static std::unordered_map<std::string, size_t> op_count = {};
static std::unordered_map<CNodePtr, std::string> name_map = {};

// Extract the op name and the topology number of the same node in the graph
// e.g, Default/Mul-op32 -> Mul-op0, Default/Mul-op35 -> Mul-op1
std::string GetNodeNameWithCount(const CNodePtr &cnode) {
  if (name_map.find(cnode) != name_map.end()) {
    return name_map[cnode];
  }

  std::string node_name;
  auto is_call_fullname_with_scope = [](const CNodePtr &cnode) {
    auto value_ptr = cnode->input(0)->cast<ValueNodePtr>();
    ValuePtr input_value = nullptr;
    if (value_ptr != nullptr) {
      input_value = value_ptr->value();
    }
    if (input_value != nullptr && input_value->cast<PrimitivePtr>() == nullptr &&
        input_value->cast<FuncGraphPtr>() == nullptr) {
      return false;
    }
    return true;
  };
  if (is_call_fullname_with_scope(cnode)) {
    auto node_name_with_scope = cnode->fullname_with_scope();
    size_t left = node_name_with_scope.rfind('/');
    size_t right = node_name_with_scope.find("-op");
    node_name = node_name_with_scope.substr(left + 1, right - left - 1);
  } else {
    node_name = cnode->ToString();
  }

  std::ostringstream oss;
  oss << node_name << '-' << op_count[node_name];
  name_map[cnode] = oss.str();
  ++op_count[node_name];
  return name_map[cnode];
}

// Renames sub-graphs according to the topology order, e.g, @5_construct.395 -> @graph_0
FuncGraphNameMap GetAllFuncGraphNameMap(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto anf_nodes = TopoSort(graph->get_return(), SuccDeeperSimple, AlwaysInclude);
  std::unordered_map<FuncGraphPtr, std::string> graph_name_map;
  size_t graph_count = 0;
  for (const auto &anf_node : anf_nodes) {
    auto belong_graph = anf_node->func_graph();
    if (belong_graph == nullptr) {
      continue;
    }
    if (graph_name_map.find(belong_graph) == graph_name_map.end()) {
      std::ostringstream oss;
      oss << "@graph_" << graph_count++;
      graph_name_map[belong_graph] = oss.str();
      oss.clear();
    }
  }
  return graph_name_map;
}

// Extract operator name from cnode
std::string GetCNodeOperatorNameWithCount(const CNodePtr &cnode, const FuncGraphNameMap &func_name_map) {
  AnfNodePtr op = cnode->input(0);
  MS_EXCEPTION_IF_NULL(op);
  std::string op_name;
  if (IsValueNode<FuncGraph>(op)) {
    const FuncGraphPtr fg = GetValueNode<FuncGraphPtr>(op);
    op_name = "call " + func_name_map.at(fg);
  } else {
    op_name = GetNodeNameWithCount(cnode);
    name_map[cnode] = op_name;
  }
  return op_name;
}

py::int_ GetPyIntValueFromIntegerImm(const ValuePtr &value_node) {
  MS_EXCEPTION_IF_NULL(value_node);
  if (!value_node->isa<IntegerImm>()) {
    MS_LOG(EXCEPTION) << "value_node is not IntegerImm";
  }

  TypePtr data_type = value_node->type();
  MS_EXCEPTION_IF_NULL(data_type);
  TypeId type_id = data_type->type_id();
  switch (type_id) {
    case kNumberTypeInt8:
      return py::int_(GetValue<int8_t>(value_node));
    case kNumberTypeInt16:
      return py::int_(GetValue<int16_t>(value_node));
    case kNumberTypeInt32:
      return py::int_(GetValue<int32_t>(value_node));
    case kNumberTypeInt64:
      return py::int_(GetValue<int64_t>(value_node));
    case kNumberTypeUInt8:
      return py::int_(GetValue<uint8_t>(value_node));
    case kNumberTypeUInt16:
      return py::int_(GetValue<uint16_t>(value_node));
    case kNumberTypeUInt32:
      return py::int_(GetValue<uint32_t>(value_node));
    case kNumberTypeUInt64:
      return py::int_(GetValue<uint64_t>(value_node));
    default:
      MS_LOG(EXCEPTION) << "The data type: " << data_type << " is invalid.";
  }
}

// Extract the list of operand names from cnode
py::list GetCNodeOperandNameList(const CNodePtr &cnode, const FuncGraphNameMap &func_name_map) {
  MS_EXCEPTION_IF_NULL(cnode);

  py::list cnode_inputs_name_list;
  auto cnode_inputs = cnode->inputs();

  // Skip cnode_inputs[0] which is Primitive value node
  for (size_t i = 1; i < cnode_inputs.size(); ++i) {
    const AnfNodePtr &input = cnode_inputs[i];
    MS_EXCEPTION_IF_NULL(input);

    if (input->isa<Parameter>()) {
      cnode_inputs_name_list.append(py::str(std::static_pointer_cast<Parameter>(input)->name()));
    } else if (IsValueNode<FuncGraph>(input)) {
      FuncGraphPtr fg = GetValueNode<FuncGraphPtr>(input);
      cnode_inputs_name_list.append(func_name_map.at(fg));
    } else if (input->isa<CNode>()) {
      cnode_inputs_name_list.append(py::str(GetNodeNameWithCount(input->cast<CNodePtr>())));
    } else if (input->isa<ValueNode>()) {
      auto value_node = GetValueNode(input);
      if (value_node->isa<IntegerImm>()) {
        cnode_inputs_name_list.append(GetPyIntValueFromIntegerImm(value_node));
      } else if (value_node->isa<FP32Imm>()) {
        cnode_inputs_name_list.append(GetValue<float>(value_node));
      } else if (value_node->isa<FP64Imm>()) {
        cnode_inputs_name_list.append(GetValue<double>(value_node));
      } else if (value_node->isa<BoolImm>()) {
        cnode_inputs_name_list.append(GetValue<bool>(value_node));
      } else if (value_node->isa<StringImm>()) {
        cnode_inputs_name_list.append(py::str(GetValue<std::string>(value_node)));
      } else {
        cnode_inputs_name_list.append(py::str(input->ToString()));
      }
    } else {
      cnode_inputs_name_list.append(py::str(input->ToString()));
    }
  }
  return cnode_inputs_name_list;
}

py::dict GetCNodeAttrs(const CNodePtr &cnode) {
  AnfNodePtr op = cnode->input(0);
  if (op == nullptr || !IsValueNode<Primitive>(op)) {
    return py::dict();
  }

  PrimitivePtr primitive = GetValueNode<PrimitivePtr>(op);
  auto attrs = primitive->attrs();
  py::dict cnode_attrs_dict;
  for (const auto &attr : attrs) {
    auto key = attr.first;
    auto value = attr.second;
    if (value->isa<BoolImm>()) {
      cnode_attrs_dict[py::str(key)] = GetValue<bool>(value);
    } else if (value->isa<IntegerImm>()) {
      cnode_attrs_dict[py::str(key)] = GetPyIntValueFromIntegerImm(value);
    } else if (value->isa<FP32Imm>()) {
      cnode_attrs_dict[py::str(key)] = GetValue<float>(value);
    } else if (value->isa<FP64Imm>()) {
      cnode_attrs_dict[py::str(key)] = GetValue<double>(value);
    } else {
      cnode_attrs_dict[py::str(attr.first)] = py::str(attr.second->ToString());
    }
  }
  return cnode_attrs_dict;
}

// Get cnode info dict in subgraph.
py::dict GetParallelCNodeInfoFromSubGraph(const FuncGraphPtr &sub_graph, const FuncGraphNameMap &func_name_map) {
  MS_EXCEPTION_IF_NULL(sub_graph);
  op_count.clear();
  name_map.clear();

  py::dict cnode_info_dict;
  auto cnodes = sub_graph->GetOrderedCnodes();
  for (auto cnode = cnodes.cbegin(); cnode != cnodes.cend(); ++cnode) {
    std::string op_name_with_count = GetCNodeOperatorNameWithCount(*cnode, func_name_map);
    py::dict cnode_info;
    cnode_info[INPUTS] = GetCNodeOperandNameList(*cnode, func_name_map);
    cnode_info[ATTRS] = GetCNodeAttrs(*cnode);
    cnode_info_dict[py::str(op_name_with_count)] = cnode_info;
  }
  return cnode_info_dict;
}
}  // namespace

py::dict GetParameterLayoutFromGraph(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  py::dict dict;
  std::vector<AnfNodePtr> graph_params = graph->parameters();

  for (auto para : graph_params) {
    auto param_ptr = para->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param_ptr);
    std::vector<std::string> names = {param_ptr->name()};
    auto param_info = param_ptr->param_info();
    if (param_info) {
      auto cloned_obj = GetPyParameterObj(param_info, CLONED_OBJ);
      if (!py::isinstance<py::none>(cloned_obj) && py::isinstance<py::list>(cloned_obj)) {
        auto obj_list = py::cast<py::list>(cloned_obj);
        for (size_t i = 0; i < obj_list.size(); ++i) {
          auto each_obj = obj_list[i];
          if (py::hasattr(each_obj, "name")) {
            auto name_obj = python_adapter::GetPyObjAttr(each_obj, "name");
            names.push_back(py::cast<std::string>(name_obj));
          }
        }
      }
    }
    auto tensor_layout = para->user_data<parallel::TensorLayout>();
    if (tensor_layout == nullptr) {
      MS_LOG(INFO) << "GetParameterLayout nullptr parameter: " << para->DebugString();
    } else {
      const auto &device_arrangement = tensor_layout->device_arrangement().array();
      const auto &tensor_map = tensor_layout->tensor_map().array();
      const auto &slice_shape = tensor_layout->slice_shape().array();
      int64_t field_size = tensor_layout->get_field_size();
      bool uniform_split = tensor_layout->uniform_split();
      const std::string &opt_shard_group = tensor_layout->opt_shard_group();
      py::tuple layout =
        py::make_tuple(device_arrangement, tensor_map, slice_shape, field_size, uniform_split, opt_shard_group);
      for (auto &name : names) {
        dict[py::str(name)] = layout;
      }
      MS_LOG(INFO) << "GetParameterLayout parameter: " << para->DebugString() << ", layout "
                   << tensor_layout->ToString();
    }
  }
  return dict;
}

py::dict GetParameterLayoutFromResource(const pipeline::ResourcePtr &resource) {
  py::dict dict;
  const auto &layout_map = resource->layout_map();
  for (auto iter = layout_map.begin(); iter != layout_map.end(); ++iter) {
    auto name = iter->first;
    auto layout = iter->second;
    const auto &device_arrangement = layout->get_device_arrangement();
    const auto &tensor_map = layout->get_tensor_map();
    const auto &slice_shape = layout->get_slice_shape();
    int64_t field_size = layout->get_field_size();
    bool uniform_split = layout->get_uniform_split();
    const std::string &opt_shard_group = layout->get_opt_shard_group();
    py::tuple layout_tuple =
      py::make_tuple(device_arrangement, tensor_map, slice_shape, field_size, uniform_split, opt_shard_group);
    dict[py::str(name)] = layout_tuple;
  }
  return dict;
}

py::dict GetAllreduceFusion(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  py::dict dict;
  auto allreduce_prim_list = FindPrimtive(graph, ALL_REDUCE);

  for (auto prim : allreduce_prim_list) {
    auto name_ptr = prim->GetAttr("parameter");
    auto fusion_ptr = prim->GetAttr("fusion");
    if (fusion_ptr == nullptr) {
      MS_LOG(EXCEPTION) << "fusion_ptr is nullptr";
    } else if (name_ptr == nullptr) {
      continue;
    }
    if (!name_ptr->isa<StringImm>()) {
      MS_LOG(EXCEPTION) << "name is not StringImm";
    }
    auto name = name_ptr->cast<StringImmPtr>()->value();
    if (!fusion_ptr->isa<Int64Imm>()) {
      MS_LOG(EXCEPTION) << "fusion is not Int64Imm";
    }
    int64_t fusion = fusion_ptr->cast<Int64ImmPtr>()->value();
    dict[py::str(name)] = fusion;
  }
  return dict;
}

// In pipeline parallel mode, many parameters are not used and need to be deleted
py::list GetParallelParameterNameListFromGraph(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);

  py::list parallel_parameter_name_list;
  std::vector<AnfNodePtr> graph_params = graph->parameters();

  for (auto param : graph_params) {
    auto param_ptr = std::static_pointer_cast<Parameter>(param);
    MS_EXCEPTION_IF_NULL(param_ptr);
    std::string name = param_ptr->name();
    parallel_parameter_name_list.append(name);
  }
  return parallel_parameter_name_list;
}

py::list GetParallelParameterNameListFromResource(const pipeline::ResourcePtr &resource) {
  auto &layout_map = resource->layout_map();
  py::list parallel_parameter_name_list;
  for (auto iter = layout_map.begin(); iter != layout_map.end(); ++iter) {
    auto name = iter->first;
    parallel_parameter_name_list.append(name);
  }
  return parallel_parameter_name_list;
}

py::dict GetParallelCNodeInfoFromGraph(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  // Search and mapping all subgraph names
  auto func_name_map = GetAllFuncGraphNameMap(graph);
  py::dict parallel_cnode_info_dict;

  // Get cnode info dict in each subgraph in turn
  for (const auto &kv : func_name_map) {
    auto sub_graph_cnode_info_dict = GetParallelCNodeInfoFromSubGraph(kv.first, func_name_map);
    parallel_cnode_info_dict[py::str(kv.second)] = sub_graph_cnode_info_dict;
  }
  op_count.clear();
  name_map.clear();
  return parallel_cnode_info_dict;
}
}  // namespace parallel
}  // namespace mindspore

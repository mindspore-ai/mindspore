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

#include <fstream>
#include <map>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <algorithm>

#include "debug/anf_ir_utils.h"
#include "proto/anf_ir.pb.h"
#include "utils/graph_utils.h"
#include "utils/symbolic.h"

namespace mindspore {
class ProtoExporter {
 public:
  ProtoExporter() {}
  ~ProtoExporter() {}

  std::string GetFuncGraphProtoString(const FuncGraphPtr &func_graph);

 private:
  void InitModelInfo();
  void GetOpNodeTypeAndAttrs(const FuncGraphPtr &func_graph, const AnfNodePtr &node, irpb::NodeProto *node_proto);
  std::string GetOpNodeInputId(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                               const std::map<AnfNodePtr, size_t> &apply_map,
                               std::map<AnfNodePtr, size_t> *const_map_ptr);
  void SetValueToProto(const ValuePtr &attr_value, irpb::ValueProto *value_proto);
  void SetScalarToProto(const ScalarPtr &val, irpb::ValueProto *value_proto);
  void SetSequenceToProto(const ValueSequeuePtr &val, irpb::ValueProto *value_proto);
  void SetDictionaryToProto(const ValueDictionaryPtr &val, irpb::ValueProto *value_proto);
  void SetNodeOutputType(const AnfNodePtr &node, irpb::TypeProto *type_proto);
  void SetNodeOutputType(const TypePtr &node, const BaseShapePtr &shape, irpb::TypeProto *type_proto);

  void ExportFuncGraph(const FuncGraphPtr &func_graph, irpb::GraphProto *graph_proto);
  void ExportParameters(const FuncGraphPtr &func_graph, irpb::GraphProto *graph_proto);
  void ExportCNodes(const FuncGraphPtr &func_graph, irpb::GraphProto *graph_proto,
                    std::map<AnfNodePtr, size_t> *const_map_ptr);
  void ExportCNode(const FuncGraphPtr &func_graph, const CNodePtr &node, std::map<AnfNodePtr, size_t> *apply_map_ptr,
                   std::map<AnfNodePtr, size_t> *const_map_ptr, irpb::GraphProto *graph_proto);
  void ExportFuncGraphOutput(const FuncGraphPtr &func_graph, const CNodePtr &ret_node,
                             const std::map<AnfNodePtr, size_t> &apply_map, std::map<AnfNodePtr, size_t> *const_map_ptr,
                             irpb::GraphProto *graph_proto);
  void ExportValueNodes(const std::map<AnfNodePtr, size_t> &const_map, irpb::GraphProto *graph_proto);

  static std::string GetConstNodeId(size_t idx) { return std::string("cst") + std::to_string(idx); }

  irpb::ModelProto model_;
};

static irpb::DataType GetNumberDataType(const TypePtr &type) {
  switch (type->type_id()) {
    case kNumberTypeBool:
      return irpb::DT_BOOL;
    case kNumberTypeInt8:
      return irpb::DT_INT8;
    case kNumberTypeInt16:
      return irpb::DT_INT16;
    case kNumberTypeInt32:
      return irpb::DT_INT32;
    case kNumberTypeInt64:
      return irpb::DT_INT64;
    case kNumberTypeUInt8:
      return irpb::DT_UINT8;
    case kNumberTypeUInt16:
      return irpb::DT_UINT16;
    case kNumberTypeUInt32:
      return irpb::DT_UINT32;
    case kNumberTypeUInt64:
      return irpb::DT_UINT64;
    case kNumberTypeFloat16:
      return irpb::DT_FLOAT16;
    case kNumberTypeFloat32:
      return irpb::DT_FLOAT32;
    case kNumberTypeFloat64:
      return irpb::DT_FLOAT64;
    case kNumberTypeInt:
      return irpb::DT_BASE_INT;
    case kNumberTypeUInt:
      return irpb::DT_BASE_UINT;
    case kNumberTypeFloat:
      return irpb::DT_BASE_FLOAT;
    default:
      MS_LOG(EXCEPTION) << "Unexpected type " << type->type_name();
  }
}

void ProtoExporter::SetNodeOutputType(const TypePtr &type, const BaseShapePtr &shape, irpb::TypeProto *type_proto) {
  if (type_proto == nullptr) {
    return;
  }

  if (type == nullptr) {
    type_proto->set_data_type(irpb::DT_UNDEFINED);
  } else if (type->isa<Number>()) {
    type_proto->set_data_type(GetNumberDataType(type));
  } else if (type->isa<TensorType>()) {
    TypePtr elem_type = dyn_cast<TensorType>(type)->element();
    type_proto->mutable_tensor_type()->set_elem_type(GetNumberDataType(elem_type));
    type_proto->set_data_type(irpb::DT_TENSOR);
    if (shape != nullptr && shape->isa<abstract::Shape>()) {
      abstract::ShapePtr shape_info = dyn_cast<abstract::Shape>(shape);
      for (const auto &elem : shape_info->shape()) {
        type_proto->mutable_tensor_type()->mutable_shape()->add_dim()->set_size(elem);
      }
    }
  } else if (type->isa<Tuple>()) {
    TuplePtr tuple_type = dyn_cast<Tuple>(type);
    type_proto->set_data_type(irpb::DT_TUPLE);
    for (const auto &elem_type : tuple_type->elements()) {
      SetNodeOutputType(elem_type, nullptr, type_proto->mutable_sequence_type()->add_elem_types());
    }
  } else if (type->isa<TypeType>()) {
    type_proto->set_data_type(irpb::DT_TYPE);
  } else if (type->isa<List>()) {
    ListPtr list_type = dyn_cast<List>(type);
    type_proto->set_data_type(irpb::DT_LIST);
    for (const auto &elem_type : list_type->elements()) {
      SetNodeOutputType(elem_type, nullptr, type_proto->mutable_sequence_type()->add_elem_types());
    }
  } else if (type->isa<TypeAnything>()) {
    type_proto->set_data_type(irpb::DT_ANYTHING);
  } else if (type->isa<RefKeyType>()) {
    type_proto->set_data_type(irpb::DT_REFKEY);
  } else if (type->isa<RefType>()) {
    type_proto->set_data_type(irpb::DT_REF);
  } else if (type->isa<Function>()) {
    type_proto->set_data_type(irpb::DT_GRAPH);
  } else if (type->isa<TypeNone>()) {
    type_proto->set_data_type(irpb::DT_NONE);
  } else if (type->isa<String>()) {
    type_proto->set_data_type(irpb::DT_STRING);
  } else if (type->isa<SymbolicKeyType>()) {
    // Do Nothing.
  } else {
    MS_LOG(EXCEPTION) << "Unknown type: " << type->type_name();
  }
}

void ProtoExporter::SetNodeOutputType(const AnfNodePtr &node, irpb::TypeProto *type_proto) {
  if (node == nullptr || type_proto == nullptr) {
    return;
  }
  SetNodeOutputType(node->Type(), node->Shape(), type_proto);
}

void ProtoExporter::SetValueToProto(const ValuePtr &val, irpb::ValueProto *value_proto) {
  if (val == nullptr || value_proto == nullptr) {
    return;
  }

  if (val->isa<StringImm>()) {
    const StringImmPtr &value = dyn_cast<StringImm>(val);
    value_proto->set_dtype(irpb::DT_STRING);
    value_proto->set_str_val(value->value());
  } else if (val->isa<Scalar>()) {
    SetScalarToProto(dyn_cast<Scalar>(val), value_proto);
  } else if (val->isa<Bool>()) {
    value_proto->set_dtype(irpb::DT_TYPE);
    value_proto->mutable_type_val()->set_data_type(irpb::DT_BOOL);
  } else if (val->isa<Int>()) {
    value_proto->set_dtype(irpb::DT_TYPE);
    value_proto->mutable_type_val()->set_data_type(irpb::DT_BASE_INT);
  } else if (val->isa<Float>()) {
    value_proto->set_dtype(irpb::DT_TYPE);
    value_proto->mutable_type_val()->set_data_type(irpb::DT_BASE_FLOAT);
  } else if (val->isa<ValueSequeue>()) {
    SetSequenceToProto(dyn_cast<ValueSequeue>(val), value_proto);
  } else if (val->isa<None>()) {
    value_proto->set_dtype(irpb::DT_NONE);
    value_proto->set_str_val("None");
  } else if (val->isa<SymbolicKeyInstance>()) {
    SymbolicKeyInstancePtr sym_inst = dyn_cast<SymbolicKeyInstance>(val);
    ParameterPtr sym_node = dyn_cast<Parameter>(sym_inst->node());
    value_proto->set_dtype(irpb::DT_SYM_INST);
    value_proto->set_str_val(sym_node == nullptr ? std::string("nullptr") : sym_node->ToString());
  } else if (val->isa<ValueDictionary>()) {
    SetDictionaryToProto(dyn_cast<ValueDictionary>(val), value_proto);
  } else if (val->isa<tensor::Tensor>()) {
    tensor::TensorPtr tensor_ptr = dyn_cast<tensor::Tensor>(val);
    value_proto->set_dtype(irpb::DT_TENSOR);
    irpb::TensorProto *tensor_proto = value_proto->mutable_tensor_val();
    tensor_proto->set_data_type(GetNumberDataType(tensor_ptr->Dtype()));
    for (auto &elem : tensor_ptr->shape()) {
      tensor_proto->add_dims(elem);
    }
  } else if (val->isa<TensorType>()) {
    value_proto->set_dtype(irpb::DT_TYPE);

    irpb::TypeProto *type_proto = value_proto->mutable_type_val();
    type_proto->set_data_type(irpb::DT_TENSOR);
    TypePtr elem_type = dyn_cast<TensorType>(val)->element();
    type_proto->mutable_tensor_type()->set_elem_type(GetNumberDataType(elem_type));
  } else {
    MS_LOG(WARNING) << "Unsupported type " << val->type_name();
  }
}

void ProtoExporter::SetScalarToProto(const ScalarPtr &val, irpb::ValueProto *value_proto) {
  if (val == nullptr || value_proto == nullptr) {
    return;
  }

  if (val->isa<BoolImm>()) {
    const BoolImmPtr &value = dyn_cast<BoolImm>(val);
    value_proto->set_dtype(irpb::DT_BOOL);
    value_proto->set_bool_val(value->value());
  } else if (val->isa<Int8Imm>()) {
    const Int8ImmPtr &value = dyn_cast<Int8Imm>(val);
    value_proto->set_dtype(irpb::DT_INT8);
    value_proto->set_int_val(value->value());
  } else if (val->isa<Int16Imm>()) {
    const Int16ImmPtr &value = dyn_cast<Int16Imm>(val);
    value_proto->set_dtype(irpb::DT_INT16);
    value_proto->set_int_val(value->value());
  } else if (val->isa<Int32Imm>()) {
    const Int32ImmPtr &value = dyn_cast<Int32Imm>(val);
    value_proto->set_dtype(irpb::DT_INT32);
    value_proto->set_int_val(value->value());
  } else if (val->isa<Int64Imm>()) {
    const Int64ImmPtr &value = dyn_cast<Int64Imm>(val);
    value_proto->set_dtype(irpb::DT_INT64);
    value_proto->set_int_val(value->value());
  } else if (val->isa<UInt8Imm>()) {
    const UInt8ImmPtr &value = dyn_cast<UInt8Imm>(val);
    value_proto->set_dtype(irpb::DT_UINT8);
    value_proto->set_uint_val(value->value());
  } else if (val->isa<UInt16Imm>()) {
    const UInt16ImmPtr &value = dyn_cast<UInt16Imm>(val);
    value_proto->set_dtype(irpb::DT_UINT16);
    value_proto->set_uint_val(value->value());
  } else if (val->isa<UInt32Imm>()) {
    const UInt32ImmPtr &value = dyn_cast<UInt32Imm>(val);
    value_proto->set_dtype(irpb::DT_UINT32);
    value_proto->set_uint_val(value->value());
  } else if (val->isa<UInt64Imm>()) {
    const UInt64ImmPtr &value = dyn_cast<UInt64Imm>(val);
    value_proto->set_dtype(irpb::DT_UINT64);
    value_proto->set_uint_val(value->value());
  } else if (val->isa<FP32Imm>()) {
    const FP32ImmPtr &value = dyn_cast<FP32Imm>(val);
    value_proto->set_dtype(irpb::DT_FLOAT32);
    value_proto->set_float_val(value->value());
  } else if (val->isa<FP64Imm>()) {
    const FP64ImmPtr &value = dyn_cast<FP64Imm>(val);
    value_proto->set_dtype(irpb::DT_FLOAT64);
    value_proto->set_double_val(value->value());
  } else {
    MS_LOG(EXCEPTION) << "Unknown scalar type " << val->ToString();
  }
}

void ProtoExporter::SetSequenceToProto(const ValueSequeuePtr &val, irpb::ValueProto *value_proto) {
  if (val == nullptr || value_proto == nullptr) {
    return;
  }

  if (val->isa<ValueTuple>()) {
    const ValueTuplePtr &value = dyn_cast<ValueTuple>(val);
    value_proto->set_dtype(irpb::DT_TUPLE);
    for (const auto &item : value->value()) {
      SetValueToProto(item, value_proto->add_values());
    }
  } else if (val->isa<ValueList>()) {
    const ValueListPtr &value = dyn_cast<ValueList>(val);
    value_proto->set_dtype(irpb::DT_LIST);
    for (const auto &item : value->value()) {
      SetValueToProto(item, value_proto->add_values());
    }
  }
}

void ProtoExporter::SetDictionaryToProto(const ValueDictionaryPtr &val, irpb::ValueProto *value_proto) {
  if (val == nullptr || value_proto == nullptr) {
    return;
  }

  value_proto->set_dtype(irpb::DT_DICT);
  for (const auto &item : val->value()) {
    irpb::NamedValueProto *named_val = value_proto->add_dict_val();
    named_val->set_key(item.first);
    SetValueToProto(item.second, named_val->mutable_value());
  }
}

void ProtoExporter::GetOpNodeTypeAndAttrs(const FuncGraphPtr &, const AnfNodePtr &node, irpb::NodeProto *node_proto) {
  if (node == nullptr || node_proto == nullptr) {
    return;
  }

  if (node->isa<CNode>() || node->isa<Parameter>() || IsValueNode<FuncGraph>(node)) {
    MS_LOG(EXCEPTION) << "Op node can not be CNode, Parameter or ValueNode Graph. But got " << node->ToString();
  }

  if (!IsValueNode<Primitive>(node)) {
    MS_LOG(EXCEPTION) << "Op node is not primitive: " << node->ToString();
  }

  const PrimitivePtr &prim = GetValueNode<PrimitivePtr>(node);
  node_proto->set_op_type(prim->name());
  for (const auto &attr : prim->attrs()) {
    irpb::AttributeProto *attr_proto = node_proto->add_attribute();
    attr_proto->set_name(attr.first);
    SetValueToProto(attr.second, attr_proto->mutable_value());
  }
  node_proto->set_scope(node->scope()->name());
}

std::string ProtoExporter::GetOpNodeInputId(const FuncGraphPtr &, const AnfNodePtr &node,
                                            const std::map<AnfNodePtr, size_t> &apply_map,
                                            std::map<AnfNodePtr, size_t> *const_map_ptr) {
  if (node == nullptr || const_map_ptr == nullptr) {
    return "";
  }

  if (node->isa<CNode>()) {
    auto iter = apply_map.find(node);
    if (iter == apply_map.end()) {
      MS_LOG(EXCEPTION) << "Can not find node '" << node->ToString() << "' in apply_map";
    }
    return std::to_string(iter->second);
  }

  if (node->isa<Parameter>()) {
    return node->ToString();
  }

  if (node->isa<ValueNode>()) {
    auto iter = const_map_ptr->find(node);
    if (iter == const_map_ptr->end()) {
      // Start index number from 1
      auto const_idx = const_map_ptr->size() + 1;
      (*const_map_ptr)[node] = const_idx;
    }
    return GetConstNodeId((*const_map_ptr)[node]);
  }

  MS_LOG(EXCEPTION) << "Unknown node type. node is '" << node->ToString() << "'";
}

std::string ProtoExporter::GetFuncGraphProtoString(const FuncGraphPtr &func_graph) {
  if (func_graph == nullptr) {
    return "";
  }

  InitModelInfo();
  irpb::GraphProto *graph_proto = model_.mutable_graph();
  ExportFuncGraph(func_graph, graph_proto);
  return model_.SerializeAsString();
}

void ProtoExporter::ExportFuncGraph(const FuncGraphPtr &func_graph, irpb::GraphProto *graph_proto) {
  if (func_graph == nullptr || graph_proto == nullptr) {
    return;
  }

  // map for store ValueNodes of this graph
  std::map<AnfNodePtr, size_t> const_map;

  // set graph name
  graph_proto->set_name(func_graph->ToString());

  ExportParameters(func_graph, graph_proto);

  ExportCNodes(func_graph, graph_proto, &const_map);

  ExportValueNodes(const_map, graph_proto);
}

void ProtoExporter::ExportParameters(const FuncGraphPtr &func_graph, irpb::GraphProto *graph_proto) {
  if (func_graph == nullptr || graph_proto == nullptr) {
    return;
  }

  std::vector<AnfNodePtr> parameters = func_graph->parameters();
  for (auto &param : parameters) {
    irpb::ParameterProto *param_proto = graph_proto->add_parameters();
    param_proto->set_name(param->ToString());

    SetNodeOutputType(param, param_proto->mutable_type());

    const ParameterPtr param_ptr = dyn_cast<Parameter>(param);
    if (param_ptr == nullptr) {
      MS_LOG(EXCEPTION) << "Parameter '" << param->ToString() << "' could not cast to parameter.";
    }
  }
}

void ProtoExporter::ExportCNodes(const FuncGraphPtr &func_graph, irpb::GraphProto *graph_proto,
                                 std::map<AnfNodePtr, size_t> *const_map_ptr) {
  if (func_graph == nullptr || graph_proto == nullptr || const_map_ptr == nullptr) {
    return;
  }
  // topo sort nodes
  std::vector<AnfNodePtr> nodes = TopoSort(func_graph->get_return(), SuccIncoming, AlwaysInclude);
  std::map<AnfNodePtr, size_t> apply_map;
  for (const AnfNodePtr &node : nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (cnode != func_graph->get_return()) {
      ExportCNode(func_graph, cnode, &apply_map, const_map_ptr, graph_proto);
    } else {
      ExportFuncGraphOutput(func_graph, cnode, apply_map, const_map_ptr, graph_proto);
    }
  }
}

void ProtoExporter::ExportCNode(const FuncGraphPtr &func_graph, const CNodePtr &node,
                                std::map<AnfNodePtr, size_t> *apply_map_ptr,
                                std::map<AnfNodePtr, size_t> *const_map_ptr, irpb::GraphProto *graph_proto) {
  if (func_graph == nullptr || node == nullptr || apply_map_ptr == nullptr || const_map_ptr == nullptr ||
      graph_proto == nullptr) {
    return;
  }

  auto apply_idx = apply_map_ptr->size() + 1;
  (*apply_map_ptr)[node] = apply_idx;

  auto &inputs = node->inputs();
  if (inputs.size() < 1) {
    MS_LOG(EXCEPTION) << "Inputs of apply node is empty";
  }
  AnfNodePtr op = inputs[0];
  irpb::NodeProto *node_proto = graph_proto->add_node();

  // CNode/ConstGraph/Const/Parameter
  if (op->isa<CNode>() || IsValueNode<FuncGraph>(op) || op->isa<Parameter>()) {
    MS_LOG(WARNING) << "Operator must be a primitive";
  } else {
    GetOpNodeTypeAndAttrs(func_graph, op, node_proto);
    node_proto->set_name(std::to_string(apply_idx));
    node_proto->set_scope(node->scope()->name());

    // process OP inputs
    for (size_t i = 1; i < inputs.size(); ++i) {
      irpb::InputProto *input_proto = node_proto->add_input();
      input_proto->set_type(irpb::InputProto_EdgeType_DATA_EDGE);
      std::string id = GetOpNodeInputId(func_graph, inputs[i], *apply_map_ptr, const_map_ptr);
      input_proto->set_name(id);
    }

    // set node output type
    SetNodeOutputType(node, node_proto->mutable_output_type());
  }
}

void ProtoExporter::ExportFuncGraphOutput(const FuncGraphPtr &func_graph, const CNodePtr &ret_node,
                                          const std::map<AnfNodePtr, size_t> &apply_map,
                                          std::map<AnfNodePtr, size_t> *const_map_ptr, irpb::GraphProto *graph_proto) {
  if (ret_node == nullptr || !ret_node->isa<CNode>()) {
    MS_LOG(EXCEPTION) << "Graph return node is illegal";
  }
  AnfNodePtr arg = ret_node->input(1);
  if (graph_proto == nullptr) {
    MS_LOG(EXCEPTION) << "graph_proto is nullptr";
  }
  irpb::OutputProto *output_proto = graph_proto->add_outputs();
  if (output_proto == nullptr) {
    MS_LOG(EXCEPTION) << "output_proto is nullptr";
  }
  std::string id = GetOpNodeInputId(func_graph, arg, apply_map, const_map_ptr);
  output_proto->set_name(id);
  SetNodeOutputType(arg, output_proto->mutable_type());
}

static bool CompareValue(const std::pair<AnfNodePtr, size_t> &x, const std::pair<AnfNodePtr, size_t> &y) {
  return x.second < y.second;
}

void ProtoExporter::ExportValueNodes(const std::map<AnfNodePtr, size_t> &const_map, irpb::GraphProto *graph_proto) {
  std::vector<std::pair<AnfNodePtr, size_t>> nodes;
  (void)std::transform(const_map.cbegin(), const_map.cend(), std::back_inserter(nodes),
                       [](const std::pair<AnfNodePtr, size_t> &item) { return item; });

  sort(nodes.begin(), nodes.end(), CompareValue);

  for (auto &item : nodes) {
    if (graph_proto == nullptr) {
      MS_LOG(EXCEPTION) << "graph_proto is nullptr";
    }
    irpb::NamedValueProto *named_value = graph_proto->add_const_vals();
    MS_EXCEPTION_IF_NULL(named_value);
    named_value->set_key(GetConstNodeId(item.second));
    SetValueToProto(GetValueNode(item.first), named_value->mutable_value());
  }
}

void ProtoExporter::InitModelInfo() { model_.set_ir_version(irpb::IR_VERSION); }

std::string GetFuncGraphProtoString(const FuncGraphPtr &func_graph) {
  ProtoExporter exporter;
  return exporter.GetFuncGraphProtoString(func_graph);
}
}  // namespace mindspore

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
#include "include/common/debug/dump_proto.h"

#include <algorithm>
#include <fstream>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "google/protobuf/util/json_util.h"

#include "proto/anf_ir.pb.h"
#include "proto/mind_ir.pb.h"
#include "ir/graph_utils.h"
#include "utils/ms_context.h"
#include "utils/symbolic.h"
#include "include/common/utils/utils.h"
#include "include/common/debug/anf_dump_utils.h"
#include "utils/anf_utils.h"
#include "frontend/parallel/ops_info/ops_utils.h"  // todo: use constant string now
#include "mindspore/core/utils/file_utils.h"

namespace mindspore {
class ProtoExporter {
 public:
  ProtoExporter() {}
  ~ProtoExporter() {}

  std::string GetFuncGraphProtoString(const FuncGraphPtr &func_graph);
  void ExportFuncGraph(const FuncGraphPtr &func_graph, irpb::GraphProto *graph_proto);

 private:
  void InitModelInfo();
  void GetOpNodeTypeAndAttrs(const FuncGraphPtr & /* func_graph */, const CNodePtr &cnode, irpb::NodeProto *node_proto);
  std::string GetOpNodeInputId(const FuncGraphPtr & /* func_graph */, const AnfNodePtr &node,
                               const std::map<AnfNodePtr, size_t> &apply_map,
                               std::map<AnfNodePtr, size_t> *const_map_ptr) const;
  void SetValueToProtoBasicTypes(const ValuePtr &val, irpb::ValueProto *const value_proto) const;
  void SetValueToProto(const ValuePtr &val, irpb::ValueProto *value_proto);
  void SetScalarToProto(const ScalarPtr &val, irpb::ValueProto *value_proto) const;
  void SetSequenceToProto(const ValueSequencePtr &val, irpb::ValueProto *value_proto);
  void SetDictionaryToProto(const ValueDictionaryPtr &val, irpb::ValueProto *value_proto);
  void SetNodeOutputType(const AnfNodePtr &node, irpb::TypeProto *type_proto);
  void SetNodeOutputType(const TypePtr &type, const BaseShapePtr &shape, irpb::TypeProto *type_proto);

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

static std::map<TypeId, irpb::DataType> number_data_type_map = {{kNumberTypeBool, irpb::DT_BOOL},
                                                                {kNumberTypeInt8, irpb::DT_INT8},
                                                                {kNumberTypeInt16, irpb::DT_INT16},
                                                                {kNumberTypeInt32, irpb::DT_INT32},
                                                                {kNumberTypeInt64, irpb::DT_INT64},
                                                                {kNumberTypeUInt8, irpb::DT_UINT8},
                                                                {kNumberTypeUInt16, irpb::DT_UINT16},
                                                                {kNumberTypeUInt32, irpb::DT_UINT32},
                                                                {kNumberTypeUInt64, irpb::DT_UINT64},
                                                                {kNumberTypeFloat16, irpb::DT_FLOAT16},
                                                                {kNumberTypeFloat32, irpb::DT_FLOAT32},
                                                                {kNumberTypeFloat64, irpb::DT_FLOAT64},
                                                                {kNumberTypeInt, irpb::DT_BASE_INT},
                                                                {kNumberTypeUInt, irpb::DT_BASE_UINT},
                                                                {kNumberTypeFloat, irpb::DT_BASE_FLOAT},
                                                                {kNumberTypeComplex64, irpb::DT_COMPLEX64},
                                                                {kNumberTypeComplex128, irpb::DT_COMPLEX128},
                                                                {kObjectTypeString, irpb::DT_STRING},
                                                                {kObjectTypeTuple, irpb::DT_TUPLE}};

static irpb::DataType GetNumberDataType(const TypePtr &type) {
  auto iter = number_data_type_map.find(type->type_id());
  if (iter != number_data_type_map.end()) {
    return (*iter).second;
  } else {
    MS_LOG(EXCEPTION) << "Unexpected type " << type->type_name();
  }
}

static inline bool IsKindOfTensorType(const TypePtr &type) {
  MS_EXCEPTION_IF_NULL(type);
  return type->isa<TensorType>() || type->isa<RowTensorType>() || type->isa<CSRTensorType>() ||
         type->isa<COOTensorType>() || type->isa<MapTensorType>();
}

void CheckIfValidType(const TypePtr &type) {
  MS_EXCEPTION_IF_NULL(type);
  if (type->isa<Problem>()) {
    MS_LOG(WARNING) << "The type: " << type->type_name();
    return;
  }
  if (!(type->isa<Number>() || IsKindOfTensorType(type) || type->isa<Tuple>() || type->isa<TypeType>() ||
        type->isa<List>() || type->isa<TypeAny>() || type->isa<RefKeyType>() || type->isa<RefType>() ||
        type->isa<Function>() || type->isa<TypeNone>() || type->isa<String>() || type->isa<UndeterminedType>() ||
        type->isa<SymbolicKeyType>() || type->isa<MonadType>() || type->isa<Dictionary>())) {
    MS_LOG(EXCEPTION) << "Unknown type: " << type->type_name();
  }
}

void SetTensorType(const TypePtr &type, const BaseShapePtr &shape, irpb::TypeProto *const type_proto) {
  TypePtr elem_type = dyn_cast<TensorType>(type)->element();
  type_proto->mutable_tensor_type()->set_elem_type(GetNumberDataType(elem_type));
  type_proto->set_data_type(irpb::DT_TENSOR);
  if (shape != nullptr && shape->isa<abstract::Shape>()) {
    abstract::ShapePtr shape_info = dyn_cast<abstract::Shape>(shape);
    for (const auto &elem : shape_info->shape()) {
      type_proto->mutable_tensor_type()->mutable_shape()->add_dim()->set_size(elem);
    }
  }
}

void ProtoExporter::SetNodeOutputType(const TypePtr &type, const BaseShapePtr &shape, irpb::TypeProto *type_proto) {
  if (type_proto == nullptr) {
    return;
  }
  if (type == nullptr) {
    type_proto->set_data_type(irpb::DT_UNDEFINED);
    return;
  }
  if (type->isa<External>()) {
    return;
  }
  CheckIfValidType(type);
  if (type->isa<Number>()) {
    type_proto->set_data_type(GetNumberDataType(type));
  } else if (type->isa<TensorType>()) {
    SetTensorType(type, shape, type_proto);
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
  } else if (type->isa<TypeAny>()) {
    type_proto->set_data_type(irpb::DT_ANY);
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
  }
}

void ProtoExporter::SetNodeOutputType(const AnfNodePtr &node, irpb::TypeProto *type_proto) {
  if (node == nullptr || type_proto == nullptr) {
    return;
  }
  SetNodeOutputType(node->Type(), node->Shape(), type_proto);
}

void ProtoExporter::SetValueToProtoBasicTypes(const ValuePtr &val, irpb::ValueProto *const value_proto) const {
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
  } else if (val->isa<UInt>()) {
    value_proto->set_dtype(irpb::DT_TYPE);
    value_proto->mutable_type_val()->set_data_type(irpb::DT_BASE_UINT);
  } else if (val->isa<Float>()) {
    value_proto->set_dtype(irpb::DT_TYPE);
    value_proto->mutable_type_val()->set_data_type(irpb::DT_BASE_FLOAT);
  }
}

void ProtoExporter::SetValueToProto(const ValuePtr &val, irpb::ValueProto *value_proto) {
  if (val == nullptr || value_proto == nullptr) {
    return;
  }

  SetValueToProtoBasicTypes(val, value_proto);

  if (val->isa<ValueSequence>()) {
    SetSequenceToProto(dyn_cast<ValueSequence>(val), value_proto);
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
  } else if (val->isa<Monad>() || val->isa<MonadType>()) {
    value_proto->set_str_val(val->ToString());
  } else if (val->isa<Complex>()) {
    value_proto->set_dtype(irpb::DT_TYPE);
    value_proto->mutable_type_val()->set_data_type(irpb::DT_BASE_COMPLEX);
  } else {
    MS_LOG(DEBUG) << "Unsupported type " << val->type_name();
  }
}

void ProtoExporter::SetScalarToProto(const ScalarPtr &val, irpb::ValueProto *value_proto) const {
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

void ProtoExporter::SetSequenceToProto(const ValueSequencePtr &val, irpb::ValueProto *value_proto) {
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
    MS_EXCEPTION_IF_NULL(item.first);
    if (!item.first->isa<StringImm>()) {
      MS_LOG(EXCEPTION) << "The key of NamedValueProto should be string type, but got " << item.first->ToString();
    }
    named_val->set_key(GetValue<std::string>(item.first));
    SetValueToProto(item.second, named_val->mutable_value());
  }
}

void ProtoExporter::GetOpNodeTypeAndAttrs(const FuncGraphPtr & /* func_graph */, const CNodePtr &cnode,
                                          irpb::NodeProto *node_proto) {
  const auto &inputs = cnode->inputs();
  AnfNodePtr op_node = inputs[0];

  if (op_node == nullptr || node_proto == nullptr) {
    return;
  }

  if (op_node->isa<CNode>() || op_node->isa<Parameter>() || IsValueNode<FuncGraph>(op_node)) {
    MS_LOG(EXCEPTION) << "Op node can not be CNode, Parameter or ValueNode Graph. But got " << op_node->ToString();
  }

  if (!IsValueNode<Primitive>(op_node)) {
    MS_LOG(EXCEPTION) << "Op node is not primitive: " << op_node->ToString();
  }

  const PrimitivePtr &prim = GetValueNode<PrimitivePtr>(op_node);
  node_proto->set_op_type(prim->name());
  for (const auto &attr : prim->attrs()) {
    irpb::AttributeProto *attr_proto = node_proto->add_attribute();
    attr_proto->set_name(attr.first);
    SetValueToProto(attr.second, attr_proto->mutable_value());
  }

  // Only CNode save the operator strategy
  auto strategy_value = AnfDumpHandler::InStrategyValue(cnode);
  if (strategy_value != nullptr) {
    irpb::AttributeProto *attr_proto = node_proto->add_attribute();
    attr_proto->set_name(mindspore::parallel::IN_STRATEGY);
    SetValueToProto(strategy_value, attr_proto->mutable_value());
  }

  node_proto->set_scope(op_node->scope()->name());
}

std::string ProtoExporter::GetOpNodeInputId(const FuncGraphPtr & /* func_graph */, const AnfNodePtr &node,
                                            const std::map<AnfNodePtr, size_t> &apply_map,
                                            std::map<AnfNodePtr, size_t> *const_map_ptr) const {
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

  if (AnfUtils::IsCustomActorNode(node)) {
    return AnfUtils::GetCustomActorName(node);
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
    MS_LOG(DEBUG) << "Operator must be a primitive";
  } else {
    GetOpNodeTypeAndAttrs(func_graph, node, node_proto);
    node_proto->set_name(std::to_string(apply_idx));
    node_proto->set_scope(node->scope()->name());
    node_proto->set_full_name(GetKernelNodeName(node));

    // process OP inputs
    for (size_t i = 1; i < inputs.size(); ++i) {
      irpb::InputProto *input_proto = node_proto->add_input();
      input_proto->set_type(irpb::InputProto_EdgeType_DATA_EDGE);
      std::string id = GetOpNodeInputId(func_graph, inputs[i], *apply_map_ptr, const_map_ptr);
      input_proto->set_name(id);
    }

    // set node output type
    SetNodeOutputType(node, node_proto->mutable_output_type());

    if (IsValueNode<Primitive>(op)) {
      PrimitivePtr primitive = GetValueNode<PrimitivePtr>(op);
      if (!primitive->instance_name().empty()) {
        node_proto->set_instance_name(primitive->instance_name());
      }
    }
  }
}

void ProtoExporter::ExportFuncGraphOutput(const FuncGraphPtr &func_graph, const CNodePtr &ret_node,
                                          const std::map<AnfNodePtr, size_t> &apply_map,
                                          std::map<AnfNodePtr, size_t> *const_map_ptr, irpb::GraphProto *graph_proto) {
  if (ret_node == nullptr || !ret_node->isa<CNode>()) {
    MS_LOG(EXCEPTION) << "Graph return node is illegal";
  }
  // ret node has two input 1 ret op + 1 value
  const size_t ret_input_size = 2;
  if (ret_node->inputs().size() != ret_input_size) {
    return;
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

void ProtoExporter::InitModelInfo() { model_.set_ir_version(static_cast<int64_t>(irpb::IR_VERSION)); }

std::string GetFuncGraphProtoString(const FuncGraphPtr &func_graph) {
  ProtoExporter exporter;
  return exporter.GetFuncGraphProtoString(func_graph);
}

std::string GetFuncGraphProtoJsonString(const FuncGraphPtr &func_graph) {
  ProtoExporter exporter;
  irpb::GraphProto graph_proto = irpb::GraphProto();
  exporter.ExportFuncGraph(func_graph, &graph_proto);
  std::string graph_proto_str;
  (void)google::protobuf::util::MessageToJsonString(graph_proto, &graph_proto_str);
  return graph_proto_str;
}

#ifdef ENABLE_DUMP_IR
void DumpIRProto(const FuncGraphPtr &func_graph, const std::string &suffix) {
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "Func graph is nullptr";
    return;
  }
  std::string file_path = GetSaveGraphsPathName("ms_output_" + suffix + ".pb");
  auto realpath = Common::CreatePrefixPath(file_path);
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Get real path failed, path=" << file_path;
    return;
  }

  ChangeFileMode(realpath.value(), S_IWUSR);
  // write to pb file
  std::ofstream ofs(file_path);
  if (!ofs.is_open()) {
    MS_LOG(ERROR) << "Open file '" << file_path << "' failed!" << ErrnoToString(errno);
    return;
  }
  ofs << GetFuncGraphProtoString(func_graph);
  ofs.close();
  // set file mode to read only by user
  ChangeFileMode(file_path, S_IRUSR);
}
#else
void DumpIRProto(const FuncGraphPtr &, const std::string &) {
  static bool already_printed = false;
  if (already_printed) {
    return;
  }
  already_printed = true;
  MS_LOG(WARNING) << "The functionality of dumping function graph IR in protobuf format is disabled, "
                  << "please recompile source to enable it. See help of building script.";
}
#endif
}  // namespace mindspore

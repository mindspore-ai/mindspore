/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "debug/debugger/proto_exporter.h"

#include <fstream>
#include <map>
#include <memory>
#include <utility>
#include <algorithm>

#include "utils/hash_map.h"
#include "utils/hash_set.h"
#include "include/common/debug/anf_dump_utils.h"
#include "debug/data_dump/dump_utils.h"
#include "include/common/debug/common.h"
#include "debug/debugger/debugger.h"
#include "debug/data_dump/dump_json_parser.h"
#include "proto/debug_graph.pb.h"
#include "ir/graph_utils.h"
#include "utils/symbolic.h"
#include "utils/trace_base.h"
#include "debug/data_dump/e2e_dump.h"
#include "mindspore/core/utils/file_utils.h"
#include "utils/anf_utils.h"

namespace mindspore {

using TypeInfoToProtoTypeMap = std::vector<std::pair<uint32_t, debugger::DataType>>;

void SetOutputType(const TypePtr &type, const BaseShapePtr &shape, debugger::TypeProto *type_proto);

void CheckIfValidType(const TypePtr &type, debugger::TypeProto *const type_proto) {
  if (!(type->isa<Number>() || type->isa<TensorType>() || type->isa<Tuple>() || type->isa<TypeType>() ||
        type->isa<List>() || type->isa<TypeAnything>() || type->isa<RefKeyType>() || type->isa<RefType>() ||
        type->isa<Function>() || type->isa<TypeNone>() || type->isa<String>() || type->isa<SymbolicKeyType>() ||
        type->isa<MapTensorType>() || type->isa<UMonadType>() || type->isa<IOMonadType>())) {
    MS_LOG(EXCEPTION) << "Unknown type: " << type->type_name();
  }
  if (type->isa<Number>()) {
    type_proto->set_data_type(GetDebuggerNumberDataType(type));
  }
}

void SetTensorTypeProto(const TypePtr &type, const BaseShapePtr &shape, debugger::TypeProto *type_proto) {
  TypePtr elem_type = dyn_cast<TensorType>(type)->element();
  type_proto->mutable_tensor_type()->set_elem_type(GetDebuggerNumberDataType(elem_type));
  if (shape != nullptr && shape->isa<abstract::Shape>()) {
    abstract::ShapePtr shape_info = dyn_cast<abstract::Shape>(shape);
    for (const auto &elem : shape_info->shape()) {
      type_proto->mutable_tensor_type()->mutable_shape()->add_dim()->set_size(elem);
    }
  }
}

void SetTupleTypeProto(const TypePtr &type, debugger::TypeProto *type_proto) {
  TuplePtr tuple_type = dyn_cast<Tuple>(type);
  for (const auto &elem_type : tuple_type->elements()) {
    SetOutputType(elem_type, nullptr, type_proto->mutable_sequence_type()->add_elem_types());
  }
}

void SetListTypeProto(const TypePtr &type, debugger::TypeProto *type_proto) {
  ListPtr list_type = dyn_cast<List>(type);
  for (const auto &elem_type : list_type->elements()) {
    SetOutputType(elem_type, nullptr, type_proto->mutable_sequence_type()->add_elem_types());
  }
}

static TypeInfoToProtoTypeMap type_info_to_proto_type = {
  {TensorType::kTypeId, debugger::DT_TENSOR},     {Tuple::kTypeId, debugger::DT_TUPLE},
  {TypeType::kTypeId, debugger::DT_TYPE},         {List::kTypeId, debugger::DT_LIST},
  {TypeAnything::kTypeId, debugger::DT_ANYTHING}, {RefKeyType::kTypeId, debugger::DT_REFKEY},
  {RefType::kTypeId, debugger::DT_REF},           {Function::kTypeId, debugger::DT_GRAPH},
  {TypeNone::kTypeId, debugger::DT_NONE},         {String::kTypeId, debugger::DT_STRING},
  {UMonadType::kTypeId, debugger::DT_UMONAD},     {IOMonadType::kTypeId, debugger::DT_IOMONAD}};

void SetOutputType(const TypePtr &type, const BaseShapePtr &shape, debugger::TypeProto *type_proto) {
  if (type_proto == nullptr) {
    return;
  }
  if (type == nullptr) {
    type_proto->set_data_type(debugger::DT_UNDEFINED);
    return;
  }
  CheckIfValidType(type, type_proto);
  for (auto &it : type_info_to_proto_type) {
    if (type->IsFromTypeId(it.first)) {
      type_proto->set_data_type(it.second);
      break;
    }
  }
  if (type->isa<TensorType>()) {
    SetTensorTypeProto(type, shape, type_proto);
    return;
  }
  if (type->isa<Tuple>()) {
    SetTupleTypeProto(type, type_proto);
    return;
  }
  if (type->isa<List>()) {
    SetListTypeProto(type, type_proto);
  }
}

void DebuggerProtoExporter::SetNodeOutputType(const AnfNodePtr &node, debugger::TypeProto *type_proto) const {
  if (node == nullptr || type_proto == nullptr) {
    return;
  }
  SetOutputType(node->Type(), node->Shape(), type_proto);
}

void DebuggerProtoExporter::SetValueToProto(const ValuePtr &val, debugger::ValueProto *value_proto) const {
  if (val == nullptr || value_proto == nullptr) {
    return;
  }

  if (val->isa<StringImm>()) {
    const StringImmPtr &value = dyn_cast<StringImm>(val);
    value_proto->set_dtype(debugger::DT_STRING);
    value_proto->set_str_val(value->value());
  } else if (val->isa<Scalar>()) {
    SetScalarToProto(dyn_cast<Scalar>(val), value_proto);
  } else if (val->isa<Bool>()) {
    value_proto->set_dtype(debugger::DT_TYPE);
    value_proto->mutable_type_val()->set_data_type(debugger::DT_BOOL);
  } else if (val->isa<Int>()) {
    value_proto->set_dtype(debugger::DT_TYPE);
    value_proto->mutable_type_val()->set_data_type(debugger::DT_BASE_INT);
  } else if (val->isa<Float>()) {
    value_proto->set_dtype(debugger::DT_TYPE);
    value_proto->mutable_type_val()->set_data_type(debugger::DT_BASE_FLOAT);
  } else if (val->isa<ValueSequence>()) {
    SetSequenceToProto(dyn_cast<ValueSequence>(val), value_proto);
  } else if (val->isa<None>()) {
    value_proto->set_dtype(debugger::DT_NONE);
    value_proto->set_str_val("None");
  } else if (val->isa<SymbolicKeyInstance>()) {
    SymbolicKeyInstancePtr sym_inst = dyn_cast<SymbolicKeyInstance>(val);
    ParameterPtr sym_node = dyn_cast<Parameter>(sym_inst->node());
    value_proto->set_dtype(debugger::DT_SYM_INST);
    value_proto->set_str_val(sym_node == nullptr ? std::string("nullptr") : sym_node->ToString());
  } else if (val->isa<ValueDictionary>()) {
    SetDictionaryToProto(dyn_cast<ValueDictionary>(val), value_proto);
  } else if (val->isa<tensor::Tensor>()) {
    tensor::TensorPtr tensor_ptr = dyn_cast<tensor::Tensor>(val);
    value_proto->set_dtype(debugger::DT_TENSOR);
    debugger::TensorProto *tensor_proto = value_proto->mutable_tensor_val();
    tensor_proto->set_data_type(GetDebuggerNumberDataType(tensor_ptr->Dtype()));
    for (auto &elem : tensor_ptr->shape()) {
      tensor_proto->add_dims(elem);
    }
    tensor_proto->set_tensor_content(tensor_ptr->data_c(), tensor_ptr->data().nbytes());
  } else if (val->isa<TensorType>()) {
    value_proto->set_dtype(debugger::DT_TYPE);

    debugger::TypeProto *type_proto = value_proto->mutable_type_val();
    type_proto->set_data_type(debugger::DT_TENSOR);
    TypePtr elem_type = dyn_cast<TensorType>(val)->element();
    type_proto->mutable_tensor_type()->set_elem_type(GetDebuggerNumberDataType(elem_type));
  } else {
    MS_LOG(INFO) << "Unsupported type " << val->type_name();
  }
}

void DebuggerProtoExporter::SetScalarToProto(const ScalarPtr &val, debugger::ValueProto *value_proto) const {
  if (val == nullptr || value_proto == nullptr) {
    return;
  }

  if (val->isa<BoolImm>()) {
    const BoolImmPtr &value = dyn_cast<BoolImm>(val);
    value_proto->set_dtype(debugger::DT_BOOL);
    value_proto->set_bool_val(value->value());
  } else if (val->isa<Int8Imm>()) {
    const Int8ImmPtr &value = dyn_cast<Int8Imm>(val);
    value_proto->set_dtype(debugger::DT_INT8);
    value_proto->set_int_val(value->value());
  } else if (val->isa<Int16Imm>()) {
    const Int16ImmPtr &value = dyn_cast<Int16Imm>(val);
    value_proto->set_dtype(debugger::DT_INT16);
    value_proto->set_int_val(value->value());
  } else if (val->isa<Int32Imm>()) {
    const Int32ImmPtr &value = dyn_cast<Int32Imm>(val);
    value_proto->set_dtype(debugger::DT_INT32);
    value_proto->set_int_val(value->value());
  } else if (val->isa<Int64Imm>()) {
    const Int64ImmPtr &value = dyn_cast<Int64Imm>(val);
    value_proto->set_dtype(debugger::DT_INT64);
    value_proto->set_int_val(value->value());
  } else if (val->isa<UInt8Imm>()) {
    const UInt8ImmPtr &value = dyn_cast<UInt8Imm>(val);
    value_proto->set_dtype(debugger::DT_UINT8);
    value_proto->set_uint_val(value->value());
  } else if (val->isa<UInt16Imm>()) {
    const UInt16ImmPtr &value = dyn_cast<UInt16Imm>(val);
    value_proto->set_dtype(debugger::DT_UINT16);
    value_proto->set_uint_val(value->value());
  } else if (val->isa<UInt32Imm>()) {
    const UInt32ImmPtr &value = dyn_cast<UInt32Imm>(val);
    value_proto->set_dtype(debugger::DT_UINT32);
    value_proto->set_uint_val(value->value());
  } else if (val->isa<UInt64Imm>()) {
    const UInt64ImmPtr &value = dyn_cast<UInt64Imm>(val);
    value_proto->set_dtype(debugger::DT_UINT64);
    value_proto->set_uint_val(value->value());
  } else if (val->isa<FP32Imm>()) {
    const FP32ImmPtr &value = dyn_cast<FP32Imm>(val);
    value_proto->set_dtype(debugger::DT_FLOAT32);
    value_proto->set_float_val(value->value());
  } else if (val->isa<FP64Imm>()) {
    const FP64ImmPtr &value = dyn_cast<FP64Imm>(val);
    value_proto->set_dtype(debugger::DT_FLOAT64);
    value_proto->set_double_val(value->value());
  } else {
    MS_LOG(EXCEPTION) << "Unknown scalar type " << val->ToString();
  }
}

void DebuggerProtoExporter::SetSequenceToProto(const ValueSequencePtr &val, debugger::ValueProto *value_proto) const {
  if (val == nullptr || value_proto == nullptr) {
    return;
  }

  if (val->isa<ValueTuple>()) {
    const ValueTuplePtr &value = dyn_cast<ValueTuple>(val);
    value_proto->set_dtype(debugger::DT_TUPLE);
    for (const auto &item : value->value()) {
      SetValueToProto(item, value_proto->add_values());
    }
  } else if (val->isa<ValueList>()) {
    const ValueListPtr &value = dyn_cast<ValueList>(val);
    value_proto->set_dtype(debugger::DT_LIST);
    for (const auto &item : value->value()) {
      SetValueToProto(item, value_proto->add_values());
    }
  }
}

void DebuggerProtoExporter::SetDictionaryToProto(const ValueDictionaryPtr &val,
                                                 debugger::ValueProto *value_proto) const {
  if (val == nullptr || value_proto == nullptr) {
    return;
  }

  value_proto->set_dtype(debugger::DT_DICT);
  for (const auto &item : val->value()) {
    debugger::NamedValueProto *named_val = value_proto->add_dict_val();
    if (!item.first->isa<StringImm>()) {
      MS_LOG(EXCEPTION) << "The key of NamedValueProto should be string type, but got " << item.first->ToString();
    }
    named_val->set_key(GetValue<std::string>(item.first));
    SetValueToProto(item.second, named_val->mutable_value());
  }
}

void DebuggerProtoExporter::GetOpNodeTypeAndAttrs(const FuncGraphPtr &, const AnfNodePtr &node,
                                                  debugger::NodeProto *node_proto) const {
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
    debugger::AttributeProto *attr_proto = node_proto->add_attribute();
    attr_proto->set_name(attr.first);
    SetValueToProto(attr.second, attr_proto->mutable_value());
  }
  node_proto->set_scope(node->scope()->name());
}

std::string DebuggerProtoExporter::GetOpNodeInputId(const FuncGraphPtr &, const AnfNodePtr &node,
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

  if (AnfUtils::IsCustomActorNode(node)) {
    return AnfUtils::GetCustomActorName(node);
  }

  if (node->isa<Parameter>()) {
    return node->ToString();
  }

  if (node->isa<ValueNode>()) {
    std::map<AnfNodePtr, size_t>::const_iterator iter = const_map_ptr->find(node);
    if (iter == const_map_ptr->end()) {
      // Start index number from 1
      const auto const_idx = const_map_ptr->size() + 1;
      (*const_map_ptr)[node] = const_idx;
    }
    return GetConstNodeId((*const_map_ptr)[node]);
  }

  MS_LOG(EXCEPTION) << "Unknown node type. node is '" << node->ToString() << "'";
}

std::string DebuggerProtoExporter::GetFuncGraphProtoString(const FuncGraphPtr &func_graph,
                                                           LocDebugDumpMode dump_location) {
  if (func_graph == nullptr) {
    return "";
  }

  InitModelInfo();
  debugger::GraphProto *graph_proto = model_.mutable_graph();
  ExportFuncGraph(func_graph, graph_proto, dump_location);
  return model_.SerializeAsString();
}

debugger::ModelProto DebuggerProtoExporter::GetFuncGraphProto(const FuncGraphPtr &func_graph) {
  if (func_graph == nullptr) {
    return ModelProto();
  }

  InitModelInfo();
  debugger::GraphProto *graph_proto = model_.mutable_graph();
  ExportFuncGraph(func_graph, graph_proto);
  return model_;
}

void DebuggerProtoExporter::ExportFuncGraph(const FuncGraphPtr &func_graph, debugger::GraphProto *const graph_proto,
                                            LocDebugDumpMode dump_location) {
  if (func_graph == nullptr || graph_proto == nullptr) {
    return;
  }

  // map for store ValueNodes of this graph
  std::map<AnfNodePtr, size_t> const_map;

  // set graph name
  graph_proto->set_name(func_graph->ToString());

  MS_LOG(INFO) << "graph names: " << func_graph->ToString();

  // cast FuncGraph to KernelGraph to access root_graph_id()
  auto kernel_graph = static_cast<session::KernelGraph *>(func_graph.get());
  uint32_t root_graph_id = kernel_graph->root_graph_id();
  uint32_t graph_id = kernel_graph->graph_id();
  MS_LOG(INFO) << "root graph id: " << root_graph_id;

  // set root graph id
  if (kernel_graph->is_graph_run_mode()) {
    graph_proto->set_root_name(std::to_string(root_graph_id));
  } else {
    graph_proto->set_root_name(std::to_string(graph_id));
  }

  ExportParameters(func_graph, graph_proto);

  ExportCNodes(func_graph, graph_proto, &const_map, dump_location);

  ExportValueNodes(const_map, graph_proto);
}

void DebuggerProtoExporter::ExportParameters(const FuncGraphPtr &func_graph, debugger::GraphProto *graph_proto) const {
  if (func_graph == nullptr || graph_proto == nullptr) {
    return;
  }

  // cast FuncGraph to KernelGraph to access inputs()
  std::vector<AnfNodePtr> parameters = static_cast<session::KernelGraph *>(func_graph.get())->inputs();

  for (auto &param : parameters) {
    debugger::ParameterProto *param_proto = graph_proto->add_parameters();
    param_proto->set_name(param->ToString());

    SetNodeOutputType(param, param_proto->mutable_type());

    const ParameterPtr param_ptr = dyn_cast<Parameter>(param);
    if (param_ptr == nullptr) {
      MS_LOG(INFO) << "Parameter '" << param->ToString() << "' could not cast to parameter.";
    }
  }
}

void DebuggerProtoExporter::ExportCNodes(const FuncGraphPtr &func_graph, debugger::GraphProto *const graph_proto,
                                         std::map<AnfNodePtr, size_t> *const_map_ptr, LocDebugDumpMode dump_location) {
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
      ExportCNode(func_graph, cnode, &apply_map, const_map_ptr, graph_proto, dump_location);
    } else {
      ExportFuncGraphOutput(func_graph, cnode, apply_map, const_map_ptr, graph_proto);
    }
  }
}

void DebuggerProtoExporter::ExportCNode(const FuncGraphPtr &func_graph, const CNodePtr &node,
                                        std::map<AnfNodePtr, size_t> *apply_map_ptr,
                                        std::map<AnfNodePtr, size_t> *const_map_ptr,
                                        debugger::GraphProto *const graph_proto, LocDebugDumpMode dump_location) {
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
  debugger::NodeProto *node_proto = graph_proto->add_node();

  // CNode/ConstGraph/Const/Parameter
  if (op->isa<CNode>() || IsValueNode<FuncGraph>(op) || op->isa<Parameter>()) {
    MS_LOG(WARNING) << "Operator must be a primitive";
  } else {
    GetOpNodeTypeAndAttrs(func_graph, op, node_proto);
    node_proto->set_name(std::to_string(apply_idx));
    node_proto->set_scope(node->scope()->name());

    // add full_name for debugger
    std::string full_name = GetKernelNodeName(node);
    node_proto->set_full_name(full_name);
    MS_LOG(INFO) << "full_name: " << full_name;
    if (dump_location == kDebugWholeStack) {
      std::ostringstream buffer;
      auto traces = mindspore::trace::GetSourceLineList(node);
      for (auto &trace : traces) {
        buffer << "      # " << trace;
      }
      node_proto->set_source_address(buffer.str());
    }
    // process OP inputs
    for (size_t i = 1; i < inputs.size(); ++i) {
      debugger::InputProto *input_proto = node_proto->add_input();
      input_proto->set_type(debugger::InputProto_EdgeType_DATA_EDGE);
      std::string id = GetOpNodeInputId(func_graph, inputs[i], *apply_map_ptr, const_map_ptr);
      input_proto->set_name(id);
    }

    // set node output type
    SetNodeOutputType(node, node_proto->mutable_output_type());
  }
}

void DebuggerProtoExporter::ExportFuncGraphOutput(const FuncGraphPtr &func_graph, const CNodePtr &ret_node,
                                                  const std::map<AnfNodePtr, size_t> &apply_map,
                                                  std::map<AnfNodePtr, size_t> *const_map_ptr,
                                                  debugger::GraphProto *graph_proto) const {
  if (ret_node == nullptr || !ret_node->isa<CNode>()) {
    MS_LOG(EXCEPTION) << "Graph return node is illegal";
  }
  AnfNodePtr arg = ret_node->input(1);
  if (graph_proto == nullptr) {
    MS_LOG(EXCEPTION) << "graph_proto is nullptr";
  }
  debugger::OutputProto *output_proto = graph_proto->add_outputs();
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

void DebuggerProtoExporter::ExportValueNodes(const std::map<AnfNodePtr, size_t> &const_map,
                                             debugger::GraphProto *graph_proto) const {
  std::vector<std::pair<AnfNodePtr, size_t>> nodes;
  (void)std::transform(const_map.cbegin(), const_map.cend(), std::back_inserter(nodes),
                       [](const std::pair<AnfNodePtr, size_t> &item) { return item; });

  sort(nodes.begin(), nodes.end(), CompareValue);

  for (auto &item : nodes) {
    if (graph_proto == nullptr) {
      MS_LOG(EXCEPTION) << "graph_proto is nullptr";
    }
    debugger::NamedValueProto *named_value = graph_proto->add_const_vals();
    MS_EXCEPTION_IF_NULL(named_value);
    named_value->set_key(GetConstNodeId(item.second));

    // cst full name: Default--data-x
    std::string node_name = GetKernelNodeName(item.first);
    GetFileKernelName(NOT_NULL(&node_name));
    named_value->set_full_name(node_name);
    if (GetValueNode(item.first)->isa<tensor::Tensor>()) {
      continue;
    }
    SetValueToProto(GetValueNode(item.first), named_value->mutable_value());
  }
}

void DebuggerProtoExporter::InitModelInfo() { model_.set_ir_version(static_cast<int64_t>(debugger::IR_VERSION)); }

debugger::ModelProto GetDebuggerFuncGraphProto(const FuncGraphPtr &func_graph) {
  DebuggerProtoExporter exporter;
  return exporter.GetFuncGraphProto(func_graph);
}

debugger::DataType GetDebuggerNumberDataType(const TypePtr &type) {
  switch (type->type_id()) {
    case kNumberTypeBool:
      return debugger::DT_BOOL;
    case kNumberTypeInt8:
      return debugger::DT_INT8;
    case kNumberTypeInt16:
      return debugger::DT_INT16;
    case kNumberTypeInt32:
      return debugger::DT_INT32;
    case kNumberTypeInt64:
      return debugger::DT_INT64;
    case kNumberTypeUInt8:
      return debugger::DT_UINT8;
    case kNumberTypeUInt16:
      return debugger::DT_UINT16;
    case kNumberTypeUInt32:
      return debugger::DT_UINT32;
    case kNumberTypeUInt64:
      return debugger::DT_UINT64;
    case kNumberTypeFloat16:
      return debugger::DT_FLOAT16;
    case kNumberTypeFloat32:
      return debugger::DT_FLOAT32;
    case kNumberTypeFloat64:
      return debugger::DT_FLOAT64;
    case kNumberTypeInt:
      return debugger::DT_BASE_INT;
    case kNumberTypeUInt:
      return debugger::DT_BASE_UINT;
    case kNumberTypeFloat:
      return debugger::DT_BASE_FLOAT;
    default:
      MS_LOG(EXCEPTION) << "Unexpected type " << type->type_name();
  }
}

#ifdef ENABLE_DUMP_IR
void DumpIRProtoWithSrcInfo(const FuncGraphPtr &func_graph, const std::string &suffix, const std::string &target_dir,
                            LocDebugDumpMode dump_location) {
  DebuggerProtoExporter exporter;
  std::string graph_proto = exporter.GetFuncGraphProtoString(func_graph, dump_location);
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "Func graph is nullptr";
    return;
  }
  std::string file_path = target_dir + "/" + "ms_output_" + suffix + ".pb";
  auto realpath = Common::CreatePrefixPath(file_path);
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Get real path failed, path=" << file_path;
    return;
  }
  ChangeFileMode(realpath.value(), S_IWUSR);

  // write to pb file
  std::ofstream ofs(realpath.value());
  if (!ofs.is_open()) {
    MS_LOG(ERROR) << "Open file '" << realpath.value() << "' failed!" << ErrnoToString(errno);
    return;
  }
  ofs << graph_proto;
  ofs.close();
  // set file mode to read only by user
  ChangeFileMode(file_path, S_IRUSR);
}

void DumpConstantInfo(const KernelGraphPtr &graph, const std::string &target_dir) {
  // Dump constant to npy file
  MS_LOG(INFO) << "Start e2e dump Const values";
  auto debugger = Debugger::GetInstance();
  E2eDump::DumpConstantData(graph.get(), target_dir, debugger.get());
}
#else
void DumpIRProtoWithSrcInfo(const FuncGraphPtr &, const std::string &, const std::string &, LocDebugDumpMode) {
  static bool already_printed = false;
  if (already_printed) {
    return;
  }
  already_printed = true;
  MS_LOG(WARNING) << "The functionality of dumping function graph IR in protobuf format is disabled,"
                  << "because ENABLE_DEBUGGER option is off"
                  << "please recompile source to enable it. See help of building script.";
}
void DumpConstantInfo(const KernelGraphPtr &, const std::string &) {
  static bool already_printed = false;
  if (already_printed) {
    return;
  }
  already_printed = true;
  MS_LOG(WARNING) << "The functionality of dumping function graph constant is disabled, "
                  << "please recompile source to enable it. See help of building script.";
}
#endif
}  // namespace mindspore

/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include <functional>

#include "ir/tensor.h"
#include "ir/param_value.h"
#include "debug/anf_ir_utils.h"
#include "frontend/operator/ops.h"
#include "proto/onnx.pb.h"

namespace mindspore {
using FloatPtr = std::shared_ptr<Float>;
using IntPtr = std::shared_ptr<Int>;

// anf type to onnx type map
static std::unordered_map<int, onnx::TensorProto_DataType> g_data_type_map = {
  {kNumberTypeBool, onnx::TensorProto_DataType_BOOL},     {kNumberTypeInt8, onnx::TensorProto_DataType_INT8},
  {kNumberTypeInt16, onnx::TensorProto_DataType_INT16},   {kNumberTypeInt32, onnx::TensorProto_DataType_INT32},
  {kNumberTypeInt64, onnx::TensorProto_DataType_INT64},   {kNumberTypeUInt8, onnx::TensorProto_DataType_UINT8},
  {kNumberTypeUInt16, onnx::TensorProto_DataType_UINT16}, {kNumberTypeUInt32, onnx::TensorProto_DataType_UINT32},
  {kNumberTypeUInt64, onnx::TensorProto_DataType_UINT64}, {kNumberTypeFloat16, onnx::TensorProto_DataType_FLOAT16},
  {kNumberTypeFloat32, onnx::TensorProto_DataType_FLOAT}, {kNumberTypeFloat64, onnx::TensorProto_DataType_DOUBLE},
  {kObjectTypeString, onnx::TensorProto_DataType_STRING},
};

static std::unordered_map<int, onnx::TensorProto_DataType> g_data_bits_int_map = {
  {8, onnx::TensorProto_DataType_INT8},
  {16, onnx::TensorProto_DataType_INT16},
  {32, onnx::TensorProto_DataType_INT32},
  {64, onnx::TensorProto_DataType_INT64},
};

static std::unordered_map<int, onnx::TensorProto_DataType> g_data_bits_float_map = {
  {16, onnx::TensorProto_DataType_FLOAT16},
  {32, onnx::TensorProto_DataType_FLOAT},
};

// Can build different builder according to format
class IrExportBuilder;
using IrExportBuilderPtr = std::shared_ptr<IrExportBuilder>;

class IrExporter {
 public:
  explicit IrExporter(IrExportBuilderPtr builder) : builder_(builder) {}
  virtual ~IrExporter() = default;
  std::string GetDumpString(const FuncGraphPtr &func_graph);

 private:
  IrExportBuilderPtr builder_;
};

class IrExportBuilder {
 public:
  IrExportBuilder() = default;
  ~IrExportBuilder() { google::protobuf::ShutdownProtobufLibrary(); }
  std::string GetProtoString(const FuncGraphPtr &func_graph);
  void BuildModelInfo();
  void BuildModel(const FuncGraphPtr &func_graph);

 private:
  void BuildFuncGraph(const FuncGraphPtr &func_graph, onnx::GraphProto *const graph_proto);
  void BuildParameters(const FuncGraphPtr &func_graph, onnx::GraphProto *const graph_proto);
  void BuildNodes(const FuncGraphPtr &func_graph, onnx::GraphProto *const graph_proto);
  void BuildOutput(const CNodePtr &node, onnx::GraphProto *const graph_proto);
  void BuildCNode(const CNodePtr &node, onnx::GraphProto *const graph_proto);
  std::string BuildInputNode(const AnfNodePtr &node, onnx::GraphProto *const graph_proto);

  void SetValueInfoProto(const AnfNodePtr &node, onnx::ValueInfoProto *const value_proto);
  void SetValueInfoProto(const TypePtr &type, const BaseShapePtr &shape, onnx::ValueInfoProto *const value_proto);
  void SetParamToTensorProto(const ParameterPtr &param, onnx::TensorProto *const tensor_proto);
  void SetTensorProto(const TypePtr &type, const BaseShapePtr &shape, onnx::TensorProto *const tensor_proto);
  void SetAttributeProto(const AnfNodePtr &node, onnx::NodeProto *const node_proto);
  void SetShapeToNodeProto(const CNodePtr &node, onnx::NodeProto *const node_proto);
  void SetShapeToNodeProto(const TypePtr &type, const BaseShapePtr &shape, onnx::AttributeProto *const attr_proto,
                           std::string *const seq_string);
  void SetValueToAttributeProto(const ValuePtr &value, onnx::AttributeProto *const attr_proto);
  void SetTypeToAttributeProto(const ValuePtr &value, onnx::AttributeProto *const attr_proto);
  void SetScalarToAttributeProto(const ValuePtr &value, onnx::AttributeProto *const attr_proto);
  void SetTensorToAttributeProto(const ValuePtr &value, onnx::AttributeProto *const attr_proto);
  void SetScalarToProto(const ValuePtr &value, onnx::TensorProto *const tensor_proto, const std::string &value_name);
  void SetSequenceToAttributeProto(const ValueSequeuePtr &value, onnx::AttributeProto *const attr_proto,
                                   std::string *const seq_string);
  void SetSeqElemToAttributeProto(const ValuePtr &value, onnx::AttributeProto *const attr_proto,
                                  std::string *const seq_string);

  onnx::TensorProto_DataType GetOnnxDataType(TypeId type_id);
  onnx::TensorProto_DataType GetOnnxDataBitsIntType(int bits);
  onnx::TensorProto_DataType GetOnnxDataBitsFloatType(int bits);
  std::string GetNodeName(const AnfNodePtr &node);
  std::string GetUniqueNodeName(const AnfNodePtr &node);
  std::string GetOpTypeName(const AnfNodePtr &node);
  size_t GetNodeIndex() { return ++node_index_; }
  void ResetNodeIndex() { node_index_ = 0; }
  size_t GetTupleIndex() { return ++shape_index_; }
  void ResetTupleIndex() { shape_index_ = 0; }

 private:
  onnx::ModelProto model_;
  onnx::NodeProto *last_node_{nullptr};
  std::list<FuncGraphPtr> todo_;
  std::map<AnfNodePtr, size_t> node_index_map_;
  size_t node_index_{0};
  size_t shape_index_{0};
};

using IrExporterPtr = std::shared_ptr<IrExporter>;

std::string IrExporter::GetDumpString(const FuncGraphPtr &func_graph) {
  if ((builder_ == nullptr) || (func_graph == nullptr)) {
    MS_LOG(EXCEPTION) << "Input params is null.";
  }

  // Export model info
  builder_->BuildModelInfo();

  // Export model and return string
  builder_->BuildModel(func_graph);

  return builder_->GetProtoString(func_graph);
}

std::string IrExportBuilder::GetProtoString(const FuncGraphPtr &func_graph) {
  MS_LOG(DEBUG) << "BuildModel complete!";
  return model_.SerializeAsString();
}

void IrExportBuilder::BuildModelInfo() {
  model_.set_ir_version(onnx::IR_VERSION_2019_1_22);
  model_.set_producer_name("MindSpore");
  model_.set_model_version(1);
}

void IrExportBuilder::BuildModel(const FuncGraphPtr &func_graph) {
  onnx::GraphProto *graph_proto = model_.mutable_graph();
  graph_proto->set_name(func_graph->ToString());
  ResetNodeIndex();
  todo_.clear();
  todo_.push_back(func_graph);
  while (!todo_.empty()) {
    FuncGraphPtr fg = todo_.back();
    todo_.pop_back();
    BuildFuncGraph(fg, graph_proto);
  }
}

void IrExportBuilder::BuildFuncGraph(const FuncGraphPtr &func_graph, onnx::GraphProto *const graph_proto) {
  // Export parameters
  // 1. parameters should be mapped to ValueInfoProto
  // 2. parameters with default value should be mapped to Initializer
  BuildParameters(func_graph, graph_proto);

  // Export operator nodes(include output)
  BuildNodes(func_graph, graph_proto);
}

void IrExportBuilder::BuildParameters(const FuncGraphPtr &func_graph, onnx::GraphProto *const graph_proto) {
  for (auto &item : func_graph->parameters()) {
    auto param = item->cast<ParameterPtr>();
    if (param == nullptr) {
      MS_LOG(EXCEPTION) << "Parameter: '" << item->ToString() << "' could not cast to parameter.";
    }
    onnx::ValueInfoProto *input_proto = graph_proto->add_input();
    std::string param_name = GetUniqueNodeName(param);
    input_proto->set_name(param_name);
    SetValueInfoProto(param, input_proto);
    if (!param->has_default()) {
      MS_LOG(DEBUG) << "Parameter: '" << item->ToString() << "' has no default.";
      continue;
    }

    // Using ONNX initializer to set parameter's default value
    onnx::TensorProto *initializer_proto = graph_proto->add_initializer();
    initializer_proto->set_name(param_name);
    SetParamToTensorProto(param, initializer_proto);
    auto tensor = std::dynamic_pointer_cast<tensor::Tensor>(param->default_param()->value());
    if (tensor) {
      initializer_proto->set_raw_data(tensor->data_c(), tensor->data().nbytes());
    }
  }
}

onnx::TensorProto_DataType IrExportBuilder::GetOnnxDataType(TypeId type_id) {
  auto iter = g_data_type_map.find(type_id);
  if (iter == g_data_type_map.end()) {
    MS_LOG(EXCEPTION) << "Convert type error, unsupported type! " << type_id;
  }
  return iter->second;
}

onnx::TensorProto_DataType IrExportBuilder::GetOnnxDataBitsIntType(int bits) {
  auto iter = g_data_bits_int_map.find(bits);
  if (iter == g_data_bits_int_map.end()) {
    MS_LOG(EXCEPTION) << "Convert bits int error, unsupported bits! " << bits;
  }
  return iter->second;
}

onnx::TensorProto_DataType IrExportBuilder::GetOnnxDataBitsFloatType(int bits) {
  auto iter = g_data_bits_float_map.find(bits);
  if (iter == g_data_bits_float_map.end()) {
    MS_LOG(EXCEPTION) << "Convert bits float error, unsupported bits! " << bits;
  }
  return iter->second;
}

void IrExportBuilder::SetValueInfoProto(const AnfNodePtr &node, onnx::ValueInfoProto *const value_proto) {
  if (node == nullptr || value_proto == nullptr) {
    MS_LOG(EXCEPTION) << "AnfNode or ValueInfo is null!";
  }
  MS_LOG(DEBUG) << "SetValueInfoProto: " << node->DebugString();
  SetValueInfoProto(node->Type(), node->Shape(), value_proto);
}

void IrExportBuilder::SetValueInfoProto(const TypePtr &type, const BaseShapePtr &shape,
                                        onnx::ValueInfoProto *const value_proto) {
  onnx::TypeProto *type_proto = value_proto->mutable_type();
  if (type->isa<TensorType>() && shape->isa<abstract::Shape>()) {
    auto tensor = type->cast<TensorTypePtr>();
    auto elem_type = tensor->element();
    const auto &dims = shape->cast<abstract::ShapePtr>()->shape();
    type_proto->mutable_tensor_type()->set_elem_type(GetOnnxDataType(elem_type->type_id()));
    if (dims.size() == 0) {
      MS_LOG(DEBUG) << "SetValueInfoProto set default dim 1.";
      type_proto->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
    } else {
      for (const auto &dim : dims) {
        MS_LOG(DEBUG) << "SetValueInfoProto dim: " << dim;
        type_proto->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dim);
      }
    }
  } else if (type->isa<Tuple>()) {
    auto tup_shape = shape->cast<abstract::TupleShapePtr>();
    type_proto->set_denotation(type->type_name() + ":" + std::to_string(tup_shape->shape().size()));
  } else if (type->isa<Number>() || type->isa<String>()) {
    type_proto->set_denotation(type->type_name());
  } else {
    MS_LOG(EXCEPTION) << "Value type: " << type->type_name() << " is not supported!";
  }
}

void IrExportBuilder::SetTensorToAttributeProto(const ValuePtr &value, onnx::AttributeProto *const attr_proto) {
  if (value == nullptr || attr_proto == nullptr) {
    MS_LOG(EXCEPTION) << "ValuePtr or AttributeProto is null!";
  }
  attr_proto->set_ref_attr_name("tensor:value0");
  attr_proto->set_type(onnx::AttributeProto_AttributeType_TENSORS);
  onnx::TensorProto *tensor_proto = attr_proto->add_tensors();
  tensor_proto->set_name("value0");
  auto data = value->cast<tensor::TensorPtr>();
  tensor_proto->set_raw_data(data->data_c(), static_cast<size_t>(data->data().nbytes()));
  auto dtype = data->data_type();
  auto shape = data->shape_c();
  tensor_proto->set_data_type(GetOnnxDataType(dtype));
  for (const auto &dim : shape) {
    tensor_proto->add_dims(dim);
  }
}

void IrExportBuilder::SetTensorProto(const TypePtr &type, const BaseShapePtr &shape,
                                     onnx::TensorProto *const tensor_proto) {
  if (!type->isa<TensorType>() || !shape->isa<abstract::Shape>()) {
    MS_LOG(EXCEPTION) << "Type or shape is not supported! " << type->ToString();
  }
  auto tensor = type->cast<TensorTypePtr>();
  const auto &dims = shape->cast<abstract::ShapePtr>()->shape();
  tensor_proto->set_data_type(GetOnnxDataType(tensor->element()->type_id()));
  for (const auto &dim : dims) {
    tensor_proto->add_dims(dim);
  }
}

void IrExportBuilder::SetParamToTensorProto(const ParameterPtr &param, onnx::TensorProto *const tensor_proto) {
  if (param == nullptr || tensor_proto == nullptr) {
    MS_LOG(EXCEPTION) << "Parameter or TensorProto is null!";
  }
  MS_LOG(DEBUG) << "SetParamToTensorProto: " << param->DebugString();
  SetTensorProto(param->Type(), param->Shape(), tensor_proto);
}

void IrExportBuilder::BuildNodes(const FuncGraphPtr &func_graph, onnx::GraphProto *const graph_proto) {
  std::vector<AnfNodePtr> nodes = TopoSort(func_graph->get_return(), SuccIncoming, AlwaysInclude);
  bool is_only_return = true;
  for (const AnfNodePtr &node : nodes) {
    if (!node->isa<CNode>()) {
      MS_LOG(DEBUG) << "Node: '" << node->ToString() << "' is not cnode";
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (cnode == func_graph->get_return()) {
      if (is_only_return) {
        MS_LOG(EXCEPTION) << "Only has return node, can't convert to binary model!";
      }
      BuildOutput(cnode, graph_proto);
    } else {
      BuildCNode(cnode, graph_proto);
      is_only_return = false;
    }
  }
}

void IrExportBuilder::BuildOutput(const CNodePtr &node, onnx::GraphProto *const graph_proto) {
  if (node->size() != 2) {
    MS_LOG(EXCEPTION) << "Number of inputs of return node is not equal to 2.";
  }
  AnfNodePtr arg = node->input(1);
  onnx::ValueInfoProto *output_proto = graph_proto->add_output();
  std::string output_name = GetUniqueNodeName(node);
  output_proto->set_name(output_name);
  last_node_->set_output(0, output_name);
  SetValueInfoProto(arg, output_proto);
}

std::string IrExportBuilder::GetOpTypeName(const AnfNodePtr &node) {
  // May be ValueNode/CNode/Parameter
  std::string type_name = "";
  if (IsValueNode<Primitive>(node)) {
    PrimitivePtr prim = GetValueNode<PrimitivePtr>(node);
    type_name = prim->ToString();
  } else if (IsValueNode<FuncGraph>(node)) {
    FuncGraphPtr fg = GetValueNode<FuncGraphPtr>(node);
    todo_.push_back(fg);
    type_name = fg->ToString();
  } else if (node->isa<CNode>() || node->isa<Parameter>()) {
    type_name = node->ToString();
  } else {
    MS_LOG(EXCEPTION) << "Need to support op type: " << node->type_name();
  }
  MS_LOG(DEBUG) << "ExportType: " << type_name;
  return type_name;
}

void IrExportBuilder::SetShapeToNodeProto(const TypePtr &type, const BaseShapePtr &shape,
                                          onnx::AttributeProto *const attr_proto, std::string *const seq_string) {
  if (type->isa<Tuple>() && seq_string != nullptr) {
    *seq_string += "Tuple[";
    auto elements = type->cast<TuplePtr>()->elements();
    auto tuple_shape = shape->cast<abstract::TupleShapePtr>()->shape();
    for (size_t i = 0; i < elements.size(); i++) {
      SetShapeToNodeProto(elements[i], tuple_shape[i], attr_proto, seq_string);
    }
    *seq_string += "],";
  } else if (type->isa<TensorType>() && shape->isa<abstract::Shape>() && seq_string != nullptr) {
    string shape_name = "shape" + std::to_string(GetTupleIndex());
    *seq_string += shape_name + ",";
    onnx::TensorProto *tensor_proto = attr_proto->add_tensors();
    tensor_proto->set_name(shape_name);
    SetTensorProto(type, shape, tensor_proto);
  } else if ((type->isa<Number>() || type->isa<String>()) && seq_string != nullptr) {
    *seq_string += type->type_name() + ",";
  } else {
    MS_LOG(EXCEPTION) << "Type of cnode need to be supported: " << type->type_name();
  }
}

void IrExportBuilder::SetShapeToNodeProto(const CNodePtr &node, onnx::NodeProto *const node_proto) {
  // Get shape of cnode
  // 1. need to get shape from tuple element
  // 2. save shape in TensorProto
  // 3. save tuple string in ref_attr_name
  auto type = node->Type();
  auto shape = node->Shape();
  ResetTupleIndex();
  std::string seq_string = "shape:";
  onnx::AttributeProto *attr_proto = node_proto->add_attribute();
  SetShapeToNodeProto(type, shape, attr_proto, &seq_string);
  attr_proto->set_ref_attr_name(seq_string);
  MS_LOG(DEBUG) << "CNode shape: " << seq_string;
}

void IrExportBuilder::BuildCNode(const CNodePtr &node, onnx::GraphProto *const graph_proto) {
  auto inputs_size = node->size();
  if (inputs_size < 1) {
    MS_LOG(EXCEPTION) << "Inputs of apply node is empty";
  }

  // Need to build input node before dealing with cnode
  std::vector<AnfNodePtr> op_inputs;
  std::vector<string> input_names;
  for (size_t i = 1; i < inputs_size; i++) {
    auto input = node->input(i);
    op_inputs.push_back(input);
    input_names.push_back(BuildInputNode(input, graph_proto));
  }

  // Build cnode
  onnx::NodeProto *node_proto = graph_proto->add_node();
  std::string output_name = GetUniqueNodeName(node);
  node_proto->add_output(output_name);
  node_proto->set_name(output_name);
  node_proto->set_domain(node->fullname_with_scope());
  AnfNodePtr op = node->input(0);
  std::string type_name = GetOpTypeName(op);
  node_proto->set_op_type(type_name);
  last_node_ = node_proto;
  SetShapeToNodeProto(node, node_proto);
  (void)std::for_each(input_names.begin(), input_names.end(),
                      [&node_proto](const string &name) { node_proto->add_input(name); });

  // Add primitive attrs
  if (IsValueNode<Primitive>(op)) {
    auto prim = GetValueNode<PrimitivePtr>(op);
    for (auto attr : prim->attrs()) {
      MS_LOG(DEBUG) << "attr: " << attr.first << " " << attr.second->DumpText() << " " << attr.second->type_name();
      onnx::AttributeProto *attr_proto = node_proto->add_attribute();
      attr_proto->set_name(attr.first);
      SetValueToAttributeProto(attr.second, attr_proto);
    }
  } else {
    MS_LOG(EXCEPTION) << "Need to support op type: " << op->type_name();
  }
}

std::string IrExportBuilder::BuildInputNode(const AnfNodePtr &node, onnx::GraphProto *const graph_proto) {
  std::string node_name = GetUniqueNodeName(node);
  if (node->isa<ValueNode>()) {
    // When node input is a ValueNode, need to create a Constant Node
    onnx::NodeProto *node_proto = graph_proto->add_node();
    node_proto->add_output(node_name);
    SetAttributeProto(node, node_proto);
  }
  return node_name;
}

std::string IrExportBuilder::GetUniqueNodeName(const AnfNodePtr &node) {
  // Naming anfnode
  // 1. parameter is unique in one func_graph
  // 2. cnode and valuenode may be reduplicative, so add index to identify.
  std::string node_name = "";
  if (node->isa<Parameter>()) {
    node_name = GetNodeName(node);
  } else if (node->isa<CNode>()) {
    auto iter = node_index_map_.find(node);
    if (iter != node_index_map_.end()) {
      node_name = GetNodeName(node) + ":" + std::to_string(iter->second);
    } else {
      auto node_idx = GetNodeIndex();
      node_index_map_[node] = node_idx;
      node_name = GetNodeName(node) + ":" + std::to_string(node_idx);
    }
  } else if (node->isa<ValueNode>()) {
    auto node_idx = GetNodeIndex();
    node_index_map_[node] = node_idx;
    node_name = GetNodeName(node) + ":" + std::to_string(node_idx);
  } else {
    MS_LOG(EXCEPTION) << "Can not support type of node:" << node->ToString();
  }
  MS_LOG(DEBUG) << "Node name: " << node_name;
  return node_name;
}

std::string IrExportBuilder::GetNodeName(const AnfNodePtr &node) {
  std::string node_name = "";
  if ((node != nullptr) && (node->func_graph() != nullptr)) {
    node_name = node->func_graph()->ToString() + ":";
  }
  node_name += node->ToString();
  MS_LOG(DEBUG) << "GetNodeName: " << node_name;
  return node_name;
}

void IrExportBuilder::SetAttributeProto(const AnfNodePtr &node, onnx::NodeProto *const node_proto) {
  if (node == nullptr || node_proto == nullptr) {
    MS_LOG(EXCEPTION) << "AnfNode or NodeProto is null!";
  }
  auto value = node->cast<ValueNodePtr>()->value();
  node_proto->set_op_type("Constant");
  onnx::AttributeProto *attr_proto = node_proto->add_attribute();
  attr_proto->set_name("value");
  MS_LOG(DEBUG) << "Set Constant attribute: " << value->ToString();
  SetValueToAttributeProto(value, attr_proto);
}

void IrExportBuilder::SetTypeToAttributeProto(const ValuePtr &value, onnx::AttributeProto *const attr_proto) {
  if (value == nullptr || attr_proto == nullptr) {
    MS_LOG(EXCEPTION) << "ValuePtr or AttributeProto is null!";
  }
  attr_proto->set_type(onnx::AttributeProto_AttributeType_TENSORS);
  onnx::TensorProto *tensor_proto = attr_proto->add_tensors();
  if (value->isa<Int>()) {
    attr_proto->set_ref_attr_name("type:value0");
    tensor_proto->set_name("value0");
    auto int_value = value->cast<IntPtr>();
    tensor_proto->set_data_type(GetOnnxDataBitsIntType(int_value->nbits()));
  } else if (value->isa<Float>()) {
    attr_proto->set_ref_attr_name("type:value0");
    tensor_proto->set_name("value0");
    auto float_value = value->cast<FloatPtr>();
    tensor_proto->set_data_type(GetOnnxDataBitsFloatType(float_value->nbits()));
  } else if (value->isa<TensorType>()) {
    attr_proto->set_ref_attr_name("type:tensor0");
    tensor_proto->set_name("tensor0");
    auto elem_type = value->cast<TensorTypePtr>()->element();
    if (elem_type->isa<Int>()) {
      auto int_value = elem_type->cast<IntPtr>();
      tensor_proto->set_data_type(GetOnnxDataBitsIntType(int_value->nbits()));
    } else if (elem_type->isa<Float>()) {
      auto float_value = elem_type->cast<FloatPtr>();
      tensor_proto->set_data_type(GetOnnxDataBitsFloatType(float_value->nbits()));
    } else {
      MS_LOG(EXCEPTION) << "Unsupported type " << elem_type->type_name();
    }
  } else {
    MS_LOG(EXCEPTION) << "Unsupported type: " << value->type_name();
  }
}

void IrExportBuilder::SetValueToAttributeProto(const ValuePtr &value, onnx::AttributeProto *const attr_proto) {
  if (value == nullptr || attr_proto == nullptr) {
    MS_LOG(EXCEPTION) << "ValuePtr or AttributeProto is null!";
  }
  if (value->isa<StringImm>() || value->isa<Scalar>()) {
    SetScalarToAttributeProto(value, attr_proto);
  } else if (value->isa<Number>() || value->isa<TensorType>()) {
    SetTypeToAttributeProto(value, attr_proto);
  } else if (value->isa<ValueSequeue>() || value->isa<ValueSequeue>()) {
    ResetTupleIndex();
    std::string seq_string = "scalar:";
    attr_proto->set_type(onnx::AttributeProto_AttributeType_TENSORS);
    SetSequenceToAttributeProto(value->cast<ValueSequeuePtr>(), attr_proto, &seq_string);
    attr_proto->set_ref_attr_name(seq_string);
    MS_LOG(DEBUG) << "Attr string: " << seq_string;
  } else if (value->isa<tensor::Tensor>()) {
    SetTensorToAttributeProto(value, attr_proto);
  } else if (value->isa<None>()) {
    attr_proto->set_ref_attr_name("none");
    MS_LOG(DEBUG) << "Attr string: " << value->type_name();
  } else {
    MS_LOG(EXCEPTION) << "Unsupported type: " << value->type_name();
  }
}

void IrExportBuilder::SetScalarToAttributeProto(const ValuePtr &value, onnx::AttributeProto *const attr_proto) {
  if (value == nullptr || attr_proto == nullptr) {
    MS_LOG(EXCEPTION) << "ValuePtr or AttributeProto is null!";
  }
  attr_proto->set_ref_attr_name("scalar:value0");
  attr_proto->set_type(onnx::AttributeProto_AttributeType_TENSORS);
  onnx::TensorProto *tensor_proto = attr_proto->add_tensors();
  SetScalarToProto(value, tensor_proto, "value0");
}

void IrExportBuilder::SetScalarToProto(const ValuePtr &value, onnx::TensorProto *const tensor_proto,
                                       const std::string &value_name) {
  if (value == nullptr || tensor_proto == nullptr) {
    MS_LOG(EXCEPTION) << "ValuePtr or TensorProto is null!";
  }
  tensor_proto->set_name(value_name);
  if (value->isa<StringImm>()) {
    tensor_proto->set_data_type(onnx::TensorProto_DataType_STRING);
    tensor_proto->add_string_data(GetValue<std::string>(value));
  } else if (value->isa<BoolImm>()) {
    tensor_proto->set_data_type(onnx::TensorProto_DataType_BOOL);
    tensor_proto->add_int32_data(GetValue<bool>(value));
  } else if (value->isa<Int8Imm>()) {
    tensor_proto->set_data_type(onnx::TensorProto_DataType_INT8);
    tensor_proto->add_int32_data(value->cast<Int8ImmPtr>()->value());
  } else if (value->isa<Int16Imm>()) {
    tensor_proto->set_data_type(onnx::TensorProto_DataType_INT16);
    tensor_proto->add_int32_data(value->cast<Int16ImmPtr>()->value());
  } else if (value->isa<Int32Imm>()) {
    tensor_proto->set_data_type(onnx::TensorProto_DataType_INT32);
    tensor_proto->add_int32_data(value->cast<Int32ImmPtr>()->value());
  } else if (value->isa<Int64Imm>()) {
    tensor_proto->set_data_type(onnx::TensorProto_DataType_INT64);
    tensor_proto->add_int64_data(value->cast<Int64ImmPtr>()->value());
  } else if (value->isa<UInt8Imm>()) {
    tensor_proto->set_data_type(onnx::TensorProto_DataType_UINT8);
    tensor_proto->add_int32_data(value->cast<UInt8ImmPtr>()->value());
  } else if (value->isa<UInt16Imm>()) {
    tensor_proto->set_data_type(onnx::TensorProto_DataType_UINT16);
    tensor_proto->add_int32_data(value->cast<UInt16ImmPtr>()->value());
  } else if (value->isa<UInt32Imm>()) {
    tensor_proto->set_data_type(onnx::TensorProto_DataType_UINT32);
    tensor_proto->add_uint64_data(value->cast<UInt32ImmPtr>()->value());
  } else if (value->isa<UInt64Imm>()) {
    tensor_proto->set_data_type(onnx::TensorProto_DataType_UINT64);
    tensor_proto->add_uint64_data(value->cast<UInt64ImmPtr>()->value());
  } else if (value->isa<FP32Imm>()) {
    tensor_proto->set_data_type(onnx::TensorProto_DataType_FLOAT);
    tensor_proto->add_float_data(GetValue<float>(value));
  } else if (value->isa<FP64Imm>()) {
    tensor_proto->set_data_type(onnx::TensorProto_DataType_DOUBLE);
    tensor_proto->add_double_data(GetValue<double>(value));
  } else {
    MS_LOG(EXCEPTION) << "Unsupported scalar type: " << value->type_name();
  }
}

void IrExportBuilder::SetSeqElemToAttributeProto(const ValuePtr &value, onnx::AttributeProto *const attr_proto,
                                                 std::string *const seq_string) {
  string value_name = "value" + std::to_string(GetTupleIndex());
  if (seq_string != nullptr) {
    *seq_string += value_name + ",";
  }
  onnx::TensorProto *tensor_proto = attr_proto->add_tensors();
  SetScalarToProto(value, tensor_proto, value_name);
}

void IrExportBuilder::SetSequenceToAttributeProto(const ValueSequeuePtr &value, onnx::AttributeProto *const attr_proto,
                                                  std::string *const seq_string) {
  if (value == nullptr || attr_proto == nullptr) {
    MS_LOG(EXCEPTION) << "ValueSequeuePtr or AttributeProto is null!";
  }
  if (value->isa<ValueTuple>() && seq_string != nullptr) {
    *seq_string += "Tuple[";
    const ValueTuplePtr &tuple_value = value->cast<ValueTuplePtr>();
    if (tuple_value->value().size() == 0) {
      MS_LOG(DEBUG) << "SetSequenceToAttributeProto tuple size is 0";
      return;
    }
    for (const auto &item : tuple_value->value()) {
      if (item->isa<ValueTuple>()) {
        SetSequenceToAttributeProto(item->cast<ValueTuplePtr>(), attr_proto, seq_string);
      } else {
        SetSeqElemToAttributeProto(item, attr_proto, seq_string);
      }
    }
    *seq_string += "],";
  } else if (value->isa<ValueList>() && seq_string != nullptr) {
    *seq_string += "List[";
    const ValueListPtr &list_value = value->cast<ValueListPtr>();
    if (list_value->value().size() == 0) {
      MS_LOG(DEBUG) << "SetSequenceToAttributeProto list size is 0.";
      return;
    }
    for (const auto &item : list_value->value()) {
      if (item->isa<ValueList>()) {
        SetSequenceToAttributeProto(item->cast<ValueListPtr>(), attr_proto, seq_string);
      } else {
        SetSeqElemToAttributeProto(item, attr_proto, seq_string);
      }
    }
    *seq_string += "],";
  }
}

std::string GetBinaryProtoString(const FuncGraphPtr &func_graph) {
  auto builder = std::make_shared<IrExportBuilder>();
  if (builder == nullptr) {
    MS_LOG(ERROR) << "Create ir exporter failed!";
    return "";
  }
  auto exporter = std::make_shared<IrExporter>(builder);
  if (exporter == nullptr) {
    return "";
  }
  return exporter->GetDumpString(func_graph);
}
}  // namespace mindspore

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

#include <map>
#include <memory>
#include <unordered_map>
#include <utility>
#include <algorithm>
#include <functional>

#include "ir/tensor.h"
#include "ir/param_info.h"
#include "ir/func_graph.h"
#include "base/core_ops.h"
#include "proto/mind_ir.pb.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
using FloatPtr = std::shared_ptr<Float>;
using IntPtr = std::shared_ptr<Int>;
using UIntPtr = std::shared_ptr<UInt>;
// anf type to mindir type map
static std::unordered_map<int, mind_ir::TensorProto_DataType> g_data_type_map = {
  {kNumberTypeBool, mind_ir::TensorProto_DataType_BOOL},
  {kNumberTypeInt8, mind_ir::TensorProto_DataType_INT8},
  {kNumberTypeInt16, mind_ir::TensorProto_DataType_INT16},
  {kNumberTypeInt32, mind_ir::TensorProto_DataType_INT32},
  {kNumberTypeInt64, mind_ir::TensorProto_DataType_INT64},
  {kNumberTypeUInt8, mind_ir::TensorProto_DataType_UINT8},
  {kNumberTypeUInt16, mind_ir::TensorProto_DataType_UINT16},
  {kNumberTypeUInt32, mind_ir::TensorProto_DataType_UINT32},
  {kNumberTypeUInt64, mind_ir::TensorProto_DataType_UINT64},
  {kNumberTypeFloat16, mind_ir::TensorProto_DataType_FLOAT16},
  {kNumberTypeFloat32, mind_ir::TensorProto_DataType_FLOAT},
  {kNumberTypeFloat64, mind_ir::TensorProto_DataType_DOUBLE},
  {kObjectTypeString, mind_ir::TensorProto_DataType_STRING},
};

static std::unordered_map<int, mind_ir::TensorProto_DataType> g_data_bits_int_map = {
  {8, mind_ir::TensorProto_DataType_INT8},
  {16, mind_ir::TensorProto_DataType_INT16},
  {32, mind_ir::TensorProto_DataType_INT32},
  {64, mind_ir::TensorProto_DataType_INT64},
};

static std::unordered_map<int, mind_ir::TensorProto_DataType> g_data_bits_uint_map = {
  {8, mind_ir::TensorProto_DataType_UINT8},
  {16, mind_ir::TensorProto_DataType_UINT16},
  {32, mind_ir::TensorProto_DataType_UINT32},
  {64, mind_ir::TensorProto_DataType_UINT64},
};

static std::unordered_map<int, mind_ir::TensorProto_DataType> g_data_bits_float_map = {
  {16, mind_ir::TensorProto_DataType_FLOAT16},
  {32, mind_ir::TensorProto_DataType_FLOAT},
  {64, mind_ir::TensorProto_DataType_FLOAT64},
};

// Can build different builder according to format
class IrExportBuilder;
using IrExportBuilderPtr = std::shared_ptr<IrExportBuilder>;

class IrExporter {
 public:
  explicit IrExporter(IrExportBuilderPtr builder) : builder_(std::move(builder)) {}
  virtual ~IrExporter() = default;
  std::string GetDumpString(const FuncGraphPtr &func_graph);
  mind_ir::ModelProto GetDumpProto(const FuncGraphPtr &func_graph, bool save_tensor_data = false);

 private:
  IrExportBuilderPtr builder_;
};

class IrExportBuilder {
 public:
  IrExportBuilder() = default;
  ~IrExportBuilder() { google::protobuf::ShutdownProtobufLibrary(); }
  std::string GetProtoString() const;
  void BuildModelInfo();
  void BuildModel(const FuncGraphPtr &func_graph, bool save_tensor_data = false);
  mind_ir::ModelProto Model() { return model_; }

 private:
  void BuildFuncGraph(const FuncGraphPtr &func_graph, mind_ir::GraphProto *const graph_proto,
                      bool save_tensor_data = false);
  void BuildParameters(const FuncGraphPtr &func_graph, mind_ir::GraphProto *const graph_proto,
                       bool save_tensor_data = false);
  void BuildNodes(const FuncGraphPtr &func_graph, mind_ir::GraphProto *const graph_proto);
  void BuildOutput(const CNodePtr &node, mind_ir::GraphProto *const graph_proto);
  void BuildCNode(const CNodePtr &node, mind_ir::GraphProto *const graph_proto);
  std::string BuildInputNode(const AnfNodePtr &node, mind_ir::GraphProto *const graph_proto);

  void SetValueInfoProto(const AnfNodePtr &node, mind_ir::ValueInfoProto *const value_proto);
  void SetValueInfoProto(const TypePtr &type, const BaseShapePtr &shape, mind_ir::ValueInfoProto *const value_proto);
  void SetParamToTensorProto(const ParameterPtr &param, mind_ir::TensorProto *const tensor_proto);
  void SetTensorProto(const TypePtr &type, const BaseShapePtr &shape, mind_ir::TensorProto *const tensor_proto);
  void SetAttributeProto(const AnfNodePtr &node, mind_ir::NodeProto *const node_proto);
  void SetShapeToNodeProto(const CNodePtr &node, mind_ir::NodeProto *const node_proto);
  void SetShapeToNodeProto(const TypePtr &type, const BaseShapePtr &shape, mind_ir::AttributeProto *const attr_proto,
                           std::string *const seq_string);
  void SetValueToAttributeProto(const ValuePtr &value, mind_ir::AttributeProto *const attr_proto);
  void SetTypeToAttributeProto(const ValuePtr &value, mind_ir::AttributeProto *const attr_proto);
  void SetScalarToAttributeProto_ir(const ValuePtr &value, mind_ir::AttributeProto *const attr_proto);
  void SetScalarToAttributeProto_irs(const ValuePtr &value, mind_ir::AttributeProto *const attr_proto);
  void SetTensorToAttributeProto(const ValuePtr &value, mind_ir::AttributeProto *const attr_proto);
  void SetSequenceToAttributeProto(const ValueSequeuePtr &value, mind_ir::AttributeProto *const attr_proto,
                                   std::string *const seq_string);
  void SetSeqElemToAttributeProto(const ValuePtr &value, mind_ir::AttributeProto *const attr_proto,
                                  std::string *const seq_string);

  mind_ir::TensorProto_DataType GetMindirDataType(TypeId type_id);
  mind_ir::TensorProto_DataType GetMindirDataBitsIntType(int bits);
  mind_ir::TensorProto_DataType GetMindirDataBitsFloatType(int bits);
  mind_ir::TensorProto_DataType GetMindirDataBitsUIntType(int bits);
  std::string GetNodeName(const AnfNodePtr &node);
  std::string GetUniqueNodeName(const AnfNodePtr &node);
  std::string GetOpTypeName(const AnfNodePtr &node);
  size_t GetNodeIndex() { return ++node_index_; }
  void ResetNodeIndex() { node_index_ = 0; }
  size_t GetTupleIndex() { return ++shape_index_; }
  void ResetTupleIndex() { shape_index_ = 0; }

  mind_ir::ModelProto model_;
  mind_ir::NodeProto *last_node_{nullptr};
  std::list<FuncGraphPtr> todo_;
  std::map<AnfNodePtr, std::string> node_index_map_;
  std::set<std::string> nodeName_;
  size_t node_index_{0};
  size_t shape_index_{0};
  bool top_graph{true};
};

using IrExporterPtr = std::shared_ptr<IrExporter>;

std::string IrExporter::GetDumpString(const FuncGraphPtr &func_graph) {
  (void)GetDumpProto(func_graph);
  return builder_->GetProtoString();
}

mind_ir::ModelProto IrExporter::GetDumpProto(const FuncGraphPtr &func_graph, bool save_tensor_data) {
  if ((builder_ == nullptr) || (func_graph == nullptr)) {
    MS_LOG(EXCEPTION) << "Input params is null.";
  }

  // Export model info
  builder_->BuildModelInfo();

  // Export model and return string
  builder_->BuildModel(func_graph, save_tensor_data);
  return builder_->Model();
}

std::string IrExportBuilder::GetProtoString() const {
  MS_LOG(DEBUG) << "BuildModel complete!";
  return model_.SerializeAsString();
}

void IrExportBuilder::BuildModelInfo() {
  constexpr auto ir_version = "0.1.0";
  constexpr auto mindspore_name = "MindSpore";
  model_.set_ir_version(ir_version);
  model_.set_producer_name(mindspore_name);
  model_.set_model_version(VERSION);
}

void IrExportBuilder::BuildModel(const FuncGraphPtr &func_graph, bool save_tensor_data) {
  MS_EXCEPTION_IF_NULL(func_graph);
  mind_ir::GraphProto *graph_proto = model_.mutable_graph();
  graph_proto->set_name(func_graph->ToString());
  graph_proto->set_bprop_hash(func_graph->bprop_hash());
  ResetNodeIndex();
  todo_.clear();
  nodeName_.clear();
  // Build the main funcGraph
  (void)nodeName_.insert(func_graph->ToString());
  top_graph = true;
  BuildFuncGraph(func_graph, graph_proto, save_tensor_data);
  std::set<FuncGraphPtr> graphVisited;
  (void)graphVisited.insert(func_graph);
  top_graph = false;
  while (!todo_.empty()) {
    FuncGraphPtr fg = todo_.back();
    todo_.pop_back();
    if (graphVisited.count(fg) > 0) {
      continue;
    }
    if (nodeName_.count(fg->ToString()) > 0) {
      MS_LOG(EXCEPTION) << "There is a duplicate name: " << fg->ToString();
    }
    (void)nodeName_.insert(fg->ToString());
    (void)graphVisited.insert(fg);
    auto graph = model_.add_functions();
    BuildFuncGraph(fg, graph, save_tensor_data);
  }
  // Release resource
  nodeName_.clear();
  node_index_map_.clear();
}

void IrExportBuilder::BuildFuncGraph(const FuncGraphPtr &func_graph, mind_ir::GraphProto *const graph_proto,
                                     bool save_tensor_data) {
  // Export funcGraph name.
  graph_proto->set_name(func_graph->ToString());
  // Export parameters
  // 1. parameters should be mapped to ValueInfoProto
  // 2. parameters with default value should be mapped to Initializer
  BuildParameters(func_graph, graph_proto, save_tensor_data);

  // Export operator nodes(include output)
  BuildNodes(func_graph, graph_proto);
}

void IrExportBuilder::BuildParameters(const FuncGraphPtr &func_graph, mind_ir::GraphProto *const graph_proto,
                                      bool save_tensor_data) {
  MS_EXCEPTION_IF_NULL(func_graph);
  for (auto &item : func_graph->parameters()) {
    MS_EXCEPTION_IF_NULL(item);
    auto param = item->cast<ParameterPtr>();
    if (param == nullptr) {
      MS_LOG(EXCEPTION) << "Parameter: '" << item->ToString() << "' could not cast to parameter.";
    }
    std::string param_name = GetUniqueNodeName(param);
    if (top_graph && param->has_default()) {
      MS_LOG(DEBUG) << "Parameter: '" << item->DebugString();
      mind_ir::TensorProto *parameter_proto = graph_proto->add_parameter();
      parameter_proto->set_name(param_name);
      SetParamToTensorProto(param, parameter_proto);
      auto tensor = std::dynamic_pointer_cast<tensor::Tensor>(param->default_param());
      if (tensor && save_tensor_data) {
        parameter_proto->set_raw_data(tensor->data_c(), static_cast<size_t>(tensor->data().nbytes()));
      }
    } else {
      mind_ir::ValueInfoProto *input_proto = graph_proto->add_input();
      input_proto->set_name(param_name);
      SetValueInfoProto(param, input_proto);
    }
    if (nodeName_.count(param_name) > 0) {
      MS_LOG(EXCEPTION) << "parameter name is duplicate:" << param_name;
    }
    (void)nodeName_.insert(param_name);
  }
}

mind_ir::TensorProto_DataType IrExportBuilder::GetMindirDataType(TypeId type_id) {
  auto iter = g_data_type_map.find(type_id);
  if (iter == g_data_type_map.end()) {
    MS_LOG(EXCEPTION) << "Convert type error, unsupported type! " << type_id;
  }
  return iter->second;
}

mind_ir::TensorProto_DataType IrExportBuilder::GetMindirDataBitsIntType(int bits) {
  auto iter = g_data_bits_int_map.find(bits);
  if (iter == g_data_bits_int_map.end()) {
    MS_LOG(EXCEPTION) << "Convert bits int error, unsupported bits! " << bits;
  }
  return iter->second;
}

mind_ir::TensorProto_DataType IrExportBuilder::GetMindirDataBitsUIntType(int bits) {
  auto iter = g_data_bits_uint_map.find(bits);
  if (iter == g_data_bits_uint_map.end()) {
    MS_LOG(EXCEPTION) << "Convert bits uint error, unsupported bits! " << bits;
  }
  return iter->second;
}

mind_ir::TensorProto_DataType IrExportBuilder::GetMindirDataBitsFloatType(int bits) {
  auto iter = g_data_bits_float_map.find(bits);
  if (iter == g_data_bits_float_map.end()) {
    MS_LOG(EXCEPTION) << "Convert bits float error, unsupported bits! " << bits;
  }
  return iter->second;
}

void IrExportBuilder::SetValueInfoProto(const AnfNodePtr &node, mind_ir::ValueInfoProto *const value_proto) {
  if (node == nullptr || value_proto == nullptr) {
    MS_LOG(EXCEPTION) << "AnfNode or ValueInfo is null!";
  }
  MS_LOG(DEBUG) << "SetValueInfoProto: " << node->DebugString();
  const TypePtr &type = node->Type();
  const BaseShapePtr &shape = node->Shape();
  if (type == nullptr || shape == nullptr) {
    return;
  }
  if (type->isa<TensorType>() && shape->isa<abstract::Shape>()) {
    auto tensor = type->cast<TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    auto elem_type = tensor->element();
    const auto &dims = shape->cast<abstract::ShapePtr>()->shape();
    mind_ir::TensorProto *tensor_proto = value_proto->add_tensor();
    tensor_proto->set_data_type(GetMindirDataType(elem_type->type_id()));
    if (dims.size() == 0) {
      MS_LOG(DEBUG) << "The dim of ValueInfoProto is 0.";
    } else {
      for (const auto &dim : dims) {
        MS_LOG(DEBUG) << "SetValueInfoProto dim: " << dim;
        tensor_proto->add_dims(dim);
      }
    }
  } else if (type->isa<Tuple>()) {
    auto tup_shape = shape->cast<abstract::TupleShapePtr>();
    value_proto->set_denotation(type->type_name() + ":" + std::to_string(tup_shape->shape().size()));
  } else {
    value_proto->set_denotation(type->type_name());
  }
  MS_LOG(DEBUG) << "Value type: " << type->type_name();
}

void IrExportBuilder::SetTensorToAttributeProto(const ValuePtr &value, mind_ir::AttributeProto *const attr_proto) {
  if (value == nullptr || attr_proto == nullptr) {
    MS_LOG(EXCEPTION) << "ValuePtr or AttributeProto is null!";
  }
  attr_proto->set_ref_attr_name("tensor:value0");
  attr_proto->set_type(mind_ir::AttributeProto_AttributeType_TENSORS);
  mind_ir::TensorProto *tensor_proto = attr_proto->add_tensors();
  tensor_proto->set_name("value0");
  auto data = value->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(data);
  tensor_proto->set_raw_data(data->data_c(), static_cast<size_t>(data->data().nbytes()));
  auto dtype = data->data_type();
  auto shape = data->shape_c();
  tensor_proto->set_data_type(GetMindirDataType(dtype));
  for (const auto &dim : shape) {
    tensor_proto->add_dims(dim);
  }
}

void IrExportBuilder::SetTensorProto(const TypePtr &type, const BaseShapePtr &shape,
                                     mind_ir::TensorProto *const tensor_proto) {
  if (!type->isa<TensorType>() || !shape->isa<abstract::Shape>()) {
    MS_LOG(EXCEPTION) << "Type or shape is not supported! " << type->ToString();
  }
  auto tensor = type->cast<TensorTypePtr>();
  const auto &dims = shape->cast<abstract::ShapePtr>()->shape();
  tensor_proto->set_data_type(GetMindirDataType(tensor->element()->type_id()));
  for (const auto &dim : dims) {
    tensor_proto->add_dims(dim);
  }
}

void IrExportBuilder::SetParamToTensorProto(const ParameterPtr &param, mind_ir::TensorProto *const tensor_proto) {
  if (param == nullptr || tensor_proto == nullptr) {
    MS_LOG(EXCEPTION) << "Parameter or TensorProto is null!";
  }
  MS_LOG(DEBUG) << "SetParamToTensorProto: " << param->DebugString();
  SetTensorProto(param->Type(), param->Shape(), tensor_proto);
}

void IrExportBuilder::BuildNodes(const FuncGraphPtr &func_graph, mind_ir::GraphProto *const graph_proto) {
  std::vector<AnfNodePtr> nodes = TopoSort(func_graph->get_return(), SuccIncoming, AlwaysInclude);
  for (const AnfNodePtr &node : nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      MS_LOG(DEBUG) << "Node: '" << node->ToString() << "' is not cnode";
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (cnode == func_graph->get_return()) {
      BuildOutput(cnode, graph_proto);
    } else {
      BuildCNode(cnode, graph_proto);
    }
  }
}

void IrExportBuilder::BuildOutput(const CNodePtr &node, mind_ir::GraphProto *const graph_proto) {
  MS_EXCEPTION_IF_NULL(node);
  const int OutputSize = 2;
  if (node->size() != OutputSize) {
    MS_LOG(EXCEPTION) << "Number of inputs of return node is not equal to 2.";
  }
  AnfNodePtr arg = node->input(1);
  std::string node_name = BuildInputNode(arg, graph_proto);
  mind_ir::ValueInfoProto *output_proto = graph_proto->add_output();
  output_proto->set_name(node_name);
  SetValueInfoProto(arg, output_proto);
}

std::string IrExportBuilder::GetOpTypeName(const AnfNodePtr &node) {
  // May be ValueNode/CNode/Parameter
  std::string type_name = "";
  if (IsValueNode<Primitive>(node)) {
    PrimitivePtr prim = GetValueNode<PrimitivePtr>(node);
    MS_EXCEPTION_IF_NULL(prim);
    type_name = prim->ToString();
  } else if (IsValueNode<FuncGraph>(node)) {
    FuncGraphPtr fg = GetValueNode<FuncGraphPtr>(node);
    MS_EXCEPTION_IF_NULL(fg);
    todo_.push_back(fg);
    type_name = "REF::" + fg->ToString();
  } else if (node->isa<CNode>() || node->isa<Parameter>()) {
    auto nodeName = GetUniqueNodeName(node);
    type_name = "REF::" + nodeName;
    if (nodeName_.count(nodeName) == 0) {
      MS_LOG(EXCEPTION) << "There is not the name: " << nodeName;
    }
  } else {
    MS_LOG(EXCEPTION) << "Need to support op type: " << node->type_name();
  }
  MS_LOG(DEBUG) << "ExportType: " << type_name;
  return type_name;
}

void IrExportBuilder::SetShapeToNodeProto(const TypePtr &type, const BaseShapePtr &shape,
                                          mind_ir::AttributeProto *const attr_proto, std::string *const seq_string) {
  MS_EXCEPTION_IF_NULL(type);
  MS_EXCEPTION_IF_NULL(shape);
  MS_EXCEPTION_IF_NULL(seq_string);
  if (type->isa<Tuple>()) {
    *seq_string += "Tuple[";
    auto elements = type->cast<TuplePtr>()->elements();
    auto tuple_shape = shape->cast<abstract::TupleShapePtr>()->shape();
    for (size_t i = 0; i < elements.size(); i++) {
      SetShapeToNodeProto(elements[i], tuple_shape[i], attr_proto, seq_string);
    }
    *seq_string += "],";
  } else if (type->isa<TensorType>() && shape->isa<abstract::Shape>()) {
    string shape_name = "shape" + std::to_string(GetTupleIndex());
    *seq_string += shape_name + ",";
    mind_ir::TensorProto *tensor_proto = attr_proto->add_tensors();
    tensor_proto->set_name(shape_name);
    SetTensorProto(type, shape, tensor_proto);
  } else if (type->isa<Number>()) {
    if (type->isa<Bool>()) {
      attr_proto->set_type(mind_ir::AttributeProto_AttributeType_BOOL);
    } else {
      string shape_name = "shape" + std::to_string(GetTupleIndex());
      *seq_string += shape_name + ",";
      mind_ir::TensorProto *tensor_proto = attr_proto->add_tensors();
      tensor_proto->set_name(shape_name);
      tensor_proto->set_data_type(mind_ir::TensorProto_DataType_UINT64);
      tensor_proto->add_dims(1);
    }
  } else if (type->isa<Function>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_GRAPH);
    *seq_string += type->type_name() + ",";
  } else if (type->isa<String>() || type->isa<UMonadType>() || type->isa<IOMonadType>()) {
    *seq_string += type->type_name() + ",";
  } else {
    MS_LOG(EXCEPTION) << "Type of cnode need to be supported: " << type->type_name();
  }
}

void IrExportBuilder::SetShapeToNodeProto(const CNodePtr &node, mind_ir::NodeProto *const node_proto) {
  // Get shape of cnode
  // 1. need to get shape from tuple element
  // 2. save shape in TensorProto
  // 3. save tuple string in ref_attr_name
  MS_EXCEPTION_IF_NULL(node);
  auto type = node->Type();
  auto shape = node->Shape();
  if (type == nullptr || shape == nullptr) {
    return;
  }
  ResetTupleIndex();
  std::string seq_string = "shape:";
  mind_ir::AttributeProto *attr_proto = node_proto->add_attribute();
  SetShapeToNodeProto(type, shape, attr_proto, &seq_string);
  attr_proto->set_ref_attr_name(seq_string);
  MS_LOG(DEBUG) << "CNode shape: " << seq_string;
}

void IrExportBuilder::BuildCNode(const CNodePtr &node, mind_ir::GraphProto *const graph_proto) {
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
  mind_ir::NodeProto *node_proto = graph_proto->add_node();
  std::string output_name = GetUniqueNodeName(node);
  if (nodeName_.count(output_name) > 0) {
    MS_LOG(EXCEPTION) << "There is a duplicate name: " << output_name;
  }
  (void)nodeName_.insert(output_name);
  node_proto->add_output(output_name);
  node_proto->set_name(output_name);
  node_proto->set_domain(node->fullname_with_scope());
  AnfNodePtr op = node->input(0);
  std::string type_name = GetOpTypeName(op);
  node_proto->set_op_type(type_name);
  last_node_ = node_proto;
  // Maybe Tensor or Function or nullptr
  SetShapeToNodeProto(node, node_proto);

  (void)std::for_each(input_names.begin(), input_names.end(),
                      [&node_proto](const string &name) { node_proto->add_input(name); });

  // Add primitive attrs
  if (IsValueNode<Primitive>(op)) {
    auto prim = GetValueNode<PrimitivePtr>(op);
    for (auto attr : prim->attrs()) {
      MS_LOG(DEBUG) << "attr: " << attr.first << " " << attr.second->DumpText() << " " << attr.second->type_name();
      mind_ir::AttributeProto *attr_proto = node_proto->add_attribute();
      attr_proto->set_name(attr.first);
      auto attr_value = attr.second;
      CheckAndConvertUtils::ConvertAttrValueInExport(type_name, attr.first, &attr_value);
      SetValueToAttributeProto(attr_value, attr_proto);
    }
  }
}

std::string IrExportBuilder::BuildInputNode(const AnfNodePtr &node, mind_ir::GraphProto *const graph_proto) {
  std::string node_name = GetUniqueNodeName(node);
  // FuncGraph will be added to functions and the input name is the function name.
  if (IsValueNode<FuncGraph>(node)) {
    FuncGraphPtr fg = GetValueNode<FuncGraphPtr>(node);
    todo_.push_back(fg);
    return fg->ToString();
  }
  if (node->isa<ValueNode>()) {
    // When node input is a ValueNode, need to create a Constant Node
    mind_ir::NodeProto *node_proto = graph_proto->add_node();
    node_proto->set_name(node_name);
    node_proto->add_output(node_name);
    SetAttributeProto(node, node_proto);
  }
  return node_name;
}

std::string IrExportBuilder::GetUniqueNodeName(const AnfNodePtr &node) {
  // Naming anfnode
  // 1. parameter is unique in one func_graph
  // 2. cnode and valuenode may be reduplicative, so add index to identify.
  auto iter = node_index_map_.find(node);
  if (iter != node_index_map_.end()) {
    return iter->second;
  } else {
    std::string node_name = GetNodeName(node);
    // Compatible before. CNode = FuncGraphName:CNodeName:index ,Parameter = FuncGraphName:ParameterName
    if (node->isa<CNode>()) {
      node_name = node_name + ":" + std::to_string(GetNodeIndex());
    }
    // Avoid duplicate name.
    while (nodeName_.count(node_name) > 0) {
      node_name = node_name + "_" + std::to_string(GetNodeIndex());
    }
    node_index_map_[node] = node_name;
    return node_name;
  }
}

std::string IrExportBuilder::GetNodeName(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  std::string node_name = "";
  if (node->func_graph() != nullptr) {
    node_name = node->func_graph()->ToString() + ":";
  }
  if (node->isa<ValueNode>()) {
    // Needn't value
    node_name += node->AnfNode::ToString();
  } else {
    node_name += node->ToString();
  }
  MS_LOG(DEBUG) << "GetNodeName: " << node_name;
  return node_name;
}

void IrExportBuilder::SetAttributeProto(const AnfNodePtr &node, mind_ir::NodeProto *const node_proto) {
  if (node == nullptr || node_proto == nullptr) {
    MS_LOG(EXCEPTION) << "AnfNode or NodeProto is null!";
  }
  auto value_node = node->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  auto value = value_node->value();
  node_proto->set_op_type("Constant");
  mind_ir::AttributeProto *attr_proto = node_proto->add_attribute();
  attr_proto->set_name("value");
  MS_LOG(DEBUG) << "Set Constant attribute: " << value->ToString();
  SetValueToAttributeProto(value, attr_proto);
}

void IrExportBuilder::SetTypeToAttributeProto(const ValuePtr &value, mind_ir::AttributeProto *const attr_proto) {
  if (value == nullptr || attr_proto == nullptr) {
    MS_LOG(EXCEPTION) << "ValuePtr or AttributeProto is null!";
  }
  attr_proto->set_type(mind_ir::AttributeProto_AttributeType_TENSORS);
  mind_ir::TensorProto *tensor_proto = attr_proto->add_tensors();
  if (value->isa<Int>()) {
    attr_proto->set_ref_attr_name("type:value0");
    tensor_proto->set_name("value0");
    auto int_value = value->cast<IntPtr>();
    tensor_proto->set_data_type(GetMindirDataBitsIntType(int_value->nbits()));
  } else if (value->isa<UInt>()) {
    attr_proto->set_ref_attr_name("type:value0");
    tensor_proto->set_name("value0");
    auto float_value = value->cast<UIntPtr>();
    tensor_proto->set_data_type(GetMindirDataBitsUIntType(float_value->nbits()));
  } else if (value->isa<Float>()) {
    attr_proto->set_ref_attr_name("type:value0");
    tensor_proto->set_name("value0");
    auto float_value = value->cast<FloatPtr>();
    tensor_proto->set_data_type(GetMindirDataBitsFloatType(float_value->nbits()));
  } else if (value->isa<Bool>()) {
    attr_proto->set_ref_attr_name("type:value0");
    tensor_proto->set_name("value0");
    tensor_proto->set_data_type(mind_ir::TensorProto_DataType_BOOL);
  } else if (value->isa<TensorType>()) {
    attr_proto->set_ref_attr_name("type:tensor0");
    tensor_proto->set_name("tensor0");
    auto elem_type = value->cast<TensorTypePtr>()->element();
    if (elem_type->isa<Int>()) {
      auto int_value = elem_type->cast<IntPtr>();
      tensor_proto->set_data_type(GetMindirDataBitsIntType(int_value->nbits()));
    } else if (elem_type->isa<Float>()) {
      auto float_value = elem_type->cast<FloatPtr>();
      tensor_proto->set_data_type(GetMindirDataBitsFloatType(float_value->nbits()));
    } else {
      MS_LOG(EXCEPTION) << "Unsupported type " << elem_type->type_name();
    }
  } else {
    MS_LOG(EXCEPTION) << "Unsupported type: " << value->type_name();
  }
}

void IrExportBuilder::SetValueToAttributeProto(const ValuePtr &value, mind_ir::AttributeProto *const attr_proto) {
  if (value == nullptr || attr_proto == nullptr) {
    MS_LOG(EXCEPTION) << "ValuePtr or AttributeProto is null!";
  }
  if (value->isa<StringImm>() || value->isa<Scalar>()) {
    SetScalarToAttributeProto_ir(value, attr_proto);
  } else if (value->isa<Number>() || value->isa<TensorType>()) {
    SetTypeToAttributeProto(value, attr_proto);
  } else if (value->isa<ValueSequeue>()) {
    ResetTupleIndex();
    std::string seq_string = "scalar:";
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_TENSORS);
    SetSequenceToAttributeProto(value->cast<ValueSequeuePtr>(), attr_proto, &seq_string);
    attr_proto->set_ref_attr_name(seq_string);
    MS_LOG(DEBUG) << "Attr string: " << seq_string;
  } else if (value->isa<tensor::Tensor>()) {
    SetTensorToAttributeProto(value, attr_proto);
  } else if (value->isa<None>()) {
    attr_proto->set_ref_attr_name("none");
    MS_LOG(DEBUG) << "Attr string: " << value->type_name();
  } else if (value->isa<Monad>()) {
    if (value->isa<UMonad>()) {
      attr_proto->set_ref_attr_name("Monad:UMonad");
    } else if (value->isa<IOMonad>()) {
      attr_proto->set_ref_attr_name("Monad:IOMonad");
    } else {
      MS_LOG(EXCEPTION) << "Unsupported Monad type: " << value->type_name();
    }
  } else {
    MS_LOG(EXCEPTION) << "Unsupported type: " << value->type_name();
  }
}

void IrExportBuilder::SetScalarToAttributeProto_ir(const ValuePtr &value, mind_ir::AttributeProto *const attr_proto) {
  if (value == nullptr || attr_proto == nullptr) {
    MS_LOG(EXCEPTION) << "ValuePtr or AttributeProto is null!";
  }
  attr_proto->set_ref_attr_name("scalar:value0");
  if (value->isa<StringImm>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_STRING);
    attr_proto->set_s(GetValue<std::string>(value));
  } else if (value->isa<BoolImm>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_BOOL);
    int64_t attr_value = GetValue<bool>(value) ? 1 : 0;
    attr_proto->set_i(attr_value);
  } else if (value->isa<Int8Imm>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_INT8);
    attr_proto->set_i(value->cast<Int8ImmPtr>()->value());
  } else if (value->isa<Int16Imm>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_INT16);
    attr_proto->set_i(value->cast<Int16ImmPtr>()->value());
  } else if (value->isa<Int32Imm>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_INT32);
    attr_proto->set_i(value->cast<Int32ImmPtr>()->value());
  } else if (value->isa<Int64Imm>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_INT64);
    attr_proto->set_i(value->cast<Int64ImmPtr>()->value());
  } else if (value->isa<UInt8Imm>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_UINT8);
    attr_proto->set_i(value->cast<UInt8ImmPtr>()->value());
  } else if (value->isa<UInt16Imm>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_UINT16);
    attr_proto->set_i(value->cast<UInt16ImmPtr>()->value());
  } else if (value->isa<UInt32Imm>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_UINT32);
    attr_proto->set_i(value->cast<UInt32ImmPtr>()->value());
  } else if (value->isa<UInt64Imm>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_UINT64);
    attr_proto->set_i(UlongToLong(value->cast<UInt64ImmPtr>()->value()));
  } else if (value->isa<FP32Imm>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_FLOAT);
    attr_proto->set_f(GetValue<float>(value));
  } else if (value->isa<FP64Imm>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_DOUBLE);
    attr_proto->set_d(GetValue<double>(value));
  } else if (value->isa<tensor::Tensor>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_TENSOR);
    SetTensorToAttributeProto(value, attr_proto);
  } else {
    MS_LOG(EXCEPTION) << "Unsupported scalar type: " << value->type_name();
  }
}

void IrExportBuilder::SetScalarToAttributeProto_irs(const ValuePtr &value, mind_ir::AttributeProto *const attr_proto) {
  if (value == nullptr || attr_proto == nullptr) {
    MS_LOG(EXCEPTION) << "ValuePtr or AttributeProto is null!";
  }
  if (value->isa<Int>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_TENSORS);
    mind_ir::TensorProto *tensor_proto = attr_proto->add_tensors();
    auto int_value = value->cast<IntPtr>();
    tensor_proto->set_data_type(GetMindirDataBitsIntType(int_value->nbits()));
  } else if (value->isa<Float>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_TENSORS);
    mind_ir::TensorProto *tensor_proto = attr_proto->add_tensors();
    auto float_value = value->cast<FloatPtr>();
    tensor_proto->set_data_type(GetMindirDataBitsFloatType(float_value->nbits()));
  } else if (value->isa<StringImm>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_STRING);
    attr_proto->add_strings(GetValue<std::string>(value));
  } else if (value->isa<BoolImm>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_BOOL);
    int attr_value = GetValue<bool>(value) ? 1 : 0;
    attr_proto->add_ints(attr_value);
  } else if (value->isa<Int8Imm>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_INT8);
    attr_proto->add_ints(value->cast<Int8ImmPtr>()->value());
  } else if (value->isa<Int16Imm>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_INT16);
    attr_proto->add_ints(value->cast<Int16ImmPtr>()->value());
  } else if (value->isa<Int32Imm>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_INT32);
    attr_proto->add_ints(value->cast<Int32ImmPtr>()->value());
  } else if (value->isa<Int64Imm>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_INT64);
    attr_proto->add_ints(value->cast<Int64ImmPtr>()->value());
  } else if (value->isa<UInt8Imm>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_UINT8);
    attr_proto->add_ints(value->cast<UInt8ImmPtr>()->value());
  } else if (value->isa<UInt16Imm>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_UINT16);
    attr_proto->add_ints(value->cast<UInt16ImmPtr>()->value());
  } else if (value->isa<UInt32Imm>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_UINT32);
    attr_proto->add_ints(value->cast<UInt32ImmPtr>()->value());
  } else if (value->isa<UInt64Imm>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_UINT64);
    attr_proto->add_ints(SizeToInt(value->cast<UInt64ImmPtr>()->value()));
  } else if (value->isa<FP32Imm>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_FLOAT);
    attr_proto->add_floats(GetValue<float>(value));
  } else if (value->isa<FP64Imm>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_DOUBLE);
    attr_proto->add_doubles(GetValue<double>(value));
  } else if (value->isa<tensor::Tensor>()) {
    attr_proto->set_type(mind_ir::AttributeProto_AttributeType_TENSOR);
    SetTensorToAttributeProto(value, attr_proto);
  } else {
    MS_LOG(EXCEPTION) << "Unsupported scalar type: " << value->type_name();
  }
}

void IrExportBuilder::SetSeqElemToAttributeProto(const ValuePtr &value, mind_ir::AttributeProto *const attr_proto,
                                                 std::string *const seq_string) {
  string value_name = "value" + std::to_string(GetTupleIndex());
  if (seq_string != nullptr) {
    *seq_string += value_name + ",";
  }
  SetScalarToAttributeProto_irs(value, attr_proto);
}

void IrExportBuilder::SetSequenceToAttributeProto(const ValueSequeuePtr &value,
                                                  mind_ir::AttributeProto *const attr_proto,
                                                  std::string *const seq_string) {
  if (value == nullptr || attr_proto == nullptr) {
    MS_LOG(EXCEPTION) << "ValueSequeuePtr or AttributeProto is null!";
  }
  if (value->isa<ValueTuple>() && seq_string != nullptr) {
    *seq_string += "Tuple[";
    const ValueTuplePtr &tuple_value = value->cast<ValueTuplePtr>();
    if (tuple_value->value().size() == 0) {
      *seq_string += "],";
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
      *seq_string += "],";
      MS_LOG(DEBUG) << "SetSequenceToAttributeProto list size is 0.";
      return;
    }
    for (const auto &item : list_value->value()) {
      MS_EXCEPTION_IF_NULL(item);
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

mind_ir::ModelProto GetBinaryProto(const FuncGraphPtr &func_graph, bool save_tensor_data) {
  auto exporter = std::make_shared<IrExporter>(std::make_shared<IrExportBuilder>());
  auto result = exporter->GetDumpProto(func_graph, save_tensor_data);
  return result;
}
}  // namespace mindspore

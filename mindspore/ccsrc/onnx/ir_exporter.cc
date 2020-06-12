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

#include "ir/param_value_py.h"
#include "debug/anf_ir_utils.h"
#include "operator/ops.h"
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
  void SetShapeToNodeProto(const CNodePtr &node, const std::vector<AnfNodePtr> &inputs,
                           onnx::NodeProto *const node_proto);
  void SetShapeToNodeProto(const TypePtr &type, const BaseShapePtr &shape, onnx::NodeProto *const node_proto);
  void SetValueToAttributeProto(const ValuePtr &value, onnx::AttributeProto *const attr_proto);
  void SetTypeToAttributeProto(const ValuePtr &value, onnx::AttributeProto *const attr_proto);
  void SetScalarToAttributeProto(const ValuePtr &value, onnx::AttributeProto *const attr_proto);
  void SetTensorToAttributeProto(const ValuePtr &value, onnx::AttributeProto *const attr_proto);
  void SetScalarToProto(const ValuePtr &value, onnx::TensorProto *const tensor_proto);
  void SetSequenceToAttributeProto(const ValueSequeuePtr &value, onnx::AttributeProto *const attr_proto);

  onnx::TensorProto_DataType GetOnnxDataType(TypeId type_id);
  onnx::TensorProto_DataType GetOnnxDataBitsIntType(int bits);
  onnx::TensorProto_DataType GetOnnxDataBitsFloatType(int bits);
  std::string GetNodeName(const AnfNodePtr &node);
  std::string GetUniqueNodeName(const AnfNodePtr &node);
  std::string GetOpTypeName(const AnfNodePtr &node);
  size_t AllocateIndex() { return ++node_index_; }
  void ResetIndex() { node_index_ = 0; }

 private:
  onnx::ModelProto model_;
  onnx::NodeProto *last_node_;
  std::list<FuncGraphPtr> todo_;
  std::map<AnfNodePtr, size_t> node_index_map_;
  size_t node_index_ = 0;
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
  ResetIndex();
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
      MS_LOG(DEBUG) << "Parameter: '" << item->ToString() << "' has no default";
      continue;
    }

    // Using ONNX initializer to set parameter's default value
    onnx::TensorProto *initializer_proto = graph_proto->add_initializer();
    initializer_proto->set_name(param_name);
    SetParamToTensorProto(param, initializer_proto);
    auto param_value = std::dynamic_pointer_cast<ParamValuePy>(param->default_param());
    py::object obj = param_value->value();
    py::object data = obj.attr("data");
    if (py::isinstance<tensor::Tensor>(data)) {
      auto method = data.attr("asnumpy");
      py::array npy_data = method();
      initializer_proto->set_raw_data(npy_data.request(true).ptr, static_cast<size_t>(npy_data.nbytes()));
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
    for (const auto &dim : dims) {
      MS_LOG(DEBUG) << "SetValueInfoProto dim: " << dim;
      type_proto->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dim);
    }
  } else if (type->isa<Tuple>()) {
    auto tup_shape = shape->cast<abstract::TupleShapePtr>();
    type_proto->set_denotation(std::to_string(tup_shape->shape().size()));
  } else {
    MS_LOG(EXCEPTION) << "Value type: " << type->type_name() << " is not supported!";
  }
}

void IrExportBuilder::SetTensorToAttributeProto(const ValuePtr &value, onnx::AttributeProto *const attr_proto) {
  if (value == nullptr || attr_proto == nullptr) {
    MS_LOG(EXCEPTION) << "ValuePtr or AttributeProto is null!";
  }
  attr_proto->set_ref_attr_name("tensor");
  attr_proto->set_type(onnx::AttributeProto_AttributeType_TENSOR);
  onnx::TensorProto *tensor_proto = attr_proto->mutable_t();
  auto data = value->cast<tensor::TensorPtr>();
  tensor_proto->set_raw_data(data->data().request(true).ptr, static_cast<size_t>(data->data().nbytes()));
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
  for (const AnfNodePtr &node : nodes) {
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

void IrExportBuilder::BuildOutput(const CNodePtr &node, onnx::GraphProto *const graph_proto) {
  if (node->size() != 2) {
    MS_LOG(EXCEPTION) << "Number of inputs of return node is not equal to 2.";
  }
  AnfNodePtr arg = node->input(1);
  // Using make_tuple to set multi-output
  if (IsPrimitiveCNode(arg, prim::kPrimMakeTuple)) {
    auto tuple_node = arg->cast<CNodePtr>();
    for (size_t i = 1; i < tuple_node->size(); i++) {
      auto input_node = arg->cast<CNodePtr>()->input(i);
      onnx::ValueInfoProto *output_proto = graph_proto->add_output();
      auto output_name = GetUniqueNodeName(tuple_node->input(i));
      output_proto->set_name(output_name);
      last_node_->add_output(output_name);
      SetValueInfoProto(tuple_node->input(i), output_proto);
    }
  } else {
    onnx::ValueInfoProto *output_proto = graph_proto->add_output();
    std::string output_name = GetUniqueNodeName(node);
    output_proto->set_name(output_name);
    last_node_->add_output(output_name);
    SetValueInfoProto(arg, output_proto);
  }
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
                                          onnx::NodeProto *const node_proto) {
  onnx::AttributeProto *attr_proto = node_proto->add_attribute();
  attr_proto->set_ref_attr_name("shape");
  attr_proto->set_name("shape");
  onnx::TensorProto *tensor_proto = attr_proto->mutable_t();
  SetTensorProto(type, shape, tensor_proto);
}

void IrExportBuilder::SetShapeToNodeProto(const CNodePtr &node, const std::vector<AnfNodePtr> &inputs,
                                          onnx::NodeProto *const node_proto) {
  // Get shape of cnode
  // 1. prim kPrimTupleGetItem need to get shape of input node according to the index
  // 2. some cnode doesn't has shape, such as LayerNorm
  // 3. other cnodes have shape
  if (node->IsApply(prim::kPrimTupleGetItem)) {
    // Get index of tuple get_item
    int index_pos = inputs.size() - 1;
    if (!inputs[index_pos]->isa<ValueNode>()) {
      MS_LOG(EXCEPTION) << "Index is not ValueNode: " << index_pos;
    }
    auto value = inputs[index_pos]->cast<ValueNodePtr>()->value();
    if (!value->isa<IntergerImm>()) {
      MS_LOG(EXCEPTION) << "Index type is not supported: " << value->type_name();
    }
    size_t index = GetValue<int>(value);

    // Get type and shape of input node
    auto tup_type = inputs[0]->Type();
    if (!tup_type->isa<Tuple>()) {
      MS_LOG(EXCEPTION) << "Input data of kPrimTupleGetItem cnode must be tuple: " << tup_type->type_name();
    }
    auto type = tup_type->cast<TuplePtr>()->elements()[index];
    auto tup_shape = inputs[0]->Shape()->cast<abstract::TupleShapePtr>();
    if (index >= tup_shape->shape().size()) {
      MS_LOG(EXCEPTION) << "Index exceed upper limit: " << tup_shape->shape().size();
    }
    auto shape = tup_shape->shape()[index];
    SetShapeToNodeProto(type, shape, node_proto);
  } else {
    auto type = node->Type();
    auto shape = node->Shape();
    if (!type->isa<TensorType>() || !shape->isa<abstract::Shape>()) {
      MS_LOG(DEBUG) << "Cnode has no shape: " << node->ToString();
      return;
    }
    SetShapeToNodeProto(type, shape, node_proto);
  }
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
  AnfNodePtr op = node->input(0);
  std::string type_name = GetOpTypeName(op);
  node_proto->set_op_type(type_name);
  last_node_ = node_proto;
  SetShapeToNodeProto(node, op_inputs, node_proto);
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
  } else if (node->isa<CNode>() || node->isa<ValueNode>()) {
    auto iter = node_index_map_.find(node);
    if (iter != node_index_map_.end()) {
      node_name = GetNodeName(node) + ":" + std::to_string(iter->second);
    } else {
      auto node_idx = AllocateIndex();
      node_index_map_[node] = node_idx;
      node_name = GetNodeName(node) + ":" + std::to_string(node_idx);
    }
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
  attr_proto->set_ref_attr_name("type");
  attr_proto->set_type(onnx::AttributeProto_AttributeType_TENSOR);
  onnx::TensorProto *tensor_proto = attr_proto->mutable_t();
  if (value->isa<Int>()) {
    auto int_value = value->cast<IntPtr>();
    tensor_proto->set_data_type(GetOnnxDataBitsIntType(int_value->nbits()));
  } else if (value->isa<Float>()) {
    auto float_value = value->cast<FloatPtr>();
    tensor_proto->set_data_type(GetOnnxDataBitsFloatType(float_value->nbits()));
  } else if (value->isa<TensorType>()) {
    tensor_proto->set_name("tensor");
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
  } else if (value->isa<ValueSequeue>()) {
    SetSequenceToAttributeProto(value->cast<ValueSequeuePtr>(), attr_proto);
  } else if (value->isa<tensor::Tensor>()) {
    SetTensorToAttributeProto(value, attr_proto);
  } else {
    MS_LOG(EXCEPTION) << "Unsupported type: " << value->type_name();
  }
}

void IrExportBuilder::SetScalarToAttributeProto(const ValuePtr &value, onnx::AttributeProto *const attr_proto) {
  if (value == nullptr || attr_proto == nullptr) {
    MS_LOG(EXCEPTION) << "ValuePtr or AttributeProto is null!";
  }
  attr_proto->set_ref_attr_name("scalar");
  attr_proto->set_type(onnx::AttributeProto_AttributeType_TENSOR);
  onnx::TensorProto *tensor_proto = attr_proto->mutable_t();
  SetScalarToProto(value, tensor_proto);
}

void IrExportBuilder::SetScalarToProto(const ValuePtr &value, onnx::TensorProto *const tensor_proto) {
  if (value == nullptr || tensor_proto == nullptr) {
    MS_LOG(EXCEPTION) << "ValuePtr or TensorProto is null!";
  }
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
  } else if (value->isa<FloatImm>()) {
    tensor_proto->set_data_type(onnx::TensorProto_DataType_FLOAT);
    tensor_proto->add_float_data(GetValue<float>(value));
  } else {
    MS_LOG(EXCEPTION) << "Unsupported scalar type: " << value->type_name();
  }
}

void IrExportBuilder::SetSequenceToAttributeProto(const ValueSequeuePtr &value,
                                                  onnx::AttributeProto *const attr_proto) {
  if (value == nullptr || attr_proto == nullptr) {
    MS_LOG(EXCEPTION) << "ValueSequeuePtr or AttributeProto is null!";
  }
  attr_proto->set_ref_attr_name("scalar");
  attr_proto->set_type(onnx::AttributeProto_AttributeType_TENSOR);
  onnx::TensorProto *tensor_proto = attr_proto->mutable_t();
  if (value->isa<ValueTuple>()) {
    const ValueTuplePtr &tuple_value = value->cast<ValueTuplePtr>();
    if (tuple_value->value().size() == 0) {
      MS_LOG(DEBUG) << "SetSequenceToAttributeProto tuple size is 0";
      return;
    }
    auto type_id = tuple_value->value()[0]->type()->type_id();
    tensor_proto->set_data_type(GetOnnxDataType(type_id));
    for (const auto &item : tuple_value->value()) {
      SetScalarToProto(item, tensor_proto);
    }
  } else if (value->isa<ValueList>()) {
    const ValueListPtr &list_value = value->cast<ValueListPtr>();
    if (list_value->value().size() == 0) {
      MS_LOG(DEBUG) << "SetSequenceToAttributeProto list size is 0";
      return;
    }
    auto type_id = list_value->value()[0]->type()->type_id();
    tensor_proto->set_data_type(GetOnnxDataType(type_id));
    for (const auto &item : list_value->value()) {
      SetScalarToProto(item, tensor_proto);
    }
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

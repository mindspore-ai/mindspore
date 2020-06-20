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

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "utils/load_onnx/anf_model_parser.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "ir/tensor.h"
#include "ir/param_value_py.h"
#include "operator/ops.h"
#include "proto/onnx.pb.h"
#include "utils/log_adapter.h"

using std::string;

namespace mindspore {
namespace lite {
static constexpr char kConstantValueNode[] = "Constant";
static constexpr char kCNodeShapeAttr[] = "shape";
enum ParseForm : int {
  FORM_PARSE_TYPE = 0,
  FORM_PARSE_SCALAR = 1,
  FORM_PARSE_TENSOR = 2,
};

static std::map<std::string, ParseForm> kParseTypeSwitchMap{
  {"type", FORM_PARSE_TYPE}, {"scalar", FORM_PARSE_SCALAR}, {"tensor", FORM_PARSE_TENSOR}};

static std::unordered_map<int, TypeId> kDefaultValueSwitchMap{
  {onnx::TensorProto_DataType_BOOL, kNumberTypeBool},     {onnx::TensorProto_DataType_INT8, kNumberTypeInt8},
  {onnx::TensorProto_DataType_INT16, kNumberTypeInt16},   {onnx::TensorProto_DataType_INT32, kNumberTypeInt32},
  {onnx::TensorProto_DataType_INT64, kNumberTypeInt64},   {onnx::TensorProto_DataType_UINT8, kNumberTypeUInt8},
  {onnx::TensorProto_DataType_UINT16, kNumberTypeUInt16}, {onnx::TensorProto_DataType_UINT32, kNumberTypeUInt32},
  {onnx::TensorProto_DataType_UINT64, kNumberTypeUInt64}, {onnx::TensorProto_DataType_FLOAT16, kNumberTypeFloat16},
  {onnx::TensorProto_DataType_FLOAT, kNumberTypeFloat32}, {onnx::TensorProto_DataType_DOUBLE, kNumberTypeFloat64},
  {onnx::TensorProto_DataType_STRING, kObjectTypeString},
};

#define PARSE_ONNXATTR_IN_SCALAR_FORM(type, valuetype)                                                \
  void ParseAttrInScalar_##type##_##valuetype(const PrimitivePtr &prim, const std::string &attr_name, \
                                              const onnx::TensorProto &attr_tensor) {                 \
    MS_EXCEPTION_IF_NULL(prim);                                                                       \
    std::vector<valuetype> attr_value_vec;                                                            \
    for (int i = 0; i < attr_tensor.type##_data_size(); ++i) {                                        \
      attr_value_vec.push_back(static_cast<valuetype>(attr_tensor.type##_data(i)));                   \
    }                                                                                                 \
    if (attr_value_vec.size() == 1) {                                                                 \
      prim->AddAttr(attr_name, MakeValue<valuetype>(attr_value_vec[0]));                              \
    } else {                                                                                          \
      prim->AddAttr(attr_name, MakeValue<std::vector<valuetype>>(attr_value_vec));                    \
    }                                                                                                 \
  }

PARSE_ONNXATTR_IN_SCALAR_FORM(double, double)
PARSE_ONNXATTR_IN_SCALAR_FORM(float, float)
PARSE_ONNXATTR_IN_SCALAR_FORM(string, string)
PARSE_ONNXATTR_IN_SCALAR_FORM(int32, int32)
PARSE_ONNXATTR_IN_SCALAR_FORM(int32, bool)
PARSE_ONNXATTR_IN_SCALAR_FORM(int64, int64)
PARSE_ONNXATTR_IN_SCALAR_FORM(uint64, uint64)

bool MSANFModelParser::BuildParameterForFuncGraph(const ParameterPtr &node, const onnx::ValueInfoProto &value_proto) {
  MS_EXCEPTION_IF_NULL(node);
  if (!value_proto.has_type() || !value_proto.has_name()) {
    MS_LOG(ERROR) << "onnx ValueInfoProto has no type or name! ";
    return false;
  }
  node->set_name(value_proto.name());
  const auto &type_proto = value_proto.type();
  if (!type_proto.has_tensor_type()) {
    MS_LOG(ERROR) << "onnx TypeProto has no tesor_type! ";
    return false;
  }
  const onnx::TypeProto_Tensor &tensor_typeproto = type_proto.tensor_type();
  if (!tensor_typeproto.has_elem_type() || !tensor_typeproto.has_shape()) {
    MS_LOG(ERROR) << "onnx TypeProto_Tensor has no elem_type or shape! ";
    return false;
  }
  const onnx::TensorShapeProto &tensor_shape = tensor_typeproto.shape();
  std::vector<int> shape;
  for (int i = 0; i < tensor_shape.dim_size(); ++i) {
    shape.push_back(tensor_shape.dim(i).dim_value());
  }

  if (kDefaultValueSwitchMap.find(tensor_typeproto.elem_type()) == kDefaultValueSwitchMap.end()) {
    MS_LOG(ERROR) << "onnx TypeProto_Tensor  elem_type is not support yet!";
    return false;
  }

  tensor::TensorPtr tensor_info =
    std::make_shared<tensor::Tensor>(kDefaultValueSwitchMap[tensor_typeproto.elem_type()], shape);
  MS_EXCEPTION_IF_NULL(tensor_info);
  auto tensor_abstract = tensor_info->ToAbstract();
  MS_EXCEPTION_IF_NULL(tensor_abstract);
  node->set_abstract(tensor_abstract);

  if (default_para_map_.find(value_proto.name()) != default_para_map_.end()) {
    const onnx::TensorProto initialize_proto = default_para_map_[value_proto.name()];
    std::string initial_data = initialize_proto.raw_data();
    auto *tensor_data_buf = reinterpret_cast<uint8_t *>(tensor_info->data_c(true));
    MS_EXCEPTION_IF_NULL(tensor_data_buf);
    memcpy_s(tensor_data_buf, tensor_info->data().nbytes(), initial_data.data(), initial_data.size());

    py::array array_data = tensor_info->data();
    ParamValuePyPtr para_value_ptr = std::make_shared<ParamValuePy>();
    MS_EXCEPTION_IF_NULL(para_value_ptr);
    para_value_ptr->set_value(array_data);
    node->set_default_param(para_value_ptr);
  }
  anfnode_build_map_[value_proto.name()] = node;
  return true;
}

bool MSANFModelParser::ImportParametersForGraph(const FuncGraphPtr &outputFuncGraph,
                                                const onnx::GraphProto &importProto) {
  MS_EXCEPTION_IF_NULL(outputFuncGraph);
  MS_LOG(INFO) << "Parameters had default paramerer size is: " << importProto.initializer_size();

  for (int i = 0; i < importProto.initializer_size(); ++i) {
    const onnx::TensorProto &initializer_proto = importProto.initializer(i);
    if (!initializer_proto.has_name()) {
      MS_LOG(ERROR) << "initializer vector of onnx GraphProto has no name at index: " << i;
      return false;
    }
    default_para_map_[initializer_proto.name()] = initializer_proto;
  }

  MS_LOG(INFO) << "all parameters size: " << importProto.input_size();
  for (int i = 0; i < importProto.input_size(); ++i) {
    const onnx::ValueInfoProto &input_proto = importProto.input(i);
    if (!BuildParameterForFuncGraph(outputFuncGraph->add_parameter(), input_proto)) {
      MS_LOG(ERROR) << "Build parameter for funcgraph fail at index: " << i;
      return false;
    }
  }
  return true;
}

bool MSANFModelParser::ObtainCNodeAttrInTypeForm(const PrimitivePtr &prim, const std::string &attr_name,
                                                 const onnx::TensorProto &attr_tensor) {
  MS_EXCEPTION_IF_NULL(prim);
  const int attr_tensor_type = attr_tensor.data_type();
  if (kDefaultValueSwitchMap.find(attr_tensor_type) == kDefaultValueSwitchMap.end()) {
    MS_LOG(ERROR) << "Obtain attr in type-form has not support input type:" << attr_tensor_type;
    return false;
  }
  prim->AddAttr(attr_name, TypeIdToType(kDefaultValueSwitchMap[attr_tensor_type]));
  return true;
}

bool MSANFModelParser::ObtainCNodeAttrInScalarForm(const PrimitivePtr &prim, const std::string &attr_name,
                                                   const onnx::TensorProto &attr_tensor) {
  MS_EXCEPTION_IF_NULL(prim);
  const int attr_tensor_type = attr_tensor.data_type();
  switch (attr_tensor_type) {
    case onnx::TensorProto_DataType_STRING: {
      ParseAttrInScalar_string_string(prim, attr_name, attr_tensor);
      break;
    }
    case onnx::TensorProto_DataType_INT32: {
      ParseAttrInScalar_int32_int32(prim, attr_name, attr_tensor);
      break;
    }
    case onnx::TensorProto_DataType_INT64: {
      ParseAttrInScalar_int64_int64(prim, attr_name, attr_tensor);
      break;
    }
    case onnx::TensorProto_DataType_UINT64: {
      ParseAttrInScalar_uint64_uint64(prim, attr_name, attr_tensor);
      break;
    }
    case onnx::TensorProto_DataType_FLOAT: {
      ParseAttrInScalar_float_float(prim, attr_name, attr_tensor);
      break;
    }
    case onnx::TensorProto_DataType_DOUBLE: {
      ParseAttrInScalar_double_double(prim, attr_name, attr_tensor);
      break;
    }
    case onnx::TensorProto_DataType_BOOL: {
      ParseAttrInScalar_int32_bool(prim, attr_name, attr_tensor);
      auto value = prim->GetAttr(attr_name);
      break;
    }
    default:
      MS_LOG(ERROR) << "Obtain attr in scalar-form has not support input type: " << attr_tensor_type;
      return false;
  }
  return true;
}

bool MSANFModelParser::ObtainCNodeAttrInTensorForm(const PrimitivePtr &prim, const std::string &attr_name,
                                                   const onnx::TensorProto &attr_tensor) {
  MS_EXCEPTION_IF_NULL(prim);
  MS_LOG(ERROR) << "parse attr type don't support attr type is tensor";
  return false;
}

bool MSANFModelParser::GetAttrValueForCNode(const PrimitivePtr &prim, const onnx::AttributeProto &attr_proto) {
  MS_EXCEPTION_IF_NULL(prim);
  const std::string &attr_name = attr_proto.name();
  if (!attr_proto.has_ref_attr_name()) {
    MS_LOG(ERROR) << "CNode parse attr type has no ref_attr_name";
    return false;
  }
  const std::string &ref_attr_name = attr_proto.ref_attr_name();
  const onnx::TensorProto &attr_tensor = attr_proto.t();
  switch (kParseTypeSwitchMap[ref_attr_name]) {
    case FORM_PARSE_TYPE: {
      return ObtainCNodeAttrInTypeForm(prim, attr_name, attr_tensor);
    }
    case FORM_PARSE_SCALAR: {
      return ObtainCNodeAttrInScalarForm(prim, attr_name, attr_tensor);
    }
    case FORM_PARSE_TENSOR: {
      return ObtainCNodeAttrInTensorForm(prim, attr_name, attr_tensor);
    }
    default:
      MS_LOG(ERROR) << "parse attr type don't support input of ref_attr_name";
      return false;
  }
}
bool MSANFModelParser::ObtainValueNodeInTensorForm(const std::string &value_node_name,
                                                   const onnx::TensorProto &attr_tensor) {
  const int attr_tensor_type = attr_tensor.data_type();
  std::vector<int> shape;
  for (int i = 0; i < attr_tensor.dims_size(); ++i) {
    shape.push_back(attr_tensor.dims(i));
  }
  tensor::TensorPtr tensor_info = std::make_shared<tensor::Tensor>(kDefaultValueSwitchMap[attr_tensor_type], shape);
  const std::string &tensor_buf = attr_tensor.raw_data();
  auto *tensor_data_buf = reinterpret_cast<uint8_t *>(tensor_info->data_c(true));
  memcpy_s(tensor_data_buf, tensor_info->data().nbytes(), tensor_buf.data(), tensor_buf.size());
  if (attr_tensor_type == onnx::TensorProto_DataType_FLOAT) {
    auto *data_valuennode = reinterpret_cast<float *>(tensor_info->data_c());
    MS_EXCEPTION_IF_NULL(data_valuennode);
    auto new_value_node = std::make_shared<ValueNode>(MakeValue(*data_valuennode));
    anfnode_build_map_[value_node_name] = new_value_node;
  } else {
    auto *data_valuenode = reinterpret_cast<int32 *>(tensor_info->data_c());
    MS_EXCEPTION_IF_NULL(data_valuenode);
    auto new_value_node = std::make_shared<ValueNode>(MakeValue(*data_valuenode));
    anfnode_build_map_[value_node_name] = new_value_node;
  }
  return true;
}

bool MSANFModelParser::ObtainValueNodeInScalarForm(const std::string &value_node_name,
                                                   const onnx::TensorProto &attr_tensor) {
  const int attr_tensor_type = attr_tensor.data_type();
  ValuePtr value_ptr = nullptr;
  switch (attr_tensor_type) {
    case onnx::TensorProto_DataType_INT32: {
      std::vector<int32> add_data;
      for (int i = 0; i < attr_tensor.int32_data_size(); ++i) {
        add_data.push_back(attr_tensor.int32_data(i));
      }
      if (add_data.size() == 1) {
        value_ptr = MakeValue(add_data[0]);
      } else if (!add_data.empty()) {
        value_ptr = MakeValue<std::vector<int32>>(add_data);
      }
      break;
    }
    case onnx::TensorProto_DataType_FLOAT: {
      std::vector<float> add_data;
      for (int i = 0; i < attr_tensor.float_data_size(); ++i) {
        add_data.push_back(attr_tensor.float_data(i));
      }

      if (add_data.size() == 1) {
        value_ptr = MakeValue(add_data[0]);
      } else if (!add_data.empty()) {
        value_ptr = MakeValue<std::vector<float>>(add_data);
      }
      break;
    }
    case onnx::TensorProto_DataType_UNDEFINED: {
      std::vector<ValuePtr> elems;
      value_ptr = std::make_shared<ValueTuple>(elems);
      break;
    }
    default:
      MS_LOG(ERROR) << "Obtain attr in scalar-form has not support input type: " << attr_tensor_type;
      return false;
  }
  auto new_value_node = NewValueNode(value_ptr);
  MS_EXCEPTION_IF_NULL(new_value_node);
  new_value_node->set_abstract(value_ptr->ToAbstract());
  anfnode_build_map_[value_node_name] = new_value_node;

  return true;
}

bool MSANFModelParser::ObtainValueNodeInTypeForm(const std::string &value_node_name,
                                                 const onnx::TensorProto &attr_tensor) {
  const int attr_tensor_type = attr_tensor.data_type();
  if (kDefaultValueSwitchMap.find(attr_tensor_type) == kDefaultValueSwitchMap.end()) {
    MS_LOG(ERROR) << "Obtain ValueNode attr in type-form has not support input type: " << attr_tensor_type;
    return false;
  }
  auto new_value_node = std::make_shared<ValueNode>(TypeIdToType(kDefaultValueSwitchMap[attr_tensor_type]));
  anfnode_build_map_[value_node_name] = new_value_node;
  return true;
}

bool MSANFModelParser::GetAttrValueForValueNode(const std::string &ref_attr_name, const std::string &value_node_name,
                                                const onnx::TensorProto &attr_tensor) {
  switch (kParseTypeSwitchMap[ref_attr_name]) {
    case FORM_PARSE_SCALAR: {
      return ObtainValueNodeInScalarForm(value_node_name, attr_tensor);
    }
    case FORM_PARSE_TENSOR: {
      return ObtainValueNodeInTensorForm(value_node_name, attr_tensor);
    }
    case FORM_PARSE_TYPE: {
      return ObtainValueNodeInTypeForm(value_node_name, attr_tensor);
    }
    default:
      MS_LOG(ERROR) << "parse ValueNode value don't support input of ref_attr_name";
      return false;
  }
  return true;
}

bool MSANFModelParser::BuildValueNodeForFuncGraph(const onnx::NodeProto &node_proto) {
  const std::string &value_node_name = node_proto.output(0);
  const onnx::AttributeProto &attr_proto = node_proto.attribute(0);
  if (!attr_proto.has_ref_attr_name()) {
    MS_LOG(ERROR) << "parse ValueNode  don't have ref_attr_name";
    return false;
  }
  const std::string &ref_attr_name = attr_proto.ref_attr_name();
  const onnx::TensorProto &attr_tensor = attr_proto.t();

  return GetAttrValueForValueNode(ref_attr_name, value_node_name, attr_tensor);
}

AbstractBasePtr MSANFModelParser::GetAbstractForCNode(const onnx::AttributeProto &attr_proto) {
  std::vector<int> shape_vec;
  const onnx::TensorProto &attr_tensor = attr_proto.t();
  for (int i = 0; i < attr_tensor.dims_size(); ++i) {
    shape_vec.push_back(attr_tensor.dims(i));
  }
  tensor::TensorPtr tensor_info =
    std::make_shared<tensor::Tensor>(kDefaultValueSwitchMap[attr_tensor.data_type()], shape_vec);
  MS_EXCEPTION_IF_NULL(tensor_info);
  return tensor_info->ToAbstract();
}

bool MSANFModelParser::BuildCNodeForFuncGraph(const FuncGraphPtr &outputFuncGraph, const onnx::NodeProto &node_proto,
                                              const onnx::GraphProto &importProto, const bool &ret_flag) {
  MS_EXCEPTION_IF_NULL(outputFuncGraph);
  if (!node_proto.has_op_type()) {
    MS_LOG(ERROR) << "Get CNode op_type failed!";
    return false;
  }
  const std::string &node_name = node_proto.output(0);
  const std::string &node_type = node_proto.op_type();
  PrimitivePtr prim = std::make_shared<Primitive>(node_type);
  MS_EXCEPTION_IF_NULL(prim);

  AbstractBasePtr abstract;
  for (int i = 0; i < node_proto.attribute_size(); ++i) {
    const onnx::AttributeProto &attr_proto = node_proto.attribute(i);
    if (attr_proto.name() == kCNodeShapeAttr) {
      abstract = GetAbstractForCNode(attr_proto);
      continue;
    }
    if (!GetAttrValueForCNode(prim, attr_proto)) {
      MS_LOG(ERROR) << "Get CNode attr failed!";
      return false;
    }
  }

  std::vector<AnfNodePtr> inputs;
  inputs.clear();
  inputs.push_back(NewValueNode(prim));
  for (int i = 0; i < node_proto.input_size(); ++i) {
    const std::string &input_name = node_proto.input(i);
    if (anfnode_build_map_.find(input_name) == anfnode_build_map_.end()) {
      MS_LOG(ERROR) << node_name << " input " << i << input_name << "can't find in nodes have parsed";
      return false;
    }
    inputs.push_back(anfnode_build_map_[input_name]);
  }
  CNodePtr cnode_ptr = outputFuncGraph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(cnode_ptr);
  cnode_ptr->set_abstract(abstract);
  if (ret_flag) {
    const onnx::ValueInfoProto &output_node = importProto.output(0);
    const ::onnx::TypeProto &output_typeproto = output_node.type();
    int output_type = output_typeproto.tensor_type().elem_type();
    std::vector<int> output_shape;
    for (int i = 0; i < output_typeproto.tensor_type().shape().dim_size(); ++i) {
      output_shape.push_back(output_typeproto.tensor_type().shape().dim(i).dim_value());
    }
    tensor::TensorPtr tensor_return =
      std::make_shared<tensor::Tensor>(kDefaultValueSwitchMap[output_type], output_shape);
    inputs.clear();
    inputs.push_back(NewValueNode(prim::kPrimReturn));
    inputs.push_back(cnode_ptr);
    auto return_node = outputFuncGraph->NewCNode(inputs);
    return_node->set_abstract(tensor_return->ToAbstract());
    outputFuncGraph->set_return(return_node);
    MS_LOG(INFO) << "Construct funcgraph finined, all success!";
  }
  anfnode_build_map_[node_name] = cnode_ptr;
  return true;
}

bool MSANFModelParser::ImportNodesForGraph(const FuncGraphPtr &outputFuncGraph, const onnx::GraphProto &importProto) {
  MS_EXCEPTION_IF_NULL(outputFuncGraph);
  bool return_flag = false;
  MS_LOG(INFO) << "The CNdoe size : " << importProto.node_size();
  for (int i = 0; i < importProto.node_size(); ++i) {
    return_flag = (i == importProto.node_size() - 1) ? true : return_flag;
    const onnx::NodeProto &node_proto = importProto.node(i);
    const std::string &node_type = node_proto.op_type();
    if (node_type == kConstantValueNode) {
      if (!BuildValueNodeForFuncGraph(node_proto)) {
        MS_LOG(ERROR) << "Build ValueNode for funcgraph fail at index: : " << i;
        return false;
      }
      continue;
    }
    if (!BuildCNodeForFuncGraph(outputFuncGraph, node_proto, importProto, return_flag)) {
      MS_LOG(ERROR) << "Build CNode for funcgraph fail at index: : " << i;
      return false;
    }
  }
  return true;
}

bool MSANFModelParser::BuildFuncGraph(const FuncGraphPtr &outputFuncGraph, const onnx::GraphProto &importProto) {
  MS_EXCEPTION_IF_NULL(outputFuncGraph);
  GraphDebugInfoPtr debug_info_ptr = outputFuncGraph->debug_info();
  MS_EXCEPTION_IF_NULL(debug_info_ptr);
  if (importProto.has_name()) {
    debug_info_ptr->set_name(importProto.name());
  } else {
    MS_LOG(ERROR) << "FuncGraph under converting has not name!";
  }

  if (!ImportParametersForGraph(outputFuncGraph, importProto)) {
    return false;
  }
  return ImportNodesForGraph(outputFuncGraph, importProto);
}

bool MSANFModelParser::MSANFParseModelConfigureInfo(const onnx::ModelProto &model_proto) {
  if (!model_proto.has_producer_name()) {
    MS_LOG(ERROR) << "Parse model producer name from pb file failed!";
    return false;
  }
  producer_name_ = model_proto.producer_name();
  MS_LOG(INFO) << "producer_name :" << producer_name_;

  if (!model_proto.has_producer_version()) {
    MS_LOG(ERROR) << "Parse model producer version from pb file failed!";
    return false;
  }
  producer_version_ = model_proto.producer_version();
  MS_LOG(INFO) << "producer_version : " << producer_version_;

  if (!model_proto.has_ir_version()) {
    MS_LOG(ERROR) << "Parse model version from pb file failed!";
    return false;
  }
  ir_version_ = model_proto.ir_version();
  MS_LOG(INFO) << "ir_version :" << ir_version_;

  const onnx::OperatorSetIdProto &opset_proto = model_proto.opset_import(0);
  if (!opset_proto.has_version()) {
    MS_LOG(ERROR) << "Parse opset version from pb file failed!";
    return false;
  }
  opset_version_ = opset_proto.version();
  MS_LOG(INFO) << "opset_version : " << opset_version_;
  return true;
}

FuncGraphPtr MSANFModelParser::Parse(const onnx::ModelProto &model_proto) {
  FuncGraphPtr dstGraph = std::make_shared<FuncGraph>();
  MS_EXCEPTION_IF_NULL(dstGraph);
  if (!MSANFParseModelConfigureInfo(model_proto)) {
    MS_LOG(ERROR) << "Parse configuration info for pb file failed!";
    return nullptr;
  }
  const onnx::GraphProto &graphBuild = model_proto.graph();
  if (!BuildFuncGraph(dstGraph, graphBuild)) {
    MS_LOG(ERROR) << "Build funcgraph failed!";
    return nullptr;
  }
  MS_LOG(INFO) << "Parse pb to build FuncGraph Success!";
  return dstGraph;
}
}  // namespace lite
}  // namespace mindspore

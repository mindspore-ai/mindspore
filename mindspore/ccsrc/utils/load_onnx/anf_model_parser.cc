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

#include "utils/load_onnx/anf_model_parser.h"
#include <functional>
#include <map>
#include <memory>
#include <stack>
#include <string>
#include <vector>
#include <unordered_map>
#include <utility>
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "ir/tensor.h"
#include "ir/param_info.h"
#include "frontend/operator/ops.h"
#include "abstract/abstract_value.h"
#include "proto/onnx.pb.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"

using std::string;

namespace mindspore {
namespace lite {
static constexpr char kConstantValueNode[] = "Constant";
static constexpr char kCNodeShapeAttr[] = "shape";
static constexpr char kCNodeShape1Attr[] = "shape1";
static constexpr char kCNodeShape2Attr[] = "shape2";
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

template <typename T, typename P>
std::shared_ptr<T> ParserAttr(const std::string &str, const std::unordered_map<string, P> &kv) {
  std::stack<std::string> rules;
  std::stack<P> value;
  int count = 0;
  for (size_t i = 0; i < str.length(); i++) {
    if (str[i] == '[') {
      rules.push("[");
    } else if (str[i] == ']') {
      // rules
      std::vector<P> vec;
      while (rules.top() != "[") {
        rules.pop();
        vec.push_back(value.top());
        value.pop();
      }
      // pop "["
      rules.pop();
      // make tuple for names
      std::string res = "dummy";
      // make tuple for values
      reverse(vec.begin(), vec.end());
      auto vt = std::make_shared<T>(vec);
      if (rules.empty() && value.empty()) {
        return vt;
      }
      rules.push(res);
      value.push(vt);
    } else if (str[i] == ',') {
      continue;
    } else {
      count++;
      if (str[i + 1] == '[' || str[i + 1] == ']' || str[i + 1] == ',') {
        auto value_name = str.substr(i - count + 1, count);
        value.push(kv.at(value_name));
        rules.push(value_name);
        count = 0;
      }
    }
  }
  return {};
}

std::shared_ptr<ValueTuple> ParserScalarAttrValue(const std::string &attr_name,
                                                  const std::unordered_map<string, ValuePtr> &kv) {
  std::string str = attr_name;
  auto replace = [&](const string &orgStr, const string &newStr) {
    std::string::size_type pos(0);
    while ((pos = str.find(orgStr)) != std::string::npos) {
      str.replace(pos, orgStr.length(), newStr);
    }
    return str;
  };
  // remove "scalar:"
  str = replace("scalar:", "");
  // remove "Tuple"
  str = replace("Tuple", "");
  // remove "List"
  str = replace("List", "");
  auto result = ParserAttr<ValueTuple>(str, kv);
  if (!result) {
    return {};
  }
  return result;
}

std::shared_ptr<abstract::AbstractTuple> ParserAttrShape(
  const std::string &attr_name, const std::unordered_map<string, abstract::AbstractBasePtr> &kv) {
  std::string str = attr_name;
  auto replace = [&](const string &orgStr, const string &newStr) {
    std::string::size_type pos(0);
    while ((pos = str.find(orgStr)) != std::string::npos) {
      str.replace(pos, orgStr.length(), newStr);
    }
    return str;
  };
  // remove "scalar:"
  str = replace("shape:", "");
  // remove "Tuple"
  str = replace("Tuple", "");
  // remove "List"
  str = replace("List", "");

  auto result = ParserAttr<abstract::AbstractTuple>(str, kv);
  if (!result) {
    return {};
  }
  return result;
}

#define PARSE_ONNXATTR_IN_SCALAR_FORM(type, valuetype)                                    \
  ValuePtr ParseAttrInScalar_##type##_##valuetype(const onnx::TensorProto &attr_tensor) { \
    auto value = static_cast<valuetype>(attr_tensor.type##_data(0));                      \
    return MakeValue<valuetype>(value);                                                   \
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
  ShapeVector shape;
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
    auto *tensor_data_buf = reinterpret_cast<uint8_t *>(tensor_info->data_c());
    MS_EXCEPTION_IF_NULL(tensor_data_buf);
    auto ret = memcpy_s(tensor_data_buf, tensor_info->data().nbytes(), initial_data.data(), initial_data.size());
    if (ret != 0) {
      MS_LOG(EXCEPTION) << "memcpy_s error, errorno" << ret;
    }

    node->set_default_param(tensor_info);
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

ValuePtr MSANFModelParser::ObtainCNodeAttrInScalarForm(const onnx::TensorProto &attr_tensor) {
  const int attr_tensor_type = attr_tensor.data_type();
  switch (attr_tensor_type) {
    case onnx::TensorProto_DataType_STRING: {
      return ParseAttrInScalar_string_string(attr_tensor);
    }
    case onnx::TensorProto_DataType_INT32: {
      return ParseAttrInScalar_int32_int32(attr_tensor);
    }
    case onnx::TensorProto_DataType_INT64: {
      return ParseAttrInScalar_int64_int64(attr_tensor);
    }
    case onnx::TensorProto_DataType_UINT64: {
      return ParseAttrInScalar_uint64_uint64(attr_tensor);
    }
    case onnx::TensorProto_DataType_FLOAT: {
      return ParseAttrInScalar_float_float(attr_tensor);
    }
    case onnx::TensorProto_DataType_DOUBLE: {
      return ParseAttrInScalar_double_double(attr_tensor);
    }
    case onnx::TensorProto_DataType_BOOL: {
      return ParseAttrInScalar_int32_bool(attr_tensor);
    }
    default:
      MS_LOG(ERROR) << "Obtain attr in scalar-form has not support input type: " << attr_tensor_type;
      return {};
  }
  return {};
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
  string type;
  std::size_t pos(0);
  if ((pos = ref_attr_name.find("scalar:")) != std::string::npos) {
    type = ref_attr_name.substr(pos, string("scalar:").length() - 1);
  } else if ((pos = ref_attr_name.find("type:")) != std::string::npos) {
    type = ref_attr_name.substr(pos, string("type:").length() - 1);
  } else if ((pos = ref_attr_name.find("tensor:")) != std::string::npos) {
    type = ref_attr_name.substr(pos, string("tensor:").length() - 1);
  }
  std::unordered_map<std::string, ValuePtr> kv;
  for (int i = 0; i < attr_proto.tensors_size(); i++) {
    const onnx::TensorProto &attr_tensor = attr_proto.tensors(i);
    switch (kParseTypeSwitchMap[type]) {
      case FORM_PARSE_TYPE: {
        ObtainCNodeAttrInTypeForm(prim, attr_name, attr_tensor);
        break;
      }
      case FORM_PARSE_SCALAR: {
        auto res = ObtainCNodeAttrInScalarForm(attr_tensor);
        kv.insert(std::pair<string, ValuePtr>(attr_tensor.name(), res));
        break;
      }
      case FORM_PARSE_TENSOR: {
        ObtainCNodeAttrInTensorForm(prim, attr_name, attr_tensor);
        break;
      }
      default:
        MS_LOG(ERROR) << "parse attr type don't support input of ref_attr_name";
        return false;
    }
  }

  if (kParseTypeSwitchMap[type] == FORM_PARSE_SCALAR) {
    if (kv.size() == 1) {
      auto iter = kv.begin();
      prim->AddAttr(attr_name, iter->second);
    } else {
      auto res = ParserScalarAttrValue(ref_attr_name, kv);
      prim->AddAttr(attr_name, res);
    }
  }
  return true;
}
bool MSANFModelParser::ObtainValueNodeInTensorForm(const std::string &value_node_name,
                                                   const onnx::TensorProto &attr_tensor) {
  const int attr_tensor_type = attr_tensor.data_type();
  ShapeVector shape;
  for (int i = 0; i < attr_tensor.dims_size(); ++i) {
    shape.push_back(attr_tensor.dims(i));
  }
  tensor::TensorPtr tensor_info = std::make_shared<tensor::Tensor>(kDefaultValueSwitchMap[attr_tensor_type], shape);
  const std::string &tensor_buf = attr_tensor.raw_data();
  auto *tensor_data_buf = reinterpret_cast<uint8_t *>(tensor_info->data_c());
  auto ret = memcpy_s(tensor_data_buf, tensor_info->data().nbytes(), tensor_buf.data(), tensor_buf.size());
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "memcpy_s error, errorno" << ret;
  }

  auto new_value_node = NewValueNode(MakeValue(tensor_info));
  MS_EXCEPTION_IF_NULL(new_value_node);
  auto tensor_abstract = tensor_info->ToAbstract();
  MS_EXCEPTION_IF_NULL(tensor_abstract);
  new_value_node->set_abstract(tensor_abstract);
  anfnode_build_map_[value_node_name] = new_value_node;
  return true;
}

bool MSANFModelParser::ObtainValueNodeInScalarForm(const std::string &value_node_name,
                                                   const onnx::TensorProto &attr_tensor) {
  const int attr_tensor_type = attr_tensor.data_type();
  ValuePtr value_ptr = nullptr;
  switch (attr_tensor_type) {
    case onnx::TensorProto_DataType_INT32: {
      std::vector<int64_t> add_data;
      for (int i = 0; i < attr_tensor.int32_data_size(); ++i) {
        add_data.push_back(attr_tensor.int32_data(i));
      }
      if (add_data.size() == 1) {
        value_ptr = MakeValue(add_data[0]);
      } else if (!add_data.empty()) {
        value_ptr = MakeValue<std::vector<int64_t>>(add_data);
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
  auto new_value_node = NewValueNode(TypeIdToType(kDefaultValueSwitchMap[attr_tensor_type]));
  abstract::AbstractTypePtr abs_type = std::make_shared<abstract::AbstractType>(std::make_shared<TypeType>());
  new_value_node->set_abstract(abs_type);
  anfnode_build_map_[value_node_name] = new_value_node;
  return true;
}

bool MSANFModelParser::GetAttrValueForValueNode(const std::string &value_node_name,
                                                const onnx::AttributeProto &attr_proto) {
  if (!attr_proto.has_ref_attr_name()) {
    MS_LOG(ERROR) << "CNode parse attr type has no ref_attr_name";
    return false;
  }
  const std::string &ref_attr_name = attr_proto.ref_attr_name();
  string type;
  std::size_t pos(0);
  if ((pos = ref_attr_name.find("scalar:")) != std::string::npos) {
    type = ref_attr_name.substr(pos, string("scalar:").length() - 1);
  } else if ((pos = ref_attr_name.find("type:")) != std::string::npos) {
    type = ref_attr_name.substr(pos, string("type:").length() - 1);
  } else if ((pos = ref_attr_name.find("tensor:")) != std::string::npos) {
    type = ref_attr_name.substr(pos, string("tensor:").length() - 1);
  }
  std::unordered_map<std::string, ValuePtr> kv;
  for (int i = 0; i < attr_proto.tensors_size(); i++) {
    const onnx::TensorProto &attr_tensor = attr_proto.tensors(i);
    auto attr_name = attr_tensor.name();
    switch (kParseTypeSwitchMap[type]) {
      case FORM_PARSE_TYPE: {
        return ObtainValueNodeInTypeForm(value_node_name, attr_tensor);
      }
      case FORM_PARSE_SCALAR: {
        auto res = ObtainCNodeAttrInScalarForm(attr_tensor);
        kv.insert(std::pair<string, ValuePtr>(attr_tensor.name(), res));
        break;
      }
      case FORM_PARSE_TENSOR: {
        return ObtainValueNodeInTensorForm(value_node_name, attr_tensor);
      }
      default:
        MS_LOG(ERROR) << "parse attr type don't support input of ref_attr_name";
        return false;
    }
  }

  ValueNodePtr new_value_node;
  if (kParseTypeSwitchMap[type] == FORM_PARSE_SCALAR) {
    if (kv.size() == 1) {
      auto iter = kv.begin();
      new_value_node = NewValueNode(iter->second);
      new_value_node->set_abstract(iter->second->ToAbstract());
    } else {
      auto value_ptr = ParserScalarAttrValue(ref_attr_name, kv);
      new_value_node = NewValueNode(value_ptr);
      new_value_node->set_abstract(value_ptr->ToAbstract());
    }
    anfnode_build_map_[value_node_name] = new_value_node;
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
  return GetAttrValueForValueNode(value_node_name, attr_proto);
}

std::unordered_map<std::string, abstract::AbstractBasePtr> MSANFModelParser::GetAbstractForCNode(
  const onnx::AttributeProto &attr_proto) {
  std::unordered_map<std::string, abstract::AbstractBasePtr> kv;
  for (int i = 0; i < attr_proto.tensors_size(); ++i) {
    ShapeVector shape_vec;
    const onnx::TensorProto &attr_tensor = attr_proto.tensors(i);
    for (int j = 0; j < attr_tensor.dims_size(); ++j) {
      shape_vec.push_back(attr_tensor.dims(j));
    }
    tensor::TensorPtr tensor_info =
      std::make_shared<tensor::Tensor>(kDefaultValueSwitchMap[attr_tensor.data_type()], shape_vec);
    MS_EXCEPTION_IF_NULL(tensor_info);
    auto abstract = tensor_info->ToAbstract();
    MS_EXCEPTION_IF_NULL(abstract);
    kv.insert(std::pair<string, abstract::AbstractBasePtr>(attr_tensor.name(), abstract));
  }
  return kv;
}

CNodePtr MSANFModelParser::BuildCNodeForFuncGraph(const FuncGraphPtr &outputFuncGraph,
                                                  const onnx::NodeProto &node_proto) {
  MS_EXCEPTION_IF_NULL(outputFuncGraph);
  if (!node_proto.has_op_type()) {
    MS_LOG(ERROR) << "Get CNode op_type failed!";
    return nullptr;
  }
  const std::string &node_name = node_proto.output(0);
  const std::string &fullname_with_scope = node_proto.domain();
  const std::string &node_type = node_proto.op_type();
  PrimitivePtr prim = std::make_shared<Primitive>(node_type);
  MS_EXCEPTION_IF_NULL(prim);
  prim->set_instance_name(node_type);

  std::unordered_map<std::string, abstract::AbstractBasePtr> kv;
  string shape_ref_attr_name;
  for (int i = 0; i < node_proto.attribute_size(); ++i) {
    const onnx::AttributeProto &attr_proto = node_proto.attribute(i);
    if (attr_proto.ref_attr_name().find("shape:") != string::npos) {
      shape_ref_attr_name = attr_proto.ref_attr_name();
      kv = GetAbstractForCNode(attr_proto);
      continue;
    }
    if (!GetAttrValueForCNode(prim, attr_proto)) {
      MS_LOG(ERROR) << "Get CNode attr failed!";
      return nullptr;
    }
  }

  std::vector<AnfNodePtr> inputs;
  inputs.clear();
  inputs.push_back(NewValueNode(prim));
  for (int i = 0; i < node_proto.input_size(); ++i) {
    const std::string &input_name = node_proto.input(i);
    if (anfnode_build_map_.find(input_name) == anfnode_build_map_.end()) {
      MS_LOG(ERROR) << node_name << " input " << i << input_name << "can't find in nodes have parsed";
      return nullptr;
    }
    inputs.push_back(anfnode_build_map_[input_name]);
  }
  CNodePtr cnode_ptr = outputFuncGraph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(cnode_ptr);
  if (0 == kv.size()) {
    AbstractBasePtrList elem;
    for (size_t index = 1; index < cnode_ptr->inputs().size(); ++index) {
      elem.push_back(cnode_ptr->input(index)->abstract());
    }
    cnode_ptr->set_abstract(std::make_shared<abstract::AbstractTuple>(elem));
  } else if (1 == kv.size()) {
    std::unordered_map<std::string, abstract::AbstractBasePtr>::iterator iter = kv.begin();
    cnode_ptr->set_abstract(iter->second);
  } else {
    auto abstract = ParserAttrShape(shape_ref_attr_name, kv);
    cnode_ptr->set_abstract(abstract);
  }
  cnode_ptr->set_fullname_with_scope(fullname_with_scope);
  anfnode_build_map_[node_name] = cnode_ptr;
  return cnode_ptr;
}

bool MSANFModelParser::BuildReturnForFuncGraph(const FuncGraphPtr &outputFuncGraph, const onnx::GraphProto &importProto,
                                               const CNodePtr &cnode_ptr) {
  MS_EXCEPTION_IF_NULL(outputFuncGraph);
  MS_EXCEPTION_IF_NULL(cnode_ptr);
  std::vector<AnfNodePtr> inputs;
  if (importProto.output_size() > 1) {
    inputs.clear();
    inputs.push_back(NewValueNode(prim::kPrimMakeTuple));
    AbstractBasePtrList elem;
    for (int out_size = 0; out_size < importProto.output_size(); ++out_size) {
      const onnx::ValueInfoProto &output_node = importProto.output(out_size);
      const std::string &out_tuple = output_node.name();
      inputs.push_back(anfnode_build_map_[out_tuple]);
      elem.push_back(anfnode_build_map_[out_tuple]->abstract());
    }
    auto maketuple_ptr = outputFuncGraph->NewCNode(inputs);
    maketuple_ptr->set_abstract(std::make_shared<abstract::AbstractTuple>(elem));
    inputs.clear();
    inputs.push_back(NewValueNode(prim::kPrimReturn));
    inputs.push_back(maketuple_ptr);
    auto return_node = outputFuncGraph->NewCNode(inputs);
    MS_EXCEPTION_IF_NULL(return_node);
    outputFuncGraph->set_return(return_node);
    MS_LOG(INFO) << "Construct funcgraph finined, all success.";
  } else {
    const onnx::ValueInfoProto &output_node = importProto.output(0);
    const onnx::TypeProto &output_typeproto = output_node.type();
    ShapeVector output_shape;
    for (int i = 0; i < output_typeproto.tensor_type().shape().dim_size(); ++i) {
      output_shape.push_back(output_typeproto.tensor_type().shape().dim(i).dim_value());
    }
    inputs.clear();
    inputs.push_back(NewValueNode(prim::kPrimReturn));
    inputs.push_back(cnode_ptr);
    auto return_node = outputFuncGraph->NewCNode(inputs);
    MS_EXCEPTION_IF_NULL(return_node);
    outputFuncGraph->set_return(return_node);
    MS_LOG(INFO) << "Construct funcgraph finined, all success!";
  }
  return true;
}

bool MSANFModelParser::ImportNodesForGraph(const FuncGraphPtr &outputFuncGraph, const onnx::GraphProto &importProto) {
  MS_EXCEPTION_IF_NULL(outputFuncGraph);
  MS_LOG(INFO) << "The CNdoe size : " << importProto.node_size();
  CNodePtr cnode_ptr = nullptr;
  for (int i = 0; i < importProto.node_size(); ++i) {
    const onnx::NodeProto &node_proto = importProto.node(i);
    const std::string &node_type = node_proto.op_type();
    if (node_type == kConstantValueNode) {
      if (!BuildValueNodeForFuncGraph(node_proto)) {
        MS_LOG(ERROR) << "Build ValueNode for funcgraph fail at index: : " << i;
        return false;
      }
      continue;
    }
    cnode_ptr = BuildCNodeForFuncGraph(outputFuncGraph, node_proto);
    if (cnode_ptr == nullptr) {
      MS_LOG(ERROR) << "Build CNode for funcgraph fail at index: : " << i;
      return false;
    }
  }

  BuildReturnForFuncGraph(outputFuncGraph, importProto, cnode_ptr);
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

  if (!model_proto.has_model_version()) {
    MS_LOG(ERROR) << "Parse model producer version from pb file failed!";
    return false;
  }
  model_version_ = model_proto.model_version();
  MS_LOG(INFO) << "producer_version : " << model_version_;

  if (!model_proto.has_ir_version()) {
    MS_LOG(ERROR) << "Parse model version from pb file failed!";
    return false;
  }
  ir_version_ = model_proto.ir_version();
  MS_LOG(INFO) << "ir_version :" << ir_version_;
  return true;
}

FuncGraphPtr MSANFModelParser::Parse(const onnx::ModelProto &model_proto) {
  FuncGraphPtr dstGraph = std::make_shared<FuncGraph>();
  MS_EXCEPTION_IF_NULL(dstGraph);
  if (!MSANFParseModelConfigureInfo(model_proto)) {
    MS_LOG(ERROR) << "Parse configuration info for pb file failed!";
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

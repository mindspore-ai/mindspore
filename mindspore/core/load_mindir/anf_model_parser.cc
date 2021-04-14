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

#include "load_mindir/anf_model_parser.h"
#include <limits.h>
#include <functional>
#include <map>
#include <memory>
#include <stack>
#include <string>
#include <vector>
#include <unordered_map>
#include <utility>
#include "ir/tensor.h"
#include "ir/param_info.h"
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "utils/check_convert_utils.h"

using std::string;

namespace mindspore {
static constexpr char kConstantValueNode[] = "Constant";
static constexpr char kCNodeShapeAttr[] = "shape";
static constexpr char kCNodeShape1Attr[] = "shape1";
static constexpr char kCNodeShape2Attr[] = "shape2";
enum ParseForm : int {
  FORM_PARSE_TYPE = 0,
  FORM_PARSE_SCALAR = 1,
  FORM_PARSE_TENSOR = 2,
  FORM_PARSE_NONE = 3,
  FORM_PARSE_MONAD = 4,
  FORM_PARSE_UNDEFINE = 5,
};

static std::map<std::string, ParseForm> kParseTypeSwitchMap{
  {"type", FORM_PARSE_TYPE}, {"scalar", FORM_PARSE_SCALAR}, {"tensor", FORM_PARSE_TENSOR},
  {"none", FORM_PARSE_NONE}, {"Monad", FORM_PARSE_MONAD},   {"", FORM_PARSE_UNDEFINE}};

static std::unordered_map<int, TypeId> kDefaultValueSwitchMap{
  {mind_ir::TensorProto_DataType_BOOL, kNumberTypeBool},
  {mind_ir::TensorProto_DataType_INT8, kNumberTypeInt8},
  {mind_ir::TensorProto_DataType_INT16, kNumberTypeInt16},
  {mind_ir::TensorProto_DataType_INT32, kNumberTypeInt32},
  {mind_ir::TensorProto_DataType_INT64, kNumberTypeInt64},
  {mind_ir::TensorProto_DataType_UINT8, kNumberTypeUInt8},
  {mind_ir::TensorProto_DataType_UINT16, kNumberTypeUInt16},
  {mind_ir::TensorProto_DataType_UINT32, kNumberTypeUInt32},
  {mind_ir::TensorProto_DataType_UINT64, kNumberTypeUInt64},
  {mind_ir::TensorProto_DataType_FLOAT16, kNumberTypeFloat16},
  {mind_ir::TensorProto_DataType_FLOAT, kNumberTypeFloat32},
  {mind_ir::TensorProto_DataType_FLOAT64, kNumberTypeFloat64},
  {mind_ir::TensorProto_DataType_DOUBLE, kNumberTypeFloat64},
  {mind_ir::TensorProto_DataType_STRING, kObjectTypeString},
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
        if (kv.find(value_name) == kv.end()) {
          MS_LOG(ERROR) << "Node's attributes and shape do not match.";
          return nullptr;
        }
        value.push(kv.at(value_name));
        rules.push(value_name);
        count = 0;
      }
    }
  }
  return {};
}

template <typename T>
std::shared_ptr<T> ParserScalarAttrValue(const std::string &attr_name, const std::unordered_map<string, ValuePtr> &kv) {
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
  auto result = ParserAttr<T>(str, kv);
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
  return result;
}

std::string ParseParameterName(const string &name) {
  string delimiter = ":";
  size_t pos(0);
  if ((pos = name.find(delimiter)) != string::npos) {
    return name.substr(pos + 1, string::npos - (pos + 1));
  }
  return name;
}

std::string ParseCNodeName(const string &name) {
  string delimiter = ":";
  size_t pos = name.find(delimiter);
  size_t end_pos = name.find_last_of(delimiter);

  if (pos != string::npos && end_pos != string::npos && pos != end_pos) {
    return name.substr(pos + 1, end_pos - (pos + 1));
  }
  return name;
}

#define PARSE_MINDIR_ATTR_IN_INT_FORM(type, valuetype)                                                    \
  ValuePtr ParseAttrInScalar_##type##_##valuetype(const mind_ir::AttributeProto &attr_proto, int index) { \
    auto value = static_cast<valuetype>(attr_proto.ints(index));                                          \
    return MakeValue<valuetype>(value);                                                                   \
  }                                                                                                       \
  ValuePtr ParseAttrInSingleScalar_##type##_##valuetype(const mind_ir::AttributeProto &attr_proto) {      \
    auto value = static_cast<valuetype>(attr_proto.i());                                                  \
    return MakeValue<valuetype>(value);                                                                   \
  }

#define PARSE_MINDIR_ATTR_IN_SCALAR_FORM(type, valuetype)                                                 \
  ValuePtr ParseAttrInScalar_##type##_##valuetype(const mind_ir::AttributeProto &attr_proto, int index) { \
    auto value = static_cast<valuetype>(attr_proto.type##s(index));                                       \
    return MakeValue<valuetype>(value);                                                                   \
  }

PARSE_MINDIR_ATTR_IN_INT_FORM(int8_t, int8_t)
PARSE_MINDIR_ATTR_IN_INT_FORM(int16_t, int16_t)
PARSE_MINDIR_ATTR_IN_INT_FORM(int32_t, int32_t)
PARSE_MINDIR_ATTR_IN_INT_FORM(int64_t, int64_t)
PARSE_MINDIR_ATTR_IN_INT_FORM(uint8_t, uint8_t)
PARSE_MINDIR_ATTR_IN_INT_FORM(uint16_t, uint16_t)
PARSE_MINDIR_ATTR_IN_INT_FORM(uint32_t, uint32_t)
PARSE_MINDIR_ATTR_IN_INT_FORM(uint64_t, uint64_t)
PARSE_MINDIR_ATTR_IN_INT_FORM(int32_t, bool)

PARSE_MINDIR_ATTR_IN_SCALAR_FORM(double, double)
PARSE_MINDIR_ATTR_IN_SCALAR_FORM(float, float)
PARSE_MINDIR_ATTR_IN_SCALAR_FORM(string, string)

ValuePtr ParseAttrInSingleScalar_string_string(const mind_ir::AttributeProto &attr_proto) {
  auto value = static_cast<string>(attr_proto.s());
  return MakeValue<string>(value);
}

ValuePtr ParseAttrInSingleScalar_float_float(const mind_ir::AttributeProto &attr_proto) {
  auto value = static_cast<float>(attr_proto.f());
  return MakeValue<float>(value);
}

ValuePtr ParseAttrInSingleScalar_double_double(const mind_ir::AttributeProto &attr_proto) {
  auto value = static_cast<double>(attr_proto.d());
  return MakeValue<double>(value);
}

tensor::TensorPtr MSANFModelParser::BuildTensorInfoForFuncGraph(const mind_ir::TensorProto &tensor_proto) {
  ShapeVector shape;
  for (int i = 0; i < tensor_proto.dims_size(); ++i) {
    shape.push_back(tensor_proto.dims(i));
  }

  if (!tensor_proto.has_data_type()) {
    MS_LOG(ERROR) << "mind_ir build tensor: " << tensor_proto.name() << " failed";
    MS_LOG(EXCEPTION) << "mind_ir TensorProto has no data_type.";
  }
  if (kDefaultValueSwitchMap.find(tensor_proto.data_type()) == kDefaultValueSwitchMap.end()) {
    MS_LOG(ERROR) << "mind_ir build tensor: " << tensor_proto.name() << " failed";
    MS_LOG(EXCEPTION) << "mind_ir TensorProto data_type: " << tensor_proto.data_type() << " is not support yet!";
  }

  tensor::TensorPtr tensor_info =
    std::make_shared<tensor::Tensor>(kDefaultValueSwitchMap[tensor_proto.data_type()], shape);
  return tensor_info;
}

bool MSANFModelParser::BuildParameterForFuncGraph(const ParameterPtr &node,
                                                  const mind_ir::TensorProto &parameter_proto) {
  MS_EXCEPTION_IF_NULL(node);

  if (!parameter_proto.has_name()) {
    MS_LOG(ERROR) << "mind_ir TensorProto has no name!";
    return false;
  }
  string debug_info_name = ParseParameterName(parameter_proto.name());
  auto debug_info_ptr = std::make_shared<NodeDebugInfo>(debug_info_name);
  node->set_debug_info(debug_info_ptr);
  node->set_name(debug_info_name);

  tensor::TensorPtr tensor_info = BuildTensorInfoForFuncGraph(parameter_proto);
  MS_EXCEPTION_IF_NULL(tensor_info);
  ParamInfoPtr param_info = std::make_shared<ParamInfo>();
  param_info->set_name(debug_info_name);
  tensor_info->set_param_info(param_info);

  auto tensor_abstract = tensor_info->ToAbstract();
  MS_EXCEPTION_IF_NULL(tensor_abstract);
  node->set_abstract(tensor_abstract);

  std::string initial_data = parameter_proto.raw_data();
  auto *tensor_data_buf = reinterpret_cast<uint8_t *>(tensor_info->data_c());
  MS_EXCEPTION_IF_NULL(tensor_data_buf);
  auto ret = memcpy_s(tensor_data_buf, tensor_info->data().nbytes(), initial_data.data(), initial_data.size());
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "memcpy_s error for build parameter, errorno " << ret;
  }

  node->set_default_param(tensor_info);

  anfnode_build_map_[parameter_proto.name()] = node;
  return true;
}

bool MSANFModelParser::BuildInputForFuncGraph(const ParameterPtr &node, const mind_ir::ValueInfoProto &value_proto) {
  MS_EXCEPTION_IF_NULL(node);

  if (!value_proto.has_name()) {
    MS_LOG(ERROR) << "mind_ir ValueInfoProto has no name!";
    return false;
  }
  string debug_info_name = ParseParameterName(value_proto.name());
  auto debug_info_ptr = std::make_shared<NodeDebugInfo>(debug_info_name);
  node->set_debug_info(debug_info_ptr);
  node->set_name(debug_info_name);

  const mind_ir::TensorProto &tensor_proto = value_proto.tensor(0);

  tensor::TensorPtr tensor_info = BuildTensorInfoForFuncGraph(tensor_proto);
  MS_EXCEPTION_IF_NULL(tensor_info);
  auto tensor_abstract = tensor_info->ToAbstract();
  node->set_abstract(tensor_abstract);

  anfnode_build_map_[value_proto.name()] = node;
  return true;
}

bool MSANFModelParser::ImportParametersForGraph(const FuncGraphPtr &outputFuncGraph,
                                                const mind_ir::GraphProto &importProto) {
  MS_EXCEPTION_IF_NULL(outputFuncGraph);
  MS_LOG(INFO) << "All inputs size is: " << importProto.input_size();
  for (int i = 0; i < importProto.input_size(); ++i) {
    const mind_ir::ValueInfoProto &input_proto = importProto.input(i);
    if (!BuildInputForFuncGraph(outputFuncGraph->add_parameter(), input_proto)) {
      MS_LOG(ERROR) << "Build input for funcgraph fail at index: " << i;
      return false;
    }
  }

  MS_LOG(INFO) << "All Parameters size is: " << importProto.parameter_size();
  for (int i = 0; i < importProto.parameter_size(); ++i) {
    const mind_ir::TensorProto &parameter_proto = importProto.parameter(i);
    if (!BuildParameterForFuncGraph(outputFuncGraph->add_parameter(), parameter_proto)) {
      MS_LOG(ERROR) << "Build parameter for funcgraph fail at index: " << i;
      return false;
    }
  }
  return true;
}

bool MSANFModelParser::ObtainCNodeAttrInTypeForm(const PrimitivePtr &prim, const mind_ir::AttributeProto &attr_proto) {
  MS_EXCEPTION_IF_NULL(prim);
  const int attr_tensor_type = attr_proto.tensors(0).data_type();
  if (kDefaultValueSwitchMap.find(attr_tensor_type) == kDefaultValueSwitchMap.end()) {
    MS_LOG(ERROR) << "Obtain attr in type-form has not support input type:" << attr_tensor_type;
    return false;
  }
  prim->AddAttr(attr_proto.name(), TypeIdToType(kDefaultValueSwitchMap[attr_tensor_type]));
  return true;
}

ValuePtr MSANFModelParser::ParseAttrInScalarForm(const mind_ir::AttributeProto &attr_proto, int index) {
  const int attr_type = attr_proto.type();
  switch (attr_type) {
    case mind_ir::AttributeProto_AttributeType_STRING: {
      return ParseAttrInScalar_string_string(attr_proto, index);
    }
    case mind_ir::AttributeProto_AttributeType_INT8: {
      return ParseAttrInScalar_int8_t_int8_t(attr_proto, index);
    }
    case mind_ir::AttributeProto_AttributeType_INT16: {
      return ParseAttrInScalar_int16_t_int16_t(attr_proto, index);
    }
    case mind_ir::AttributeProto_AttributeType_INT32: {
      return ParseAttrInScalar_int32_t_int32_t(attr_proto, index);
    }
    case mind_ir::AttributeProto_AttributeType_INT64: {
      return ParseAttrInScalar_int64_t_int64_t(attr_proto, index);
    }
    case mind_ir::AttributeProto_AttributeType_UINT8: {
      return ParseAttrInScalar_uint8_t_uint8_t(attr_proto, index);
    }
    case mind_ir::AttributeProto_AttributeType_UINT16: {
      return ParseAttrInScalar_uint16_t_uint16_t(attr_proto, index);
    }
    case mind_ir::AttributeProto_AttributeType_UINT32: {
      return ParseAttrInScalar_uint32_t_uint32_t(attr_proto, index);
    }
    case mind_ir::AttributeProto_AttributeType_UINT64: {
      return ParseAttrInScalar_uint64_t_uint64_t(attr_proto, index);
    }
    case mind_ir::AttributeProto_AttributeType_FLOAT: {
      return ParseAttrInScalar_float_float(attr_proto, index);
    }
    case mind_ir::AttributeProto_AttributeType_DOUBLE: {
      return ParseAttrInScalar_double_double(attr_proto, index);
    }
    case mind_ir::AttributeProto_AttributeType_BOOL: {
      return ParseAttrInScalar_int32_t_bool(attr_proto, index);
    }
    default:
      MS_LOG(ERROR) << "Obtain attr in scalar-form has not support input type: " << attr_type;
      return {};
  }
  return {};
}

void MSANFModelParser::ObtainCNodeAttrInScalarForm(const mind_ir::AttributeProto &attr_proto,
                                                   std::unordered_map<std::string, ValuePtr> *multi_value_map) {
  string name;
  for (int i = 0; i < attr_proto.ints_size(); i++) {
    auto res = ParseAttrInScalarForm(attr_proto, i);
    name = "value" + std::to_string(i + 1);
    multi_value_map->insert(std::pair<string, ValuePtr>(name, res));
  }
  for (int i = 0; i < attr_proto.doubles_size(); i++) {
    auto res = ParseAttrInScalarForm(attr_proto, i);
    name = "value" + std::to_string(i + 1);
    multi_value_map->insert(std::pair<string, ValuePtr>(name, res));
  }
  for (int i = 0; i < attr_proto.floats_size(); i++) {
    auto res = ParseAttrInScalarForm(attr_proto, i);
    name = "value" + std::to_string(i + 1);
    multi_value_map->insert(std::pair<string, ValuePtr>(name, res));
  }
  for (int i = 0; i < attr_proto.strings_size(); i++) {
    auto res = ParseAttrInScalarForm(attr_proto, i);
    name = "value" + std::to_string(i + 1);
    multi_value_map->insert(std::pair<string, ValuePtr>(name, res));
  }
}

ValuePtr MSANFModelParser::ObtainCNodeAttrInSingleScalarForm(const mind_ir::AttributeProto &attr_proto) {
  const int attr_type = attr_proto.type();
  switch (attr_type) {
    case mind_ir::AttributeProto_AttributeType_STRING: {
      return ParseAttrInSingleScalar_string_string(attr_proto);
    }
    case mind_ir::AttributeProto_AttributeType_INT8: {
      return ParseAttrInSingleScalar_int8_t_int8_t(attr_proto);
    }
    case mind_ir::AttributeProto_AttributeType_INT16: {
      return ParseAttrInSingleScalar_int16_t_int16_t(attr_proto);
    }
    case mind_ir::AttributeProto_AttributeType_INT32: {
      return ParseAttrInSingleScalar_int32_t_int32_t(attr_proto);
    }
    case mind_ir::AttributeProto_AttributeType_INT64: {
      return ParseAttrInSingleScalar_int64_t_int64_t(attr_proto);
    }
    case mind_ir::AttributeProto_AttributeType_UINT8: {
      return ParseAttrInSingleScalar_uint8_t_uint8_t(attr_proto);
    }
    case mind_ir::AttributeProto_AttributeType_UINT16: {
      return ParseAttrInSingleScalar_uint16_t_uint16_t(attr_proto);
    }
    case mind_ir::AttributeProto_AttributeType_UINT32: {
      return ParseAttrInSingleScalar_uint32_t_uint32_t(attr_proto);
    }
    case mind_ir::AttributeProto_AttributeType_UINT64: {
      return ParseAttrInSingleScalar_uint64_t_uint64_t(attr_proto);
    }
    case mind_ir::AttributeProto_AttributeType_FLOAT: {
      return ParseAttrInSingleScalar_float_float(attr_proto);
    }
    case mind_ir::AttributeProto_AttributeType_DOUBLE: {
      return ParseAttrInSingleScalar_double_double(attr_proto);
    }
    case mind_ir::AttributeProto_AttributeType_BOOL: {
      return ParseAttrInSingleScalar_int32_t_bool(attr_proto);
    }
    default:
      MS_LOG(ERROR) << "Obtain attr in scalar-form has not support input type: " << attr_type;
      return {};
  }
  return {};
}

bool MSANFModelParser::ObtainCNodeAttrInTensorForm(const PrimitivePtr &prim,
                                                   const mind_ir::AttributeProto &attr_proto) {
  MS_EXCEPTION_IF_NULL(prim);
  const mind_ir::TensorProto attr_tensor = attr_proto.tensors(0);
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
  prim->AddAttr(attr_proto.name(), MakeValue(tensor_info));
  return true;
}

bool MSANFModelParser::GetAttrValueForCNode(const PrimitivePtr &prim, const mind_ir::AttributeProto &attr_proto) {
  MS_EXCEPTION_IF_NULL(prim);
  const std::string &attr_name = attr_proto.name();
  if (!attr_proto.has_ref_attr_name()) {
    MS_LOG(ERROR) << "CNode parse attr type has no ref_attr_name";
    return false;
  }
  const std::string &ref_attr_name = attr_proto.ref_attr_name();
  string type = "";
  std::size_t pos(0);
  if ((pos = ref_attr_name.find("scalar:")) != std::string::npos) {
    type = ref_attr_name.substr(pos, string("scalar:").length() - 1);
  } else if ((pos = ref_attr_name.find("type:")) != std::string::npos) {
    type = ref_attr_name.substr(pos, string("type:").length() - 1);
  } else if ((pos = ref_attr_name.find("tensor:")) != std::string::npos) {
    type = ref_attr_name.substr(pos, string("tensor:").length() - 1);
  }
  std::unordered_map<std::string, ValuePtr> multi_value_map;
  switch (kParseTypeSwitchMap[type]) {
    case FORM_PARSE_TYPE: {
      ObtainCNodeAttrInTypeForm(prim, attr_proto);
      break;
    }
    case FORM_PARSE_SCALAR: {
      std::size_t value_pos(0);
      if ((value_pos = ref_attr_name.find("value0")) != std::string::npos) {
        ValuePtr res = ObtainCNodeAttrInSingleScalarForm(attr_proto);
        const std::string &op_type = prim->name();
        if (!IsLite()) {
          CheckAndConvertUtils::ConvertAttrValueInLoad(op_type, attr_name, &res);
        }
        prim->AddAttr(attr_name, res);
        break;
      }
      ObtainCNodeAttrInScalarForm(attr_proto, &multi_value_map);
      break;
    }
    case FORM_PARSE_TENSOR: {
      ObtainCNodeAttrInTensorForm(prim, attr_proto);
      break;
    }
    default:
      MS_LOG(ERROR) << "parse attr type don't support the ref_attr_name: " << ref_attr_name;
      return false;
  }

  if (kParseTypeSwitchMap[type] == FORM_PARSE_SCALAR && multi_value_map.size() != 0) {
    if ((pos = ref_attr_name.find("Tuple")) != std::string::npos) {
      auto value_tuple_ptr = ParserScalarAttrValue<ValueTuple>(ref_attr_name, multi_value_map);
      prim->AddAttr(attr_name, value_tuple_ptr);
    } else {
      auto value_list_ptr = ParserScalarAttrValue<ValueList>(ref_attr_name, multi_value_map);
      prim->AddAttr(attr_name, value_list_ptr);
    }
  }
  return true;
}

bool MSANFModelParser::ObtainValueNodeInTensorForm(const std::string &value_node_name,
                                                   const mind_ir::TensorProto &attr_tensor) {
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

bool MSANFModelParser::ObtainValueNodeInTypeForm(const std::string &value_node_name,
                                                 const mind_ir::TensorProto &attr_tensor) {
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

bool MSANFModelParser::ObtainValueNodeInNoneForm(const std::string &value_node_name,
                                                 const mind_ir::AttributeProto &attr_proto) {
  auto new_value_node = NewValueNode(kNone);
  MS_EXCEPTION_IF_NULL(new_value_node);
  new_value_node->set_abstract(kNone->ToAbstract());
  anfnode_build_map_[value_node_name] = new_value_node;
  return true;
}

bool MSANFModelParser::ObtainValueNodeInMonadForm(const std::string &value_node_name,
                                                  const mind_ir::AttributeProto &attr_proto) {
  const std::string &ref_attr_name = attr_proto.ref_attr_name();
  if (ref_attr_name.find("UMonad") != std::string::npos) {
    const ValuePtr kUMonad = std::make_shared<UMonad>();
    auto monad_abs = kUMonad->ToAbstract();
    auto new_value_node = NewValueNode(kUMonad);
    MS_EXCEPTION_IF_NULL(new_value_node);
    new_value_node->set_abstract(monad_abs);
    anfnode_build_map_[value_node_name] = new_value_node;
  } else if (ref_attr_name.find("IOMonad") != std::string::npos) {
    const ValuePtr kIOMonad = std::make_shared<IOMonad>();
    auto monad_abs = kIOMonad->ToAbstract();
    auto new_value_node = NewValueNode(kIOMonad);
    MS_EXCEPTION_IF_NULL(new_value_node);
    new_value_node->set_abstract(monad_abs);
    anfnode_build_map_[value_node_name] = new_value_node;
  } else {
    return false;
  }
  return true;
}

bool MSANFModelParser::GetAttrValueForValueNode(const std::string &value_node_name,
                                                const mind_ir::AttributeProto &attr_proto) {
  if (!attr_proto.has_ref_attr_name()) {
    MS_LOG(ERROR) << "CNode parse attr type has no ref_attr_name";
    return false;
  }
  const std::string &ref_attr_name = attr_proto.ref_attr_name();
  string type = "";
  std::size_t pos(0);
  if ((pos = ref_attr_name.find("scalar:")) != std::string::npos) {
    type = ref_attr_name.substr(pos, string("scalar:").length() - 1);
  } else if ((pos = ref_attr_name.find("type:")) != std::string::npos) {
    type = ref_attr_name.substr(pos, string("type:").length() - 1);
  } else if ((pos = ref_attr_name.find("tensor:")) != std::string::npos) {
    type = ref_attr_name.substr(pos, string("tensor:").length() - 1);
  } else if ((pos = ref_attr_name.find("Monad:")) != std::string::npos) {
    type = ref_attr_name.substr(pos, string("Monad:").length() - 1);
  } else if (ref_attr_name == "none") {
    type = ref_attr_name;
  }

  ValueNodePtr new_value_node;
  std::unordered_map<std::string, ValuePtr> multi_value_map;
  switch (kParseTypeSwitchMap[type]) {
    case FORM_PARSE_TYPE: {
      ObtainValueNodeInTypeForm(value_node_name, attr_proto.tensors(0));
      break;
    }
    case FORM_PARSE_SCALAR: {
      std::size_t value_pos(0);
      if ((value_pos = ref_attr_name.find("value0")) != std::string::npos) {
        auto res = ObtainCNodeAttrInSingleScalarForm(attr_proto);
        new_value_node = NewValueNode(res);
        new_value_node->set_abstract(res->ToAbstract());
        anfnode_build_map_[value_node_name] = new_value_node;
        break;
      }
      if ((value_pos = ref_attr_name.find("Tuple[]")) != std::string::npos) {
        MS_LOG(INFO) << "Build Tuple() ValueNode for primitive.";
        ValuePtr res = MakeValue(std::vector<ValuePtr>{});
        new_value_node = NewValueNode(res);
        new_value_node->set_abstract(res->ToAbstract());
        anfnode_build_map_[value_node_name] = new_value_node;
        break;
      }
      ObtainCNodeAttrInScalarForm(attr_proto, &multi_value_map);
      break;
    }
    case FORM_PARSE_TENSOR: {
      ObtainValueNodeInTensorForm(value_node_name, attr_proto.tensors(0));
      break;
    }
    case FORM_PARSE_NONE: {
      ObtainValueNodeInNoneForm(value_node_name, attr_proto);
      break;
    }
    case FORM_PARSE_MONAD: {
      ObtainValueNodeInMonadForm(value_node_name, attr_proto);
      break;
    }
    default:
      MS_LOG(ERROR) << "parse attr type don't support the ref_attr_name: " << ref_attr_name;
      return false;
  }

  if (kParseTypeSwitchMap[type] == FORM_PARSE_SCALAR && multi_value_map.size() != 0) {
    if ((pos = ref_attr_name.find("Tuple")) != std::string::npos) {
      auto value_tuple_ptr = ParserScalarAttrValue<ValueTuple>(ref_attr_name, multi_value_map);
      new_value_node = NewValueNode(value_tuple_ptr);
      new_value_node->set_abstract(value_tuple_ptr->ToAbstract());
    } else {
      auto value_list_ptr = ParserScalarAttrValue<ValueList>(ref_attr_name, multi_value_map);
      new_value_node = NewValueNode(value_list_ptr);
      new_value_node->set_abstract(value_list_ptr->ToAbstract());
    }
    anfnode_build_map_[value_node_name] = new_value_node;
  }
  return true;
}

bool MSANFModelParser::BuildValueNodeForFuncGraph(const mind_ir::NodeProto &node_proto) {
  const std::string &value_node_name = node_proto.output(0);
  const mind_ir::AttributeProto &attr_proto = node_proto.attribute(0);
  if (!attr_proto.has_ref_attr_name()) {
    MS_LOG(ERROR) << "parse ValueNode  don't have ref_attr_name";
    return false;
  }
  return GetAttrValueForValueNode(value_node_name, attr_proto);
}

std::unordered_map<std::string, abstract::AbstractBasePtr> MSANFModelParser::GetAbstractForCNode(
  const mind_ir::AttributeProto &attr_proto) {
  std::unordered_map<std::string, abstract::AbstractBasePtr> kv;
  for (int i = 0; i < attr_proto.tensors_size(); ++i) {
    ShapeVector shape_vec;
    const mind_ir::TensorProto &attr_tensor = attr_proto.tensors(i);
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
                                                  const mind_ir::NodeProto &node_proto) {
  MS_EXCEPTION_IF_NULL(outputFuncGraph);
  if (!node_proto.has_op_type()) {
    MS_LOG(ERROR) << "Get CNode op_type failed!";
    return nullptr;
  }
  const std::string &node_name = node_proto.output(0);
  const std::string &fullname_with_scope = node_proto.domain();
  const std::string &node_type = node_proto.op_type();

  std::shared_ptr<Primitive> prim;
  auto op_primc_fns = ops::OpPrimCRegister::GetInstance().GetPrimCMap();
  if (op_primc_fns.find(node_type) != op_primc_fns.end()) {
    prim = op_primc_fns[node_type]();
  } else {
    prim = std::make_shared<Primitive>(node_type);
    prim->set_instance_name(node_type);
  }
  MS_EXCEPTION_IF_NULL(prim);

  std::unordered_map<std::string, abstract::AbstractBasePtr> kv;
  string shape_ref_attr_name;
  for (int i = 0; i < node_proto.attribute_size(); ++i) {
    const mind_ir::AttributeProto &attr_proto = node_proto.attribute(i);
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
  for (int i = 0; i < node_proto.input_size(); ++i) {
    const std::string &input_name = node_proto.input(i);
    if (anfnode_build_map_.find(input_name) == anfnode_build_map_.end()) {
      MS_LOG(ERROR) << node_name << " input " << i << input_name << "can't find in nodes have parsed";
      return nullptr;
    }

    inputs.push_back(anfnode_build_map_[input_name]);
  }
  prim->set_attr("is_load", MakeValue(true));
  auto cnode_ptr = outputFuncGraph->NewCNode(prim, inputs);
  MS_EXCEPTION_IF_NULL(cnode_ptr);

  if (0 == kv.size()) {
    if (node_type == "UpdateState") {
      const ValuePtr kUMonad = std::make_shared<UMonad>();
      auto monad_abs = kUMonad->ToAbstract();
      cnode_ptr->set_abstract(monad_abs);
    } else {
      AbstractBasePtrList elem;
      for (size_t index = 1; index < cnode_ptr->inputs().size(); ++index) {
        elem.push_back(cnode_ptr->input(index)->abstract());
      }
      cnode_ptr->set_abstract(std::make_shared<abstract::AbstractTuple>(elem));
    }
  } else if (1 == kv.size()) {
    std::unordered_map<std::string, abstract::AbstractBasePtr>::iterator iter = kv.begin();
    cnode_ptr->set_abstract(iter->second);
  } else {
    auto abstract = ParserAttrShape(shape_ref_attr_name, kv);
    if (abstract == nullptr) {
      MS_LOG(ERROR) << "Node's attribute is nullptr.";
      return nullptr;
    }
    cnode_ptr->set_abstract(abstract);
  }

  string debug_info_name = ParseCNodeName(node_name);
  auto debug_info_ptr = std::make_shared<NodeDebugInfo>(debug_info_name);
  cnode_ptr->set_debug_info(debug_info_ptr);
  cnode_ptr->set_fullname_with_scope(fullname_with_scope);
  cnode_ptr->set_load_flag(true);

  anfnode_build_map_[node_name] = cnode_ptr;
  return cnode_ptr;
}

bool MSANFModelParser::BuildReturnForFuncGraph(const FuncGraphPtr &outputFuncGraph,
                                               const mind_ir::GraphProto &importProto, const CNodePtr &cnode_ptr) {
  MS_EXCEPTION_IF_NULL(outputFuncGraph);
  MS_EXCEPTION_IF_NULL(cnode_ptr);
  if (importProto.output_size() < 0 || importProto.output_size() > INT_MAX) {
    MS_LOG(ERROR) << "importProto.output_size is : " << importProto.output_size();
    return false;
  }
  std::vector<AnfNodePtr> inputs;
  if (importProto.output_size() > 1) {
    inputs.clear();
    inputs.push_back(NewValueNode(prim::kPrimMakeTuple));
    AbstractBasePtrList elem;
    for (int out_size = 0; out_size < importProto.output_size(); ++out_size) {
      const mind_ir::ValueInfoProto &output_node = importProto.output(out_size);
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
    return_node->set_load_flag(true);
    outputFuncGraph->set_return(return_node);
    MS_LOG(INFO) << "Construct funcgraph finined, all success.";
  } else {
    inputs.clear();
    inputs.push_back(NewValueNode(prim::kPrimReturn));
    inputs.push_back(cnode_ptr);
    auto return_node = outputFuncGraph->NewCNode(inputs);
    MS_EXCEPTION_IF_NULL(return_node);
    return_node->set_load_flag(true);
    outputFuncGraph->set_return(return_node);
    MS_LOG(INFO) << "Construct funcgraph finined, all success!";
  }
  return true;
}

bool MSANFModelParser::ImportNodesForGraph(const FuncGraphPtr &outputFuncGraph,
                                           const mind_ir::GraphProto &importProto) {
  MS_EXCEPTION_IF_NULL(outputFuncGraph);
  if (importProto.node_size() < 0 || importProto.node_size() > INT_MAX) {
    MS_LOG(ERROR) << "importProto.node_size is : " << importProto.node_size();
    return false;
  }
  MS_LOG(INFO) << "The CNdoe size : " << importProto.node_size();
  CNodePtr cnode_ptr = nullptr;
  for (int i = 0; i < importProto.node_size(); ++i) {
    const mind_ir::NodeProto &node_proto = importProto.node(i);
    const std::string &node_type = node_proto.op_type();
    if (node_type == kConstantValueNode) {
      if (!BuildValueNodeForFuncGraph(node_proto)) {
        MS_LOG(ERROR) << "Build ValueNode for funcgraph fail at index: " << i;
        return false;
      }
      continue;
    }
    cnode_ptr = BuildCNodeForFuncGraph(outputFuncGraph, node_proto);
    if (cnode_ptr == nullptr) {
      MS_LOG(ERROR) << "Build CNode for funcgraph fail at index: " << i;
      return false;
    }
  }

  BuildReturnForFuncGraph(outputFuncGraph, importProto, cnode_ptr);
  return true;
}

bool MSANFModelParser::BuildFuncGraph(const FuncGraphPtr &outputFuncGraph, const mind_ir::GraphProto &importProto) {
  MS_EXCEPTION_IF_NULL(outputFuncGraph);
  GraphDebugInfoPtr debug_info_ptr = outputFuncGraph->debug_info();
  MS_EXCEPTION_IF_NULL(debug_info_ptr);
  if (importProto.has_name()) {
    debug_info_ptr->set_name(importProto.name());
  } else {
    MS_LOG(ERROR) << "FuncGraph under converting has not name!";
  }

  if (!ImportParametersForGraph(outputFuncGraph, importProto)) {
    MS_LOG(ERROR) << "import parameters for graph fail!";
    return false;
  }
  return ImportNodesForGraph(outputFuncGraph, importProto);
}

bool MSANFModelParser::MSANFParseModelConfigureInfo(const mind_ir::ModelProto &model_proto) {
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

FuncGraphPtr MSANFModelParser::Parse(const mind_ir::ModelProto &model_proto) {
  FuncGraphPtr dstGraph = std::make_shared<FuncGraph>();
  MS_EXCEPTION_IF_NULL(dstGraph);
  if (!MSANFParseModelConfigureInfo(model_proto)) {
    MS_LOG(ERROR) << "Parse configuration info for pb file failed!";
  }
  const mind_ir::GraphProto &graphBuild = model_proto.graph();
  if (!BuildFuncGraph(dstGraph, graphBuild)) {
    MS_LOG(ERROR) << "Build funcgraph failed!";
    return nullptr;
  }
  MS_LOG(INFO) << "Parse pb to build FuncGraph Success!";
  return dstGraph;
}
}  // namespace mindspore

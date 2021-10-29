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
#include <climits>
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
std::map<std::string, tensor::TensorPtr> MSANFModelParser::load_tensor_map_;
static constexpr char kConstantValueNode[] = "Constant";
static constexpr char kCNodeShapeAttr[] = "shape";
static constexpr char kCNodeShape1Attr[] = "shape1";
static constexpr char kCNodeShape2Attr[] = "shape2";
static constexpr char kDoSignaturePrimitivePrefix[] = "S-Prim-";
static constexpr char kHyperMapPrefix[] = "hyper_map";

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
      rules.push(std::string("["));
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
        auto value_name = str.substr(static_cast<int>(i) - count + 1, count);
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
  MS_LOG(DEBUG) << "Load parameter name: " << debug_info_name;
  if (!IsIncLoad() || load_tensor_map_.find(debug_info_name) == load_tensor_map_.end()) {
    load_tensor_map_[debug_info_name] = tensor_info;
  } else {
    MS_LOG(DEBUG) << "Parameter: " << debug_info_name << " has been already loaded, use it again.";
    tensor::TensorPtr load_tensor_info = load_tensor_map_[debug_info_name];
    auto tensor_abstract = load_tensor_info->ToAbstract();
    MS_EXCEPTION_IF_NULL(tensor_abstract);
    node->set_abstract(tensor_abstract);
    node->set_default_param(load_tensor_info);
    anfnode_build_map_[parameter_proto.name()] = node;
    return true;
  }
  ParamInfoPtr param_info = std::make_shared<ParamInfo>();
  param_info->set_name(debug_info_name);
  tensor_info->set_param_info(param_info);

  auto tensor_abstract = tensor_info->ToAbstract();
  MS_EXCEPTION_IF_NULL(tensor_abstract);
  node->set_abstract(tensor_abstract);

  std::string initial_data = parameter_proto.raw_data();
  auto *tensor_data_buf = reinterpret_cast<uint8_t *>(tensor_info->data_c());
  MS_EXCEPTION_IF_NULL(tensor_data_buf);
  auto ret = memcpy_s(tensor_data_buf, static_cast<size_t>(tensor_info->data().nbytes()), initial_data.data(),
                      initial_data.size());
  if (ret != 0) {
    MS_LOG(ERROR) << "Build parameter occur memcpy_s error.";
    return false;
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

  // Set abstract of the parameter
  if (value_proto.tensor_size() > 0) {
    const mind_ir::TensorProto &tensor_proto = value_proto.tensor(0);
    tensor::TensorPtr tensor_info = BuildTensorInfoForFuncGraph(tensor_proto);
    MS_EXCEPTION_IF_NULL(tensor_info);
    auto tensor_abstract = tensor_info->ToAbstract();
    node->set_abstract(tensor_abstract);
  } else if (value_proto.has_denotation()) {
    MS_LOG(DEBUG) << "Not tensor. parameter type: " << value_proto.denotation();
  }
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
    case mind_ir::AttributeProto_AttributeType_TENSORS: {
      const int attr_tensor_type = attr_proto.tensors(index).data_type();
      if (kDefaultValueSwitchMap.find(attr_tensor_type) == kDefaultValueSwitchMap.end()) {
        MS_LOG(ERROR) << "Obtain attr in type-form has not support input type:" << attr_tensor_type;
        return {};
      }
      return TypeIdToType(kDefaultValueSwitchMap[attr_tensor_type]);
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
  auto func = [&name, &multi_value_map, this](const mind_ir::AttributeProto &attr_proto, int i) -> void {
    auto res = this->ParseAttrInScalarForm(attr_proto, i);
    name = "value" + std::to_string(i + 1);
    (void)multi_value_map->emplace(name, res);
  };
  for (int i = 0; i < attr_proto.ints_size(); i++) {
    func(attr_proto, i);
  }
  for (int i = 0; i < attr_proto.doubles_size(); i++) {
    func(attr_proto, i);
  }
  for (int i = 0; i < attr_proto.floats_size(); i++) {
    func(attr_proto, i);
  }
  for (int i = 0; i < attr_proto.strings_size(); i++) {
    func(attr_proto, i);
  }
  for (int i = 0; i < attr_proto.tensors_size(); i++) {
    func(attr_proto, i);
  }
}

ValuePtr MSANFModelParser::ObtainCNodeAttrInSingleScalarForm(const mind_ir::AttributeProto &attr_proto) const {
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
  MS_EXCEPTION_IF_NULL(tensor_info);
  const std::string &tensor_buf = attr_tensor.raw_data();
  auto *tensor_data_buf = reinterpret_cast<uint8_t *>(tensor_info->data_c());
  auto ret = memcpy_s(tensor_data_buf, tensor_info->data().nbytes(), tensor_buf.data(), tensor_buf.size());
  if (ret != 0) {
    MS_LOG(ERROR) << "Obtain CNode in TensorForm occur memcpy_s error.";
    return false;
  }
  prim->AddAttr(attr_proto.name(), MakeValue(tensor_info));
  return true;
}

string GetTypeString(const std::string &ref_attr_name, size_t *pos) {
  if ((*pos = ref_attr_name.find("scalar:")) != std::string::npos) {
    return ref_attr_name.substr(*pos, string("scalar:").length() - 1);
  } else if ((*pos = ref_attr_name.find("type:")) != std::string::npos) {
    return ref_attr_name.substr(*pos, string("type:").length() - 1);
  } else if ((*pos = ref_attr_name.find("tensor:")) != std::string::npos) {
    return ref_attr_name.substr(*pos, string("tensor:").length() - 1);
  }
  return "";
}

bool MSANFModelParser::GetAttrValueForCNode(const PrimitivePtr &prim, const mind_ir::AttributeProto &attr_proto) {
  MS_EXCEPTION_IF_NULL(prim);
  const std::string &attr_name = attr_proto.name();
  if (!attr_proto.has_ref_attr_name()) {
    MS_LOG(ERROR) << "CNode parse attr type has no ref_attr_name";
    return false;
  }
  const std::string &ref_attr_name = attr_proto.ref_attr_name();

  std::size_t pos(0);
  string type = GetTypeString(ref_attr_name, &pos);
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
        if (op_type == "HistogramFixedWidth" && attr_name == "dtype" && res->isa<StringImm>()) {
          auto str_dtype = GetValue<std::string>(res);
          if (str_dtype == "int32") {
            const int64_t attr_value = 3;
            (void)prim->AddAttr(attr_name, MakeValue<int64_t>(attr_value));
            break;
          }
          MS_EXCEPTION(NotSupportError)
            << "The primtive[HistogramFixedWidth] not supported only support attribute[dtype] is 'int32',but got"
            << res->ToString();
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
  MS_EXCEPTION_IF_NULL(tensor_info);
  const std::string &tensor_buf = attr_tensor.raw_data();
  auto *tensor_data_buf = reinterpret_cast<uint8_t *>(tensor_info->data_c());
  auto ret =
    memcpy_s(tensor_data_buf, static_cast<size_t>(tensor_info->data().nbytes()), tensor_buf.data(), tensor_buf.size());
  if (ret != 0) {
    MS_LOG(ERROR) << "Obtain ValueNode in TensorForm occur memcpy_s error.";
    return false;
  }

  auto new_value_node = NewValueNode(MakeValue(tensor_info));
  MS_EXCEPTION_IF_NULL(new_value_node);
  auto tensor_abstract = tensor_info->ToAbstract();
  MS_EXCEPTION_IF_NULL(tensor_abstract);
  new_value_node->set_abstract(tensor_abstract);
  anfnode_build_map_[value_node_name] = new_value_node;
  return true;
}

bool MSANFModelParser::ObtainValueNodeInTupleTensorForm(const std::string &value_node_name,
                                                        const mind_ir::AttributeProto &attr_proto) {
  std::vector<tensor::TensorPtr> tensor_vec;
  for (int i = 0; i < attr_proto.tensors_size(); ++i) {
    mind_ir::TensorProto attr_tensor = attr_proto.tensors(i);
    const int attr_tensor_type = attr_tensor.data_type();
    ShapeVector shape;
    for (int j = 0; j < attr_tensor.dims_size(); ++j) {
      shape.push_back(attr_tensor.dims(j));
    }
    tensor::TensorPtr tensor_info = std::make_shared<tensor::Tensor>(kDefaultValueSwitchMap[attr_tensor_type], shape);
    const std::string &tensor_buf = attr_tensor.raw_data();
    auto *tensor_data_buf = reinterpret_cast<uint8_t *>(tensor_info->data_c());
    auto ret = memcpy_s(tensor_data_buf, static_cast<size_t>(tensor_info->data().nbytes()), tensor_buf.data(),
                        tensor_buf.size());
    if (ret != 0) {
      MS_LOG(ERROR) << "Obtain ValueNode in TupleTensorForm occur memcpy_s error.";
      return false;
    }
    tensor_vec.push_back(tensor_info);
  }
  auto new_value_node = NewValueNode(MakeValue(tensor_vec));
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

bool MSANFModelParser::ObtainValueNodeInNoneForm(const std::string &value_node_name) {
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
    auto monad_abs = kUMonad->ToAbstract();
    auto new_value_node = NewValueNode(kUMonad);
    MS_EXCEPTION_IF_NULL(new_value_node);
    new_value_node->set_abstract(monad_abs);
    anfnode_build_map_[value_node_name] = new_value_node;
  } else if (ref_attr_name.find("IOMonad") != std::string::npos) {
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

namespace {
std::string GetTypeFromAttrName(const std::string &ref_attr_name) {
  string type = "";
  std::size_t pos(0);
  if ((pos = ref_attr_name.find("scalar:")) != std::string::npos) {
    return ref_attr_name.substr(pos, string("scalar:").length() - 1);
  } else if ((pos = ref_attr_name.find("type:")) != std::string::npos) {
    return ref_attr_name.substr(pos, string("type:").length() - 1);
  } else if ((pos = ref_attr_name.find("tensor:")) != std::string::npos) {
    return ref_attr_name.substr(pos, string("tensor:").length() - 1);
  } else if ((pos = ref_attr_name.find("Monad:")) != std::string::npos) {
    return ref_attr_name.substr(pos, string("Monad:").length() - 1);
  } else if (ref_attr_name == "none") {
    return ref_attr_name;
  }
  return type;
}
}  // namespace

bool MSANFModelParser::GetAttrValueForValueNode(const std::string &value_node_name,
                                                const mind_ir::AttributeProto &attr_proto) {
  if (!attr_proto.has_ref_attr_name()) {
    MS_LOG(ERROR) << "CNode parse attr type has no ref_attr_name";
    return false;
  }
  const std::string &ref_attr_name = attr_proto.ref_attr_name();
  auto type = GetTypeFromAttrName(ref_attr_name);
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
      if ((value_pos = ref_attr_name.find("Tuple[value")) != std::string::npos && attr_proto.tensors_size() > 1) {
        MS_LOG(INFO) << "Build TupleTensor ValueNode for primitive.";
        if (!ObtainValueNodeInTupleTensorForm(value_node_name, attr_proto)) {
          MS_LOG(ERROR) << "Obtain valuenode in tuple tensor Form failed. ";
          return false;
        }
        break;
      }
      ObtainCNodeAttrInScalarForm(attr_proto, &multi_value_map);
      break;
    }
    case FORM_PARSE_TENSOR: {
      (void)ObtainValueNodeInTensorForm(value_node_name, attr_proto.tensors(0));
      break;
    }
    case FORM_PARSE_NONE: {
      (void)ObtainValueNodeInNoneForm(value_node_name);
      break;
    }
    case FORM_PARSE_MONAD: {
      (void)ObtainValueNodeInMonadForm(value_node_name, attr_proto);
      break;
    }
    default:
      MS_LOG(ERROR) << "parse attr type don't support the ref_attr_name: " << ref_attr_name;
      return false;
  }

  if (kParseTypeSwitchMap[type] == FORM_PARSE_SCALAR && multi_value_map.size() != 0) {
    if (ref_attr_name.find("Tuple") != std::string::npos) {
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
    (void)kv.emplace(attr_tensor.name(), abstract);
  }
  return kv;
}

// S-Prim-xxx or S-Prim-hyper_map[xxx] -> xxx
static std::string GetDoSignaturePrimitiveName(const std::string &node_type) {
  // Remove `S-Prim-` prefix.
  auto prim_name = node_type.substr(strlen(kDoSignaturePrimitivePrefix));
  if (prim_name.compare(0, strlen(kHyperMapPrefix), kHyperMapPrefix) != 0) {
    return prim_name;
  }
  // hyper_map[xxx] -> xxx
  constexpr auto offset = 2;
  auto op_name = prim_name.substr(strlen(kHyperMapPrefix) + 1, prim_name.length() - strlen(kHyperMapPrefix) - offset);
  return op_name;
}

AnfNodePtr MSANFModelParser::BuildOperatorNode(const mind_ir::NodeProto &node_proto) {
  const std::string kOperatorTypeFlag = std::string("REF::");
  const size_t kOpTypeFlagSize = kOperatorTypeFlag.length();
  const std::string &node_type = node_proto.op_type();
  MS_LOG(DEBUG) << "Process Operator :" << node_type;
  // Operator maybe CNode,FuncGraph or Parameter.

  if (node_type.size() > kOpTypeFlagSize && node_type.substr(0, kOpTypeFlagSize) == kOperatorTypeFlag) {
    auto anfNode = GetAnfNode(node_type.substr(kOpTypeFlagSize));
    if (anfNode == nullptr) {
      MS_LOG(EXCEPTION) << "Can't find the ref:" << node_type;
    }
    return anfNode;
  }

  // Operator is  primitive.
  std::shared_ptr<Primitive> prim;
  auto op_primc_fns = ops::OpPrimCRegister::GetInstance().GetPrimCMap();
  if (op_primc_fns.find(node_type) != op_primc_fns.end()) {
    prim = op_primc_fns[node_type]();
  } else {
    if (node_type.compare(0, strlen(kDoSignaturePrimitivePrefix), kDoSignaturePrimitivePrefix) == 0) {
      auto op_name = GetDoSignaturePrimitiveName(node_type);
      prim = std::make_shared<prim::DoSignaturePrimitive>(op_name, std::make_shared<Primitive>(op_name));
      MS_EXCEPTION_IF_NULL(prim);
      prim->set_instance_name(op_name);
    } else {
      MS_LOG(DEBUG) << "Special node_type: " << node_type;
      prim = std::make_shared<Primitive>(node_type);
      MS_EXCEPTION_IF_NULL(prim);
      prim->set_instance_name(node_type);
    }
  }
  MS_EXCEPTION_IF_NULL(prim);
  for (int i = 0; i < node_proto.attribute_size(); ++i) {
    const mind_ir::AttributeProto &attr_proto = node_proto.attribute(i);
    // CNode abstract
    if (attr_proto.ref_attr_name().find("shape:") != string::npos) {
      continue;
    }
    if (!GetAttrValueForCNode(prim, attr_proto)) {
      MS_LOG(EXCEPTION) << "Parser prim: " << node_type << " attributes error : " << attr_proto.DebugString();
    }
  }
  prim->set_attr("is_load", MakeValue(true));
  return std::make_shared<ValueNode>(prim);
}

// Set CNode abstract.
void MSANFModelParser::SetCNodeAbastract(const mind_ir::NodeProto &node_proto, CNodePtr cnode_ptr) {
  const std::string &node_type = node_proto.op_type();
  // Handle control flow operator.
  auto operatorPtr = cnode_ptr->input(0);
  // Set abstract of switch(c,f,t),switchLayer(c,tup) and
  // partial(func,args) to null
  auto prim = GetValueNode<PrimitivePtr>(operatorPtr);
  if (IsPrimitiveEquals(prim::kPrimSwitch, prim) || IsPrimitiveEquals(prim::kPrimSwitchLayer, prim) ||
      IsPrimitiveEquals(prim::kPrimPartial, prim)) {
    cnode_ptr->set_abstract(nullptr);
    return;
  }

  // If the operator is not a primitive, the abstract will been set to null.
  // Because there are not some operators in front end, the abstract of primitive should be reserved.
  if (prim == nullptr) {
    cnode_ptr->set_abstract(nullptr);
    return;
  }

  std::unordered_map<std::string, abstract::AbstractBasePtr> kv;
  string shape_ref_attr_name;

  for (int i = 0; i < node_proto.attribute_size(); ++i) {
    const mind_ir::AttributeProto &attr_proto = node_proto.attribute(i);
    if (attr_proto.ref_attr_name().find("shape:") != string::npos) {
      shape_ref_attr_name = attr_proto.ref_attr_name();
      kv = GetAbstractForCNode(attr_proto);
      break;
    }
  }

  // Because there is not context in unit test,
  // abstract->broaden() is replaced by abstract->set_value(kAnyValue).
  if (kv.size() == 0) {
    if (node_type == "UpdateState") {
      cnode_ptr->set_abstract(kUMonad->ToAbstract());
    } else if (node_type == "Depend") {
      cnode_ptr->set_abstract(kBool->ToAbstract());
    } else {
      AbstractBasePtrList elem;
      for (size_t index = 1; index < cnode_ptr->inputs().size(); ++index) {
        auto abs = cnode_ptr->input(index)->abstract();
        if (abs != nullptr) {
          if (abs->GetValueTrack() == nullptr) {
            abs->set_value(kAnyValue);
          }
          elem.push_back(abs);
        }
      }
      if (!elem.empty()) {
        cnode_ptr->set_abstract(std::make_shared<abstract::AbstractTuple>(elem));
      }
    }
  } else if (kv.size() == 1) {
    std::unordered_map<std::string, abstract::AbstractBasePtr>::iterator iter = kv.begin();
    if (iter->second != nullptr) {
      iter->second->set_value(kAnyValue);
      cnode_ptr->set_abstract(iter->second);
    }
  } else {
    auto abstract = ParserAttrShape(shape_ref_attr_name, kv);
    if (abstract == nullptr) {
      cnode_ptr->set_abstract(nullptr);
      MS_LOG(ERROR) << "Node's attribute is nullptr.";
    } else {
      abstract->set_value(kAnyValue);
      cnode_ptr->set_abstract(abstract);
    }
  }
}

CNodePtr MSANFModelParser::BuildCNodeForFuncGraph(const FuncGraphPtr &outputFuncGraph,
                                                  const mind_ir::NodeProto &node_proto) {
  MS_EXCEPTION_IF_NULL(outputFuncGraph);
  if (!node_proto.has_op_type()) {
    MS_LOG(ERROR) << "Get CNode op_type failed!";
    return nullptr;
  }
  const std::string &node_name = node_proto.output(0);
  MS_LOG(DEBUG) << "Process CNode: " << node_name;
  // Build inputs.
  std::vector<AnfNodePtr> inputs;
  inputs.push_back(BuildOperatorNode(node_proto));
  for (int i = 0; i < node_proto.input_size(); ++i) {
    auto anfNode = GetAnfNode(node_proto.input(i));
    if (anfNode == nullptr) {
      MS_LOG(ERROR) << node_name << " input " << i << node_proto.input(i) << "can't find in nodes have parsed";
      return nullptr;
    }
    inputs.push_back(anfNode);
  }

  CNodePtr cnode_ptr = outputFuncGraph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(cnode_ptr);
  SetCNodeAbastract(node_proto, cnode_ptr);

  const std::string &fullname_with_scope = node_proto.domain();
  string debug_info_name = ParseCNodeName(node_name);
  auto debug_info_ptr = std::make_shared<NodeDebugInfo>(debug_info_name);
  cnode_ptr->set_debug_info(debug_info_ptr);
  cnode_ptr->set_fullname_with_scope(fullname_with_scope);
  cnode_ptr->set_load_flag(true);
  if (anfnode_build_map_.count(node_name) > 0) {
    MS_LOG(EXCEPTION) << "Duplicate CNode name: " << node_name;
  }
  anfnode_build_map_[node_name] = cnode_ptr;
  return cnode_ptr;
}

bool MSANFModelParser::BuildReturnForFuncGraph(const FuncGraphPtr &outputFuncGraph,
                                               const mind_ir::GraphProto &importProto) {
  MS_EXCEPTION_IF_NULL(outputFuncGraph);
  std::vector<AnfNodePtr> inputs;
  if (importProto.output_size() > 1) {
    inputs.push_back(NewValueNode(prim::kPrimMakeTuple));
    AbstractBasePtrList elem;
    for (int out_size = 0; out_size < importProto.output_size(); ++out_size) {
      const mind_ir::ValueInfoProto &output_node = importProto.output(out_size);
      const std::string &out_tuple = output_node.name();
      auto anfNode = GetAnfNode(out_tuple);
      if (anfNode == nullptr) {
        MS_LOG(ERROR) << "Miss return node: " << out_tuple;
        return false;
      }
      inputs.push_back(anfNode);
      elem.push_back(anfNode->abstract());
    }
    auto maketuple_ptr = outputFuncGraph->NewCNode(inputs);
    MS_EXCEPTION_IF_NULL(maketuple_ptr);
    maketuple_ptr->set_abstract(std::make_shared<abstract::AbstractTuple>(elem));
    inputs.push_back(NewValueNode(prim::kPrimReturn));
    inputs.push_back(maketuple_ptr);
    auto return_node = outputFuncGraph->NewCNode(inputs);
    MS_EXCEPTION_IF_NULL(return_node);
    return_node->set_load_flag(true);
    outputFuncGraph->set_return(return_node);
    MS_LOG(DEBUG) << "Construct funcgraph finined, all success.";
  } else {
    inputs.clear();
    inputs.push_back(NewValueNode(prim::kPrimReturn));
    auto nodeName = importProto.output(0).name();
    auto anfNode = GetAnfNode(nodeName);
    if (anfNode == nullptr) {
      MS_LOG(ERROR) << "Miss return node: " << nodeName;
      return false;
    }
    inputs.push_back(anfNode);
    auto return_node = outputFuncGraph->NewCNode(inputs);
    MS_EXCEPTION_IF_NULL(return_node);
    return_node->set_load_flag(true);
    outputFuncGraph->set_return(return_node);
    MS_LOG(DEBUG) << "Construct funcgraph finined, all success!";
  }
  return true;
}

bool MSANFModelParser::ImportNodesForGraph(const FuncGraphPtr &outputFuncGraph,
                                           const mind_ir::GraphProto &importProto) {
  MS_EXCEPTION_IF_NULL(outputFuncGraph);
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

  return BuildReturnForFuncGraph(outputFuncGraph, importProto);
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
  if (importProto.has_bprop_hash()) {
    outputFuncGraph->set_bprop_hash(importProto.bprop_hash());
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

  // Forward declare FuncGraph name
  // Compatible with the previous proto.
  if (graphBuild.has_name()) {
    anfnode_build_map_[graphBuild.name()] = std::make_shared<ValueNode>(dstGraph);
  }
  for (int i = 0; i < model_proto.functions_size(); ++i) {
    FuncGraphPtr graph = std::make_shared<FuncGraph>();
    const auto &graph_proto = model_proto.functions(i);
    if (!graph_proto.has_name()) {
      MS_LOG(EXCEPTION) << "The function has not a name. Please export mindIR again. ";
    }
    if (anfnode_build_map_.count(graph_proto.name()) > 0) {
      MS_LOG(EXCEPTION) << "There is a duplication function graph name: " << graph_proto.name();
    }
    anfnode_build_map_[graph_proto.name()] = std::make_shared<ValueNode>(graph);
  }

  // Parser the proto.
  if (!BuildFuncGraph(dstGraph, graphBuild)) {
    MS_LOG(ERROR) << "Build funcgraph failed!";
    return nullptr;
  }
  MS_LOG(DEBUG) << "Parse pb to build FuncGraph Success! " << graphBuild.name();
  for (int i = 0; i < model_proto.functions_size(); ++i) {
    const auto &graph_proto = model_proto.functions(i);
    FuncGraphPtr graph = GetValueNode<FuncGraphPtr>(anfnode_build_map_[graph_proto.name()]);
    if (!BuildFuncGraph(graph, graph_proto)) {
      MS_LOG(ERROR) << "Build funcgraph failed!";
      return nullptr;
    }
    MS_LOG(DEBUG) << "Parse pb to build FuncGraph Success! " << graph_proto.name();
  }
  // Release resource
  anfnode_build_map_.clear();
  return dstGraph;
}

AnfNodePtr MSANFModelParser::GetAnfNode(const std::string &node_name) {
  auto it = anfnode_build_map_.find(node_name);
  if (it == anfnode_build_map_.end()) {
    return nullptr;
  }
  FuncGraphPtr func_graph_ptr = GetValueNode<FuncGraphPtr>(it->second);
  if (func_graph_ptr) {
    return NewValueNode(func_graph_ptr);
  } else {
    return it->second;
  }
}
}  // namespace mindspore

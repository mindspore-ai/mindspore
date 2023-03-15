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

#include "load_mindir/anf_model_parser.h"
#include <climits>
#include <functional>
#include <map>
#include <memory>
#include <stack>
#include <string>
#include <vector>
#include <utility>
#include <fstream>
#include <algorithm>
#include "ir/tensor.h"
#include "ir/param_info.h"
#include "ir/map_tensor.h"
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "abstract/ops/primitive_infer_map.h"
#include "utils/hash_map.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/ms_utils_secure.h"
#include "abstract/abstract_function.h"
#include "load_mindir/infer_mindir.h"

using std::string;

namespace mindspore {
std::map<std::string, tensor::TensorPtr> MSANFModelParser::load_tensor_map_;
std::mutex MSANFModelParser::load_tensor_map_mutex_;
namespace {
static constexpr char kConstantValueNode[] = "Constant";
static constexpr char kDoSignaturePrimitivePrefix[] = "S-Prim-";
static constexpr char kQuantParam[] = "quant_param";

enum ParseForm : int {
  FORM_PARSE_TYPE = 0,
  FORM_PARSE_SCALAR = 1,
  FORM_PARSE_TENSOR = 2,
  FORM_PARSE_NONE = 3,
  FORM_PARSE_MONAD = 4,
  FORM_PARSE_SEQUENCE = 5,
  FORM_PARSE_UNDEFINE = 6,
};

static std::map<std::string, ParseForm> kParseTypeSwitchMap{
  {"type", FORM_PARSE_TYPE}, {"scalar", FORM_PARSE_SCALAR}, {"tensor", FORM_PARSE_TENSOR},
  {"none", FORM_PARSE_NONE}, {"Monad", FORM_PARSE_MONAD},   {"Sequence", FORM_PARSE_SEQUENCE}};

static mindspore::HashMap<int, TypeId> kDefaultValueSwitchMap{
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
  {mind_ir::TensorProto_DataType_COMPLEX64, kNumberTypeComplex64},
  {mind_ir::TensorProto_DataType_COMPLEX128, kNumberTypeComplex128}};

template <typename T, typename P>
std::shared_ptr<T> ParserAttr(const std::string &str, const mindspore::HashMap<string, P> &kv) {
  std::stack<std::string> rules;
  std::stack<P> value;
  size_t count = 0;
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
        auto value_name = str.substr((i - count) + 1, count);
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
std::shared_ptr<T> ParserScalarAttrValue(const std::string &attr_name, const mindspore::HashMap<string, ValuePtr> &kv) {
  std::string str = attr_name;
  auto replace = [&](const string &orgStr, const string &newStr) {
    std::string::size_type pos;
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
  auto result = ParserAttr<T, ValuePtr>(str, kv);
  return result;
}

std::shared_ptr<abstract::AbstractTuple> ParserAttrShape(
  const std::string &attr_name, const mindspore::HashMap<string, abstract::AbstractBasePtr> &kv) {
  std::string str = attr_name;
  auto replace = [&](const string &orgStr, const string &newStr) {
    std::string::size_type pos;
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

  auto result = ParserAttr<abstract::AbstractTuple, abstract::AbstractBasePtr>(str, kv);
  return result;
}

std::string ParseParameterName(const string &name) {
  string delimiter = ":";
  size_t pos;
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
    if (attr_proto.ints_size() > index) {                                                                 \
      auto value = static_cast<valuetype>(attr_proto.ints(index));                                        \
      return MakeValue<valuetype>(value);                                                                 \
    }                                                                                                     \
    MS_LOG(EXCEPTION) << "Parse MindIR attr failed.";                                                     \
  }                                                                                                       \
  ValuePtr ParseAttrInSingleScalar_##type##_##valuetype(const mind_ir::AttributeProto &attr_proto) {      \
    if (attr_proto.has_i()) {                                                                             \
      auto value = static_cast<valuetype>(attr_proto.i());                                                \
      return MakeValue<valuetype>(value);                                                                 \
    }                                                                                                     \
    MS_LOG(EXCEPTION) << "Parse MindIR attr failed.";                                                     \
  }

#define PARSE_MINDIR_ATTR_IN_SCALAR_FORM(type, valuetype)                                                 \
  ValuePtr ParseAttrInScalar_##type##_##valuetype(const mind_ir::AttributeProto &attr_proto, int index) { \
    if (attr_proto.type##s_size() > index) {                                                              \
      auto value = static_cast<valuetype>(attr_proto.type##s(index));                                     \
      return MakeValue<valuetype>(value);                                                                 \
    }                                                                                                     \
    MS_LOG(EXCEPTION) << "Parse MindIR attr failed.";                                                     \
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

ParseForm GetParseFormType(const std::string &ref_attr_name) {
  for (const auto &iter : kParseTypeSwitchMap) {
    if (ref_attr_name.find(iter.first) == 0) {
      return iter.second;
    }
  }
  return FORM_PARSE_UNDEFINE;
}

template <typename T>
AnfNodePtr NewValueNodeWithAbstract(const T &value) {
  auto node = NewValueNode(value);
  node->set_abstract(value->ToAbstract());
  return node;
}
}  // namespace
ValuePtr MSANFModelParser::GetValueFromAttributeProto(const mind_ir::AttributeProto &attr_proto) {
  auto attr_name = attr_proto.name();
  switch (attr_proto.type()) {
    case mind_ir::AttributeProto_AttributeType_TENSORS: {
      mind_ir::TensorProto tensor_proto = attr_proto.tensors(0);
      if (tensor_proto.has_raw_data()) {
        // For real tensor.
        tensor::TensorPtr tensor_info = GenerateTensorPtrFromTensorProto(tensor_proto);
        if (tensor_info == nullptr) {
          MS_LOG(ERROR) << "Failed to get the tensor for ValueNode.";
          return nullptr;
        }
        return tensor_info;
      } else if (tensor_proto.name() == kQuantParam) {
        auto quantization_param_vector = GenerateQuantizationParam(tensor_proto);
        if (!quantization_param_vector.empty()) {
          return quantization_param_vector[0];
        }
      } else {
        // For data type.
        const int attr_tensor_type = tensor_proto.data_type();
        auto iter = kDefaultValueSwitchMap.find(attr_tensor_type);
        if (iter == kDefaultValueSwitchMap.end()) {
          MS_LOG(ERROR) << "Obtain ValueNode attr in type-form has not support input type: " << attr_tensor_type;
          return nullptr;
        }
        return TypeIdToType(iter->second);
      }
      MS_LOG(ERROR) << "Failed to get the tensor for value.";
      return nullptr;
    }
    case mind_ir::AttributeProto_AttributeType_NONE: {
      return kNone;
    }
    case mind_ir::AttributeProto_AttributeType_TUPLE:
    case mind_ir::AttributeProto_AttributeType_LIST: {
      auto sequence_value = ObtainValueInSequenceForm(attr_proto);
      if (sequence_value == nullptr) {
        MS_LOG(ERROR) << "Failed to get sequence value for " << attr_name;
        return nullptr;
      }
      return sequence_value;
    }
    case mind_ir::AttributeProto_AttributeType_DICT: {
      auto dict_value = ObtainValueInDictionaryForm(attr_proto);
      if (dict_value == nullptr) {
        MS_LOG(ERROR) << "Failed to get dictionary value for " << attr_name;
        return nullptr;
      }
      return dict_value;
    }
    default: {
      ValuePtr value = ObtainCNodeAttrInSingleScalarForm(attr_proto);
      if (value == nullptr) {
        MS_LOG(ERROR) << "Can not get the value for attr: " << attr_name;
        return nullptr;
      }
      return value;
    }
  }
  return nullptr;
}

tensor::TensorPtr MSANFModelParser::GetIncTensor(const std::string &tensor_name) {
  std::lock_guard<std::mutex> lock(load_tensor_map_mutex_);
  auto it = load_tensor_map_.find(tensor_name);
  if (it != load_tensor_map_.end()) {
    return it->second;
  }
  return nullptr;
}

void MSANFModelParser::SetIncTensor(const std::string &tensor_name, const tensor::TensorPtr &tensor) {
  std::lock_guard<std::mutex> lock(load_tensor_map_mutex_);
  load_tensor_map_[tensor_name] = tensor;
}

void MSANFModelParser::LoadTensorMapClear() {
  std::lock_guard<std::mutex> lock(load_tensor_map_mutex_);
  load_tensor_map_.clear();
}

tensor::TensorPtr MSANFModelParser::GenerateTensorPtrFromTensorProto(const mind_ir::TensorProto &attr_tensor) {
  ShapeVector shape;
  const int attr_tensor_type = attr_tensor.data_type();
  for (int i = 0; i < attr_tensor.dims_size(); ++i) {
    shape.push_back(attr_tensor.dims(i));
  }
  tensor::TensorPtr tensor = nullptr;
  if (!attr_tensor.has_compression_type() ||
      attr_tensor.compression_type() == mind_ir::TensorProto_CompressionType_NO_COMPRESSION) {
    tensor = std::make_shared<tensor::Tensor>(kDefaultValueSwitchMap[attr_tensor_type], shape);
  } else {
    auto compression_type = static_cast<TensorCompressionType>(static_cast<int>(attr_tensor.compression_type()));
    size_t data_size = 0;
    if (!attr_tensor.has_external_data()) {
      data_size = attr_tensor.raw_data().size();
    } else {
      data_size = attr_tensor.external_data().length();
    }
    tensor =
      std::make_shared<tensor::Tensor>(kDefaultValueSwitchMap[attr_tensor_type], shape, data_size, compression_type);
  }

  auto quantization_param_vector = GenerateQuantizationParam(attr_tensor);
  if (!quantization_param_vector.empty()) {
    tensor->set_quant_param(quantization_param_vector);
  }
  if (!IsIncLoad() || !GetIncTensor(attr_tensor.name())) {
    SetIncTensor(attr_tensor.name(), tensor);
  }

  MS_EXCEPTION_IF_NULL(tensor);
  const std::string &tensor_buf = attr_tensor.raw_data();
  if (attr_tensor.has_raw_data() && tensor->data().nbytes() != 0) {
    auto *tensor_data_buf = reinterpret_cast<uint8_t *>(tensor->data_c());
    errno_t ret = memcpy_s(tensor_data_buf, tensor->data().nbytes(), tensor_buf.data(), tensor_buf.size());
    if (ret != EOK) {
      MS_LOG(ERROR) << "Failed to copy data from tensor proto.";
      return nullptr;
    }
  } else if (attr_tensor.has_external_data()) {
    auto ret = GetTensorDataFromExternal(attr_tensor, tensor);
    if (!ret) {
      MS_LOG(ERROR) << "Failed to get external data from tensor proto.";
      return nullptr;
    }
  } else {
    MS_LOG(DEBUG) << "Parameter will load initialized data.";
  }
  return tensor;
}

std::vector<std::shared_ptr<mindspore::QuantizationParam>> MSANFModelParser::GenerateQuantizationParam(
  const mind_ir::TensorProto &attr_tensor) {
  auto quant_param_proto = attr_tensor.quant_params();
  std::vector<std::shared_ptr<mindspore::QuantizationParam>> quantization_param_vector;
  for (int i = 0; i < quant_param_proto.size(); i++) {
    auto quant_data = quant_param_proto.Get(i);
    QuantizationParam quantization_param(quant_data.quant_algo_name());
    for (int index = 0; index < quant_data.attribute_size(); index++) {
      auto quant_attr_proto = quant_data.attribute().Get(index);
      if (quant_attr_proto.type() != mind_ir::AttributeProto_AttributeType_LIST) {
        MS_LOG(ERROR) << "quant_attr_proto.type is " << quant_attr_proto.type()
                      << ", is should be mind_ir::AttributeProto_AttributeType_LIST ("
                      << mind_ir::AttributeProto_AttributeType_LIST << ")";
        return {};
      }
      auto sequence_value = ObtainValueInSequenceForm(quant_attr_proto);
      quantization_param.SetAttr(quant_attr_proto.name(), sequence_value);
    }
    quantization_param_vector.push_back(std::make_shared<mindspore::QuantizationParam>(quantization_param));
  }
  return quantization_param_vector;
}

abstract::AbstractBasePtr MSANFModelParser::GetNodeAbstractFromAttrProtoWithType(
  const mind_ir::AttributeProto &attr_proto) {
  switch (attr_proto.type()) {
    case mind_ir::AttributeProto_AttributeType_TENSORS: {
      const mind_ir::TensorProto &attr_tensor = attr_proto.tensors(0);
      return GetAbsTensorFromTensorProto(attr_tensor);
    }
    case mind_ir::AttributeProto_AttributeType_CSR_TENSOR: {
      return BuildAbstractCSRTensorFromAttrProto(attr_proto);
    }
    case mind_ir::AttributeProto_AttributeType_COO_TENSOR: {
      return BuildAbstractCOOTensorFromAttrProto(attr_proto);
    }
    case mind_ir::AttributeProto_AttributeType_MAP_TENSOR: {
      return BuildAbstractMapTensorFromAttrProto(attr_proto);
    }
    case mind_ir::AttributeProto_AttributeType_LIST:
    case mind_ir::AttributeProto_AttributeType_TUPLE: {
      return BuildAbstractSequence(attr_proto);
    }
    case mind_ir::AttributeProto_AttributeType_UMONAD: {
      return kUMonad->ToAbstract();
    }
    case mind_ir::AttributeProto_AttributeType_IOMONAD: {
      return kIOMonad->ToAbstract();
    }
    case mind_ir::AttributeProto_AttributeType_BOOL: {
      return kBool->ToAbstract();
    }
    case mind_ir::AttributeProto_AttributeType_NONE: {
      return kNone->ToAbstract();
    }
    case mind_ir::AttributeProto_AttributeType_FUNCGRAPHCLOSURE:
    case mind_ir::AttributeProto_AttributeType_PRIMITIVECLOSURE:
    case mind_ir::AttributeProto_AttributeType_PARTIALCLOSURE:
    case mind_ir::AttributeProto_AttributeType_UNIONFUNCCLOSURE: {
      return BuildAbstractFunction(attr_proto);
    }
    default: {
      MS_LOG(INFO) << "Not support to get the abstract from AttrProto type: " << attr_proto.type();
      return nullptr;
    }
  }
}

bool MSANFModelParser::SetNodeAbstractFromAttrProto(const mind_ir::AttributeProto &attr_proto,
                                                    const AnfNodePtr &node_ptr) {
  mindspore::HashMap<std::string, abstract::AbstractBasePtr> kv;
  string shape_ref_attr_name;
  if (attr_proto.ref_attr_name().find("shape:") == string::npos) {
    MS_LOG(ERROR) << "Cannot use a attr_proto " << attr_proto.ref_attr_name() << " to init shape.";
    return false;
  }

  shape_ref_attr_name = attr_proto.ref_attr_name();
  bool is_tuple_or_list =
    shape_ref_attr_name.find("Tuple[") != string::npos || shape_ref_attr_name.find("List[") != string::npos;
  kv = GetAbstractForNode(attr_proto);
  if (kv.empty()) {
    return SetEmptyTensorProtoCNodeAbstract(node_ptr);
  } else if (!is_tuple_or_list) {
    auto iter = kv.begin();
    if (iter->second != nullptr) {
      node_ptr->set_abstract(iter->second);
    }
  } else {
    auto abstract = ParserAttrShape(shape_ref_attr_name, kv);
    node_ptr->set_abstract(abstract);
    if (abstract == nullptr) {
      MS_LOG(ERROR) << "Node's attribute is nullptr.";
      return false;
    }
  }
  return true;
}

void MSANFModelParser::SetCNodePrimAttrAndAbstract(const mind_ir::NodeProto &node_proto, const CNodePtr &cnode_ptr) {
  auto prim_to_add_attr = GetCNodePrimitiveWithoutDoSignature(cnode_ptr);
  if (prim_to_add_attr != nullptr) {
    prim_to_add_attr->set_attr("is_load", MakeValue(true));
  }
  for (int i = 0; i < node_proto.attribute_size(); ++i) {
    const mind_ir::AttributeProto &attr_proto = node_proto.attribute(i);
    // Compatible with older versions.
    if (attr_proto.has_ref_attr_name()) {
      if (attr_proto.ref_attr_name().find("shape:") != string::npos) {
        SetCNodeAbstract(attr_proto, cnode_ptr);
        continue;
      }
      if (prim_to_add_attr != nullptr && !GetAttrValueForCNode(prim_to_add_attr, attr_proto)) {
        MS_LOG(ERROR) << "Parse prim: " << prim_to_add_attr->ToString()
                      << ", attributes error: " << attr_proto.DebugString();
      }
    } else {
      // ref_attr_name is removed in newer versions.
      if (attr_proto.name() == "shape") {
        SetCNodeAbstract(attr_proto, cnode_ptr);
        continue;
      }
      if (prim_to_add_attr != nullptr && !SetPrimitiveAttrWithType(prim_to_add_attr, attr_proto)) {
        MS_LOG(ERROR) << "Parse prim: " << prim_to_add_attr->ToString()
                      << ", attributes error: " << attr_proto.DebugString();
      }
    }
  }
}

abstract::AbstractTensorPtr MSANFModelParser::GetAbsTensorFromTensorProto(const mind_ir::TensorProto &tensor_proto) {
  ShapeVector shape;
  for (int i = 0; i < tensor_proto.dims_size(); ++i) {
    (void)shape.emplace_back(tensor_proto.dims(i));
  }

  if (!tensor_proto.has_data_type()) {
    MS_LOG(ERROR) << "mind_ir build tensor: " << tensor_proto.name() << " failed";
    MS_LOG(ERROR) << "mind_ir TensorProto has no data_type.";
    return nullptr;
  }
  auto iter = kDefaultValueSwitchMap.find(tensor_proto.data_type());
  if (iter == kDefaultValueSwitchMap.end()) {
    MS_LOG(ERROR) << "mind_ir build tensor: " << tensor_proto.name() << " failed";
    MS_LOG(ERROR) << "mind_ir TensorProto data_type: " << tensor_proto.data_type() << " is not support yet!";
    return nullptr;
  }
  auto tensor_shape = std::make_shared<abstract::Shape>(shape);
  auto tensor_info = std::make_shared<abstract::AbstractTensor>(TypeIdToType(iter->second), tensor_shape);
  if (tensor_proto.has_ref_key()) {
    auto ref_key = std::make_shared<RefKey>(tensor_proto.ref_key());
    auto abs_ref = std::make_shared<abstract::AbstractRefTensor>(tensor_info, ref_key);
    return abs_ref;
  }
  if (tensor_proto.has_name()) {
    tensor_info->set_name(tensor_proto.name());
  }
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

  ParamInfoPtr param_info = std::make_shared<ParamInfo>();
  param_info->set_name(debug_info_name);

  MS_LOG(DEBUG) << "Load parameter name: " << parameter_proto.name();
  if (IsIncLoad()) {
    auto load_tensor_info = GetIncTensor(parameter_proto.name());
    if (load_tensor_info) {
      MS_LOG(DEBUG) << "Parameter: " << parameter_proto.name() << " has been already loaded, use it again.";
      load_tensor_info->set_param_info(param_info);
      node->set_default_param(load_tensor_info);
      node->set_abstract(load_tensor_info->ToAbstract());
      anfnode_build_map_[parameter_proto.name()] = node;
      return true;
    }
  }
  auto tensor = GenerateTensorPtrFromTensorProto(parameter_proto);
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "Build tensor failed from the parameter proto.";
    return false;
  }
  tensor->set_param_info(param_info);
  node->set_default_param(tensor);
  node->set_abstract(tensor->ToAbstract());

  anfnode_build_map_[parameter_proto.name()] = node;
  return true;
}

abstract::AbstractCOOTensorPtr MSANFModelParser::BuildAbstractCOOTensorFromAttrProto(
  const mind_ir::AttributeProto &attr_proto) {
  std::vector<abstract::AbstractBasePtr> vec;
  for (int i = 0; i < attr_proto.values_size(); ++i) {
    auto abs = GetNodeAbstractFromAttrProtoWithType(attr_proto.values(i));
    if (abs == nullptr) {
      MS_LOG(WARNING) << "Failed to get the COOTensor's abstract from AttrProto. " << attr_proto.DebugString();
      return nullptr;
    }
    (void)vec.emplace_back(abs);
  }
  return std::make_shared<abstract::AbstractCOOTensor>(vec);
}

abstract::AbstractCSRTensorPtr MSANFModelParser::BuildAbstractCSRTensorFromAttrProto(
  const mind_ir::AttributeProto &attr_proto) {
  std::vector<abstract::AbstractBasePtr> vec;
  for (int i = 0; i < attr_proto.values_size(); ++i) {
    auto abs = GetNodeAbstractFromAttrProtoWithType(attr_proto.values(i));
    if (abs == nullptr) {
      MS_LOG(WARNING) << "Failed to get the CSRTensor's abstract from AttrProto. " << attr_proto.DebugString();
      return nullptr;
    }
    (void)vec.emplace_back(abs);
  }
  return std::make_shared<abstract::AbstractCSRTensor>(vec);
}

abstract::AbstractMapTensorPtr MSANFModelParser::BuildAbstractMapTensorFromAttrProto(
  const mind_ir::AttributeProto &attr_proto) {
  // default value
  if (attr_proto.values_size() != 1) {
    MS_LOG(EXCEPTION) << "AttrProto for AbstractMapTensor should has 1 value, but got " << attr_proto.values_size();
  }
  const auto &default_value_proto = attr_proto.values(0);
  auto default_value = ObtainCNodeAttrInSingleScalarForm(default_value_proto);
  MS_EXCEPTION_IF_NULL(default_value);

  constexpr size_t kAbstractMapTensorAttrProtoTensorsSize = 2;
  if (attr_proto.tensors_size() != kAbstractMapTensorAttrProtoTensorsSize) {
    MS_LOG(EXCEPTION) << "AttrProto for AbstractMapTensor should has 2 tensors, but got " << attr_proto.tensors_size();
  }
  // key tensor
  const auto &key_tensor_proto = attr_proto.tensors(0);
  auto key_tensor_abs = GetAbsTensorFromTensorProto(key_tensor_proto);
  MS_EXCEPTION_IF_NULL(key_tensor_abs);
  // value tensor
  const auto &value_tensor_proto = attr_proto.tensors(1);
  auto value_tensor_abs = GetAbsTensorFromTensorProto(value_tensor_proto);
  MS_EXCEPTION_IF_NULL(value_tensor_abs);
  auto value_build_shape_ptr = value_tensor_abs->BuildShape();
  if (!value_build_shape_ptr->isa<abstract::Shape>()) {
    MS_LOG(EXCEPTION) << "value_shape of AbstractMapTensor should be a Shape, but got "
                      << value_build_shape_ptr->ToString();
  }
  auto value_shape_ptr = value_build_shape_ptr->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(value_shape_ptr);
  auto map_tensor = std::make_shared<tensor::MapTensor>(key_tensor_abs->BuildType()->type_id(),
                                                        value_tensor_abs->BuildType()->type_id(),
                                                        value_shape_ptr->shape(), default_value);
  return std::make_shared<abstract::AbstractMapTensor>(map_tensor);
}

abstract::AbstractSequencePtr MSANFModelParser::BuildAbstractSequence(const mind_ir::AttributeProto &attr_proto) {
  std::vector<abstract::AbstractBasePtr> vec;

  for (int i = 0; i < attr_proto.values_size(); ++i) {
    auto abs = GetNodeAbstractFromAttrProtoWithType(attr_proto.values(i));
    if (abs == nullptr) {
      MS_LOG(WARNING) << "Failed to get the tuple's abstract from AttrProto. " << attr_proto.DebugString();
      return nullptr;
    }
    (void)vec.emplace_back(abs);
  }
  abstract::AbstractSequencePtr seq_abs;
  if (attr_proto.type() == mind_ir::AttributeProto_AttributeType_TUPLE) {
    seq_abs = std::make_shared<abstract::AbstractTuple>(vec);
  } else {
    seq_abs = std::make_shared<abstract::AbstractList>(vec);
  }
  if (attr_proto.has_seq_info()) {
    auto seq_info = attr_proto.seq_info();
    seq_abs->set_dynamic_len(seq_info.is_dyn_len());
    if (seq_info.has_tuple_elem_item()) {
      auto elem_proto = seq_info.tuple_elem_item();
      auto elem_abs = GetNodeAbstractFromAttrProtoWithType(elem_proto);
      seq_abs->set_dynamic_len_element_abs(elem_abs);
    }
  }
  return seq_abs;
}

bool MSANFModelParser::BuildMapParameterFromMapTensorProto(const ParameterPtr &node,
                                                           const mind_ir::MapTensorProto &map_parameter_proto) {
  MS_EXCEPTION_IF_NULL(node);

  if (!map_parameter_proto.has_name()) {
    MS_LOG(ERROR) << "mind_ir MapTensorProto has no name!";
    return false;
  }

  string debug_info_name = ParseParameterName(map_parameter_proto.name());
  auto debug_info_ptr = std::make_shared<NodeDebugInfo>(debug_info_name);
  node->set_debug_info(debug_info_ptr);
  node->set_name(debug_info_name);

  ParamInfoPtr param_info = std::make_shared<ParamInfo>();
  param_info->set_name(debug_info_name);

  MS_LOG(DEBUG) << "Load map parameter name: " << map_parameter_proto.name();
  if (IsIncLoad() && GetIncTensor(map_parameter_proto.name()) != nullptr) {
    MS_LOG(ERROR) << "MapParameter dose not support incremental loading, param_name: " << map_parameter_proto.name();
    return false;
  }

  // default value
  if (!map_parameter_proto.has_default_value()) {
    MS_LOG(ERROR) << "MapTensorProto should have default value: " << map_parameter_proto.name();
    return false;
  }
  const auto &default_value_proto = map_parameter_proto.default_value();
  auto default_value = BuildValueFromAttributeProto(default_value_proto);
  if (default_value == nullptr) {
    MS_LOG(ERROR) << "Build default value from AttributeProto failed.";
    return false;
  }
  // key tensor
  if (!map_parameter_proto.has_key_tensor()) {
    MS_LOG(ERROR) << "MapTensorProto should have key tensor: " << map_parameter_proto.name();
    return false;
  }
  const auto &key_tensor_proto = map_parameter_proto.key_tensor();
  auto key_tensor = GenerateTensorPtrFromTensorProto(key_tensor_proto);
  if (key_tensor == nullptr) {
    MS_LOG(ERROR) << "Generate key tensor from TensorProto failed.";
    return false;
  }
  // value tensor
  if (!map_parameter_proto.has_value_tensor()) {
    MS_LOG(ERROR) << "MapTensorProto should have value tensor: " << map_parameter_proto.name();
    return false;
  }
  const auto &value_tensor_proto = map_parameter_proto.value_tensor();
  auto value_tensor = GenerateTensorPtrFromTensorProto(value_tensor_proto);
  if (value_tensor == nullptr) {
    MS_LOG(ERROR) << "Generate value tensor from TensorProto failed.";
    return false;
  }
  // status tensor
  if (!map_parameter_proto.has_status_tensor()) {
    MS_LOG(ERROR) << "MapTensorProto should have status tensor: " << map_parameter_proto.name();
    return false;
  }
  const auto &status_tensor_proto = map_parameter_proto.status_tensor();
  auto status_tensor = GenerateTensorPtrFromTensorProto(status_tensor_proto);
  if (status_tensor == nullptr) {
    MS_LOG(ERROR) << "Generate status tensor from TensorProto failed.";
    return false;
  }

  auto map_tensor = std::make_shared<tensor::MapTensor>(key_tensor, value_tensor, status_tensor, default_value);
  map_tensor->set_param_info(param_info);
  node->set_default_param(map_tensor);
  node->set_abstract(map_tensor->ToAbstract());

  anfnode_build_map_[map_parameter_proto.name()] = node;
  return true;
}

bool MSANFModelParser::GetTensorDataFromExternal(const mind_ir::TensorProto &tensor_proto,
                                                 const tensor::TensorPtr &tensor_info) {
  if (!tensor_proto.has_external_data()) {
    return false;
  }
  const unsigned char *data = nullptr;
  auto it = tenor_data_.find(tensor_proto.external_data().location());
  if (it != tenor_data_.end()) {
    data = it->second.get();
  } else {
    std::string file = mindir_path_ + "/" + tensor_proto.external_data().location();
    if (mindir_dec_key_ != nullptr) {
      size_t plain_len;
      auto plain_data = Decrypt(&plain_len, file, mindir_dec_key_, mindir_key_size_, mindir_dec_mode_);
      if (plain_data == nullptr) {
        MS_LOG(ERROR) << "Decrypt MindIR file failed, please check the correctness of the dec_key or dec_mode.";
        return false;
      }
      data = plain_data.get();
      (void)tenor_data_.emplace(tensor_proto.external_data().location(), std::move(plain_data));
    } else {
      // Read file
      std::basic_ifstream<char> fid(file, std::ios::in | std::ios::binary);
      if (!fid) {
        MS_LOG(EXCEPTION) << "Open file '" << file << "' failed, please check the correct of the file.";
      }
      (void)fid.seekg(0, std::ios_base::end);
      size_t file_size = static_cast<size_t>(fid.tellg());
      fid.clear();
      (void)fid.seekg(0);
      auto plain_data = std::make_unique<char[]>(file_size);
      constexpr Byte is_little_endian = 1;
      constexpr int byte_order_index = 0;
      (void)fid.read(plain_data.get(), SizeToLong(file_size));
      fid.close();
      // if byte order is not same return false
      if ((plain_data[byte_order_index] == is_little_endian) ^ little_endian()) {
        MS_LOG(ERROR) << "The byte order of export MindIr device and load MindIr device is not same!";
        return false;
      }
      data = reinterpret_cast<const unsigned char *>(plain_data.get());
      (void)tenor_data_.emplace(tensor_proto.external_data().location(),
                                std::unique_ptr<Byte[]>(reinterpret_cast<Byte *>(plain_data.release())));
    }
  }
  auto *tensor_data_buf = reinterpret_cast<uint8_t *>(tensor_info->data_c());
  MS_EXCEPTION_IF_NULL(tensor_data_buf);
  MS_EXCEPTION_IF_NULL(data);

  if (tensor_info->data().nbytes() == 0 || tensor_proto.external_data().length() == 0) {
    // no need to copy data
    return true;
  }

  auto ret =
    common::huge_memcpy(tensor_data_buf, tensor_info->data().nbytes(), data + tensor_proto.external_data().offset(),
                        LongToSize(tensor_proto.external_data().length()));
  if (ret != EOK) {
    MS_LOG(ERROR) << "Build parameter occur memcpy_s error.";
    return false;
  }
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
    auto tensor_info = GetAbsTensorFromTensorProto(tensor_proto);
    if (tensor_info == nullptr) {
      MS_LOG(ERROR) << "Get tensor_info fail.";
      return false;
    }
    node->set_abstract(tensor_info);
    if (tensor_proto.has_ref_key() && top_graph_ != nullptr) {
      auto parameters = top_graph_->parameters();
      for (const auto &parameter : parameters) {
        auto parameter_abs = parameter->abstract();
        if (parameter_abs->isa<abstract::AbstractRefTensor>()) {
          auto parameter_abs_value = parameter_abs->cast<abstract::AbstractRefPtr>()->ref_key_value();
          auto ref_key_value = parameter_abs_value->cast<StringImmPtr>();
          if (ref_key_value != nullptr && ref_key_value->value() == tensor_proto.ref_key()) {
            node->set_default_param(parameter->cast<ParameterPtr>()->default_param());
            break;
          }
        }
      }
    }
  } else if (value_proto.has_denotation()) {
    if (value_proto.denotation() == "UMonadType") {
      node->set_abstract(kUMonad->ToAbstract());
    } else if (value_proto.denotation() == "IOMonadType") {
      node->set_abstract(kIOMonad->ToAbstract());
    }
    MS_LOG(DEBUG) << "Not tensor. parameter type: " << value_proto.denotation();
  }
  if (value_proto.has_attr_info()) {
    auto attr_proto = value_proto.attr_info();
    // Compatible with the previous proto.
    if (attr_proto.has_ref_attr_name()) {
      if (!SetNodeAbstractFromAttrProto(attr_proto, node)) {
        MS_LOG(ERROR) << "Failed to get abstract for input node " << node->name()
                      << " from proto:" << attr_proto.DebugString();
      }
    } else {
      auto abs = GetNodeAbstractFromAttrProtoWithType(attr_proto);
      if (abs == nullptr) {
        MS_LOG(ERROR) << "Failed to get abstract for input node " << node->name()
                      << " from attr_proto:" << attr_proto.DebugString();
      }
      node->set_abstract(abs);
    }
  }
  if (node->abstract() == nullptr) {
    MS_LOG(INFO) << "Failed to build abstract of node:" << node->name()
                 << " from ValueInfoProto:" << value_proto.DebugString();
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
  outputFuncGraph->set_fv_param_count(IntToSize(importProto.parameter_size()));
  return true;
}

bool MSANFModelParser::ImportMapParametersForGraph(const FuncGraphPtr &outputFuncGraph,
                                                   const mind_ir::GraphProto &importProto) {
  MS_EXCEPTION_IF_NULL(outputFuncGraph);
  MS_LOG(INFO) << "All MapParameters size is: " << importProto.map_parameter_size();
  for (int i = 0; i < importProto.map_parameter_size(); ++i) {
    const mind_ir::MapTensorProto &map_parameter_proto = importProto.map_parameter(i);
    if (!BuildMapParameterFromMapTensorProto(outputFuncGraph->add_parameter(), map_parameter_proto)) {
      MS_LOG(ERROR) << "Build map parameter for funcgraph fail at index: " << i;
      return false;
    }
  }
  outputFuncGraph->set_fv_param_count(IntToSize(importProto.parameter_size()));
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
  const int attr_type = static_cast<int>(attr_proto.type());
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
        return nullptr;
      }
      return TypeIdToType(kDefaultValueSwitchMap[attr_tensor_type]);
    }
    default:
      MS_LOG(ERROR) << "Obtain attr in scalar-form has not support input type: " << attr_type;
      return nullptr;
  }
}

void MSANFModelParser::ObtainCNodeAttrInScalarForm(const mind_ir::AttributeProto &attr_proto,
                                                   mindspore::HashMap<std::string, ValuePtr> *multi_value_map) {
  string name;
  auto func = [&name, &multi_value_map, this](const mind_ir::AttributeProto &attr_proto, int length) -> void {
    for (int i = 0; i < length; ++i) {
      auto res = this->ParseAttrInScalarForm(attr_proto, i);
      name = "value" + std::to_string(i + 1);
      (void)multi_value_map->emplace(name, res);
    }
  };
  func(attr_proto, attr_proto.ints_size());
  func(attr_proto, attr_proto.doubles_size());
  func(attr_proto, attr_proto.floats_size());
  func(attr_proto, attr_proto.strings_size());
  func(attr_proto, attr_proto.tensors_size());
}

ValuePtr MSANFModelParser::ObtainCNodeAttrInSingleScalarForm(const mind_ir::AttributeProto &attr_proto) {
  const int attr_type = static_cast<int>(attr_proto.type());
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
      return nullptr;
  }
}

bool MSANFModelParser::ObtainCNodeAttrInTensorForm(const PrimitivePtr &prim,
                                                   const mind_ir::AttributeProto &attr_proto) {
  MS_EXCEPTION_IF_NULL(prim);
  const mind_ir::TensorProto attr_tensor = attr_proto.tensors(0);
  auto tensor_info = GenerateTensorPtrFromTensorProto(attr_tensor);
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "Failed to get attr[" << attr_proto.name() << "] for node " << prim->ToString()
                  << " from the proto.";
    return false;
  }
  prim->AddAttr(attr_proto.name(), MakeValue(tensor_info));
  return true;
}

bool MSANFModelParser::SetPrimitiveAttrWithType(const PrimitivePtr &prim, const mind_ir::AttributeProto &attr_proto) {
  MS_EXCEPTION_IF_NULL(prim);
  const std::string &attr_name = attr_proto.name();
  auto value = GetValueFromAttributeProto(attr_proto);
  if (value == nullptr) {
    MS_LOG(ERROR) << "Failed to get value from proto.\n proto info:" << attr_proto.name();
    return false;
  }
  const std::string &op_type = prim->name();
  CheckAndConvertUtils::ConvertAttrValueInLoad(op_type, attr_name, &value);
  // Compatible with older versions.
  if (op_type == "HistogramFixedWidth" && attr_name == "dtype" && value->isa<StringImm>()) {
    auto str_dtype = GetValue<std::string>(value);
    if (str_dtype == "int32") {
      int64_t index = 3;
      (void)prim->AddAttr(attr_name, MakeValue<int64_t>(index));
    }
    MS_EXCEPTION(NotSupportError)
      << "The primtive[HistogramFixedWidth] not supported only support attribute[dtype] is 'int32',but got"
      << value->ToString();
  }
  (void)prim->AddAttr(attr_name, value);
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
  ParseForm type = GetParseFormType(ref_attr_name);
  mindspore::HashMap<std::string, ValuePtr> multi_value_map;
  switch (type) {
    case FORM_PARSE_TYPE: {
      ObtainCNodeAttrInTypeForm(prim, attr_proto);
      break;
    }
    case FORM_PARSE_SCALAR: {
      if (ref_attr_name.find("value0") != std::string::npos) {
        ValuePtr res = ObtainCNodeAttrInSingleScalarForm(attr_proto);
        const std::string &op_type = prim->name();
        CheckAndConvertUtils::ConvertAttrValueInLoad(op_type, attr_name, &res);
        if (op_type == "HistogramFixedWidth" && attr_name == "dtype" && res->isa<StringImm>()) {
          auto str_dtype = GetValue<std::string>(res);
          if (str_dtype == "int32") {
            int64_t index = 3;
            (void)prim->AddAttr(attr_name, MakeValue<int64_t>(index));
            break;
          }
          MS_EXCEPTION(NotSupportError)
            << "The primtive[HistogramFixedWidth] not supported only support attribute[dtype] is 'int32',but got"
            << res->ToString();
        }
        (void)prim->AddAttr(attr_name, res);
        break;
      } else if (ref_attr_name.find("Tuple[]") != std::string::npos) {
        (void)prim->AddAttr(attr_name, std::make_shared<ValueTuple>(std::vector<ValuePtr>()));
        break;
      } else if (ref_attr_name.find("List[]") != std::string::npos) {
        (void)prim->AddAttr(attr_name, std::make_shared<ValueList>(std::vector<ValuePtr>()));
        break;
      }
      ObtainCNodeAttrInScalarForm(attr_proto, &multi_value_map);
      break;
    }
    case FORM_PARSE_TENSOR: {
      ObtainCNodeAttrInTensorForm(prim, attr_proto);
      break;
    }
    case FORM_PARSE_NONE: {
      (void)prim->AddAttr(attr_name, kNone);
      break;
    }
    default:
      MS_LOG(ERROR) << "parse attr type don't support the ref_attr_name: " << ref_attr_name;
      return false;
  }

  if (type == FORM_PARSE_SCALAR && multi_value_map.size() != 0) {
    if (ref_attr_name.find("Tuple") != std::string::npos) {
      auto value_tuple_ptr = ParserScalarAttrValue<ValueTuple>(ref_attr_name, multi_value_map);
      (void)prim->AddAttr(attr_name, value_tuple_ptr);
    } else {
      auto value_list_ptr = ParserScalarAttrValue<ValueList>(ref_attr_name, multi_value_map);
      (void)prim->AddAttr(attr_name, value_list_ptr);
    }
  }
  return true;
}

bool MSANFModelParser::ObtainValueNodeInTensorForm(const std::string &value_node_name,
                                                   const mind_ir::TensorProto &attr_tensor) {
  tensor::TensorPtr tensor_info = GenerateTensorPtrFromTensorProto(attr_tensor);
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "Failed to get the tensor for ValueNode.";
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
    tensor::TensorPtr tensor_info = nullptr;
    if (!attr_tensor.has_compression_type() ||
        attr_tensor.compression_type() == mind_ir::TensorProto_CompressionType_NO_COMPRESSION) {
      tensor_info = std::make_shared<tensor::Tensor>(kDefaultValueSwitchMap[attr_tensor_type], shape);
    } else {
      auto compression_type = static_cast<TensorCompressionType>(static_cast<int>(attr_tensor.compression_type()));
      size_t data_size = 0;
      if (!attr_tensor.has_external_data()) {
        data_size = attr_tensor.raw_data().size();
      } else {
        data_size = attr_tensor.external_data().length();
      }
      tensor_info =
        std::make_shared<tensor::Tensor>(kDefaultValueSwitchMap[attr_tensor_type], shape, data_size, compression_type);
    }
    const std::string &tensor_buf = attr_tensor.raw_data();
    auto *tensor_data_buf = reinterpret_cast<uint8_t *>(tensor_info->data_c());
    errno_t ret = memcpy_s(tensor_data_buf, tensor_info->data().nbytes(), tensor_buf.data(), tensor_buf.size());
    if (ret != EOK) {
      MS_LOG(ERROR) << "Obtain ValueNode in TupleTensorForm occur memcpy_s error.";
      return false;
    }
    tensor_vec.push_back(tensor_info);
  }
  auto value = MakeValue(tensor_vec);
  auto new_value_node = NewValueNode(value);
  new_value_node->set_abstract(value->ToAbstract());
  anfnode_build_map_[value_node_name] = new_value_node;
  return true;
}

bool MSANFModelParser::ObtainValueNodeInTypeForm(const std::string &value_node_name,
                                                 const mind_ir::TensorProto &attr_tensor) {
  const int attr_tensor_type = attr_tensor.data_type();
  auto iter = kDefaultValueSwitchMap.find(attr_tensor_type);
  if (iter == kDefaultValueSwitchMap.end()) {
    MS_LOG(ERROR) << "Obtain ValueNode attr in type-form has not support input type: " << attr_tensor_type;
    return false;
  }
  auto value = TypeIdToType(iter->second);
  auto new_value_node = NewValueNode(value);
  new_value_node->set_abstract(value->ToAbstract());
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

ValuePtr MSANFModelParser::ObtainValueInDictionaryForm(const mind_ir::AttributeProto &attr_proto) {
  std::vector<std::pair<ValuePtr, ValuePtr>> key_values;
  for (int i = 0; i < attr_proto.values_size(); ++i) {
    const mind_ir::AttributeProto &key_value_proto = attr_proto.values(i);
    if (!key_value_proto.has_name()) {
      MS_LOG(EXCEPTION) << "Dict type AttributeProto should has name as key of dictionary";
    }
    auto key = std::make_shared<abstract::AbstractScalar>(key_value_proto.name())->BuildValue();
    MS_EXCEPTION_IF_NULL(key);
    auto &values = key_value_proto.values();
    if (values.size() != 1) {
      MS_LOG(EXCEPTION) << "Dict type AttributeProto should has exactly one value, but got " << values.size()
                        << " value(s).";
    }
    auto &value = values[0];
    switch (value.type()) {
      case mind_ir::AttributeProto_AttributeType_TENSORS: {
        const mind_ir::TensorProto &tensor_proto = value.tensors(0);
        if (tensor_proto.has_raw_data()) {
          // For real tensor.
          tensor::TensorPtr tensor_info = GenerateTensorPtrFromTensorProto(tensor_proto);
          if (tensor_info == nullptr) {
            MS_LOG(ERROR) << "Failed to get the tensor for ValueNode.";
            return nullptr;
          }
          (void)key_values.emplace_back(std::make_pair(key, tensor_info));
        } else {
          // For data type.
          const int attr_tensor_type = tensor_proto.data_type();
          auto iter = kDefaultValueSwitchMap.find(attr_tensor_type);
          if (iter == kDefaultValueSwitchMap.end()) {
            MS_LOG(ERROR) << "Obtain ValueNode attr in type-form has not support input type: " << attr_tensor_type;
            return nullptr;
          }
          (void)key_values.emplace_back(std::make_pair(key, TypeIdToType(iter->second)));
        }
        break;
      }
      case mind_ir::AttributeProto_AttributeType_TUPLE:
      case mind_ir::AttributeProto_AttributeType_LIST: {
        auto sequence_value = ObtainValueInSequenceForm(value);
        if (sequence_value == nullptr) {
          MS_LOG(ERROR) << "Failed to get the sequence value";
          return nullptr;
        }
        (void)key_values.emplace_back(std::make_pair(key, sequence_value));
        break;
      }
      case mind_ir::AttributeProto_AttributeType_DICT: {
        auto dict_value = ObtainValueInDictionaryForm(value);
        if (dict_value == nullptr) {
          MS_LOG(ERROR) << "Failed to get the dictionary value";
          return nullptr;
        }
        (void)key_values.emplace_back(std::make_pair(key, dict_value));
        break;
      }
      default: {
        // For string and scalar.
        auto scalar_value = ParseAttrInScalarForm(value, 0);
        if (scalar_value == nullptr) {
          MS_LOG(ERROR) << "Failed to get the scalar for ValueNode.";
          return nullptr;
        }
        (void)key_values.emplace_back(std::make_pair(key, scalar_value));
      }
    }
  }
  return std::make_shared<ValueDictionary>(key_values);
}

ValuePtr MSANFModelParser::ObtainValueInSequenceForm(const mind_ir::AttributeProto &attr_proto) {
  std::vector<ValuePtr> vec;
  for (int i = 0; i < attr_proto.values_size(); ++i) {
    mind_ir::AttributeProto elem_attr_proto = attr_proto.values(i);
    switch (elem_attr_proto.type()) {
      case mind_ir::AttributeProto_AttributeType_TENSORS: {
        mind_ir::TensorProto tensor_proto = elem_attr_proto.tensors(0);
        if (tensor_proto.has_raw_data()) {
          // For real tensor.
          tensor::TensorPtr tensor_info = GenerateTensorPtrFromTensorProto(tensor_proto);
          if (tensor_info == nullptr) {
            MS_LOG(ERROR) << "Failed to get the tensor for ValueNode.";
            return nullptr;
          }
          (void)vec.emplace_back(tensor_info);
        } else {
          // For data type.
          const int attr_tensor_type = tensor_proto.data_type();
          auto iter = kDefaultValueSwitchMap.find(attr_tensor_type);
          if (iter == kDefaultValueSwitchMap.end()) {
            MS_LOG(ERROR) << "Obtain ValueNode attr in type-form has not support input type: " << attr_tensor_type;
            return nullptr;
          }
          (void)vec.emplace_back(TypeIdToType(iter->second));
        }
        break;
      }
      case mind_ir::AttributeProto_AttributeType_TUPLE:
      case mind_ir::AttributeProto_AttributeType_LIST: {
        auto sequence_value = ObtainValueInSequenceForm(elem_attr_proto);
        if (sequence_value == nullptr) {
          MS_LOG(ERROR) << "Failed to get the sequence value";
          return nullptr;
        }
        (void)vec.emplace_back(sequence_value);
        break;
      }
      default: {
        // For string and scalar.
        auto scalar_value = ParseAttrInScalarForm(elem_attr_proto, 0);
        if (scalar_value == nullptr) {
          MS_LOG(ERROR) << "Failed to get the scalar for ValueNode.";
          return nullptr;
        }
        (void)vec.emplace_back(scalar_value);
      }
    }
  }
  auto type = attr_proto.type();
  ValuePtr value_sequence;
  if (type == mind_ir::AttributeProto_AttributeType_TUPLE) {
    value_sequence = std::make_shared<ValueTuple>(vec);
  } else if (type == mind_ir::AttributeProto_AttributeType_LIST) {
    value_sequence = std::make_shared<ValueList>(vec);
  } else {
    MS_LOG(EXCEPTION) << "The attribute type should be tuple or list, but it is " << type;
  }

  return value_sequence;
}

ValuePtr MSANFModelParser::BuildValueFromAttributeProto(const mind_ir::AttributeProto &attr_proto) {
  switch (attr_proto.type()) {
    case mind_ir::AttributeProto_AttributeType_TENSORS: {
      const auto &tensor_proto = attr_proto.tensors(0);
      if (tensor_proto.has_raw_data()) {
        // For real tensor.
        tensor::TensorPtr tensor_info = GenerateTensorPtrFromTensorProto(tensor_proto);
        if (tensor_info == nullptr) {
          MS_LOG(ERROR) << "Failed to GenerateTensorPtrFromTensorProto.";
          return nullptr;
        }
        return MakeValue(tensor_info);
      } else {
        // For data type.
        const int attr_tensor_type = tensor_proto.data_type();
        auto iter = kDefaultValueSwitchMap.find(attr_tensor_type);
        if (iter == kDefaultValueSwitchMap.end()) {
          MS_LOG(ERROR) << "Obtain ValueNode attr in type-form has not support input type: " << attr_tensor_type;
          return nullptr;
        }
        return TypeIdToType(iter->second);
      }
    }
    case mind_ir::AttributeProto_AttributeType_NONE: {
      return kNone;
    }
    case mind_ir::AttributeProto_AttributeType_UMONAD: {
      return kUMonad;
    }
    case mind_ir::AttributeProto_AttributeType_IOMONAD: {
      return kIOMonad;
    }
    case mind_ir::AttributeProto_AttributeType_TUPLE:
    case mind_ir::AttributeProto_AttributeType_LIST: {
      return ObtainValueInSequenceForm(attr_proto);
    }
    case mind_ir::AttributeProto_AttributeType_CLASS_TYPE: {
      auto class_type = static_cast<std::string>(attr_proto.s());
      return std::make_shared<MindIRClassType>(class_type);
    }
    case mind_ir::AttributeProto_AttributeType_TYPE_NULL: {
      return kTypeNull;
    }
    case mind_ir::AttributeProto_AttributeType_NAME_SPACE: {
      auto name_space = static_cast<std::string>(attr_proto.s());
      return std::make_shared<MindIRNameSpace>(name_space);
    }
    case mind_ir::AttributeProto_AttributeType_SYMBOL: {
      auto symbol = static_cast<std::string>(attr_proto.s());
      return std::make_shared<MindIRSymbol>(symbol);
    }
    default: {
      return ObtainCNodeAttrInSingleScalarForm(attr_proto);
    }
  }
}

bool MSANFModelParser::GetAttrValueForValueNodeWithType(const std::string &value_node_name,
                                                        const mind_ir::AttributeProto &attr_proto) {
  auto value = BuildValueFromAttributeProto(attr_proto);
  if (value == nullptr) {
    MS_LOG(ERROR) << "Failed to build value from AttributeProto while building valuenode: " << value_node_name;
    return false;
  }
  auto abstract = value->ToAbstract();
  MS_EXCEPTION_IF_NULL(abstract);
  ValueNodePtr new_value_node = NewValueNode(value);
  new_value_node->set_abstract(abstract);
  anfnode_build_map_[value_node_name] = new_value_node;
  return true;
}

bool MSANFModelParser::GetAttrValueForValueNode(const std::string &value_node_name,
                                                const mind_ir::AttributeProto &attr_proto) {
  if (!attr_proto.has_ref_attr_name()) {
    MS_LOG(ERROR) << "CNode parse attr type has no ref_attr_name";
    return false;
  }
  const std::string &ref_attr_name = attr_proto.ref_attr_name();
  ParseForm type = GetParseFormType(ref_attr_name);
  ValueNodePtr new_value_node;
  mindspore::HashMap<std::string, ValuePtr> multi_value_map;
  switch (type) {
    case FORM_PARSE_TYPE: {
      ObtainValueNodeInTypeForm(value_node_name, attr_proto.tensors(0));
      break;
    }
    case FORM_PARSE_SCALAR: {
      if (ref_attr_name.find("value0") != std::string::npos) {
        auto res = ObtainCNodeAttrInSingleScalarForm(attr_proto);
        new_value_node = NewValueNode(res);
        new_value_node->set_abstract(res->ToAbstract());
        anfnode_build_map_[value_node_name] = new_value_node;
        break;
      }
      if (ref_attr_name.find("Tuple[]") != std::string::npos) {
        MS_LOG(INFO) << "Build Tuple() ValueNode for primitive.";
        ValuePtr res = MakeValue(std::vector<ValuePtr>{});
        new_value_node = NewValueNode(res);
        new_value_node->set_abstract(res->ToAbstract());
        anfnode_build_map_[value_node_name] = new_value_node;
        break;
      }
      if (ref_attr_name.find("Tuple[value") != std::string::npos && attr_proto.tensors_size() > 1) {
        MS_LOG(INFO) << "Build TupleTensor ValueNode for primitive.";
        (void)ObtainValueNodeInTupleTensorForm(value_node_name, attr_proto);
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
      (void)ObtainValueNodeInNoneForm(value_node_name);
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
  if (type == FORM_PARSE_SCALAR && !multi_value_map.empty()) {
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
  if (node_proto.output_size() == 0) {
    MS_LOG(ERROR) << "The Proto output is empty.";
    return false;
  }
  const std::string &value_node_name = node_proto.output(0);
  const mind_ir::AttributeProto &attr_proto = node_proto.attribute(0);
  if (attr_proto.has_ref_attr_name()) {
    return GetAttrValueForValueNode(value_node_name, attr_proto);
  }
  return GetAttrValueForValueNodeWithType(value_node_name, attr_proto);
}

mindspore::HashMap<std::string, abstract::AbstractBasePtr> MSANFModelParser::GetAbstractForNode(
  const mind_ir::AttributeProto &attr_proto) {
  mindspore::HashMap<std::string, abstract::AbstractBasePtr> kv;
  for (int i = 0; i < attr_proto.tensors_size(); ++i) {
    const mind_ir::TensorProto &attr_tensor = attr_proto.tensors(i);
    auto tensor_info = GetAbsTensorFromTensorProto(attr_tensor);
    (void)kv.emplace(attr_tensor.name(), tensor_info);
  }
  return kv;
}

AnfNodePtr MSANFModelParser::BuildOperatorNode(const mind_ir::NodeProto &node_proto) {
  const std::string kOperatorTypeFlag = std::string("REF::");
  const size_t kOpTypeFlagSize = kOperatorTypeFlag.length();
  const std::string &node_type = node_proto.op_type();
  MS_LOG(DEBUG) << "Process Operator:" << node_type;
  // Operator maybe CNode,FuncGraph or Parameter.

  if (node_type.size() > kOpTypeFlagSize && node_type.substr(0, kOpTypeFlagSize) == kOperatorTypeFlag) {
    auto anfNode = GetAnfNode(node_type.substr(kOpTypeFlagSize));
    if (anfNode == nullptr) {
      MS_LOG(ERROR) << "Can't find the ref:" << node_type;
      return nullptr;
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
      auto op_name = node_type.substr(strlen(kDoSignaturePrimitivePrefix));
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
  prim->set_attr("is_load", MakeValue(true));
  return NewValueNodeWithAbstract(prim);
}

bool MSANFModelParser::SetEmptyTensorProtoCNodeAbstract(const AnfNodePtr &node_ptr) {
  auto primitive = GetCNodePrimitive(node_ptr);
  if (primitive != nullptr) {
    auto node_type = primitive->name();
    if (node_type == "UpdateState") {
      node_ptr->set_abstract(kUMonad->ToAbstract());
    } else if (node_type == "Depend") {
      node_ptr->set_abstract(kBool->ToAbstract());
    } else {
      auto cnode_ptr = node_ptr->cast<CNodePtr>();
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
        node_ptr->set_abstract(std::make_shared<abstract::AbstractTuple>(elem));
      }
    }
  } else {
    MS_LOG(ERROR) << "Failed to get the abstract of node:" << node_ptr->DebugString();
    return false;
  }
  return true;
}

// Set CNode abstract.
void MSANFModelParser::SetCNodeAbstract(const mind_ir::AttributeProto &attr_proto, const CNodePtr &cnode_ptr) {
  if (attr_proto.has_ref_attr_name()) {
    if (!SetNodeAbstractFromAttrProto(attr_proto, cnode_ptr)) {
      MS_LOG(ERROR) << "Failed to get CNode abstract from proto.";
    }
  } else {
    auto abs = GetNodeAbstractFromAttrProtoWithType(attr_proto);
    cnode_ptr->set_abstract(abs);
  }
  if (cnode_ptr->abstract() == nullptr) {
    MS_LOG(INFO) << "Failed to Build CNode abstract from proto. CNode: " << cnode_ptr->ToString()
                 << " attr_proto: " << attr_proto.DebugString();
    abstract_valid_ = false;
  }
}

CNodePtr MSANFModelParser::BuildCNodeForFuncGraph(const FuncGraphPtr &outputFuncGraph,
                                                  const mind_ir::NodeProto &node_proto) {
  MS_EXCEPTION_IF_NULL(outputFuncGraph);
  if (!node_proto.has_op_type()) {
    MS_LOG(ERROR) << "Get CNode op_type failed!";
    return nullptr;
  }
  if (node_proto.output_size() <= 0) {
    MS_LOG(ERROR) << "Get CNode out failed!";
    return nullptr;
  }
  const std::string &node_name = node_proto.output(0);
  MS_LOG(DEBUG) << "Process CNode: " << node_name;
  // Build inputs.
  std::vector<AnfNodePtr> inputs;
  auto operator_node = BuildOperatorNode(node_proto);
  if (operator_node == nullptr) {
    MS_LOG(ERROR) << "Build operator node " << node_name << " failed!";
    return nullptr;
  }
  inputs.push_back(operator_node);
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
  if (anfnode_build_map_.count(node_name) > 0) {
    MS_LOG(EXCEPTION) << "Duplicate CNode name: " << node_name;
  }
  const std::string &fullname_with_scope = node_proto.domain();
  string debug_info_name = ParseCNodeName(node_name);
  auto debug_info_ptr = std::make_shared<NodeDebugInfo>(debug_info_name);
  cnode_ptr->set_debug_info(debug_info_ptr);
  cnode_ptr->set_fullname_with_scope(fullname_with_scope);
  cnode_ptr->set_load_flag(true);
  anfnode_build_map_[node_name] = cnode_ptr;

  // Set Abstract and prim attr for CNode
  SetCNodePrimAttrAndAbstract(node_proto, cnode_ptr);
  BuildAttrForCNode(cnode_ptr, node_proto);
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
    inputs.clear();
    inputs.push_back(NewValueNode(prim::kPrimReturn));
    inputs.push_back(maketuple_ptr);
    auto return_node = outputFuncGraph->NewCNode(inputs);
    MS_EXCEPTION_IF_NULL(return_node);
    return_node->set_abstract(maketuple_ptr->abstract());
    return_node->set_load_flag(true);
    outputFuncGraph->set_return(return_node);
    MS_LOG(DEBUG) << "Construct funcgraph finined, all success.";
    return true;
  } else if (importProto.output_size() == 1) {
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
    return_node->set_abstract(anfNode->abstract());
    return_node->set_load_flag(true);
    outputFuncGraph->set_return(return_node);
    MS_LOG(DEBUG) << "Construct funcgraph finined, all success!";
    return true;
  }

  return false;
}

bool MSANFModelParser::ImportNodesForGraph(const FuncGraphPtr &outputFuncGraph,
                                           const mind_ir::GraphProto &importProto) {
  MS_EXCEPTION_IF_NULL(outputFuncGraph);
  MS_LOG(DEBUG) << "The node size: " << importProto.node_size();
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

bool MSANFModelParser::BuildAttrForFuncGraph(const FuncGraphPtr &outputFuncGraph,
                                             const mind_ir::GraphProto &importProto) {
  for (auto i = 0; i < importProto.attribute_size(); ++i) {
    const mind_ir::AttributeProto &attr_proto = importProto.attribute(i);
    auto value = GetValueFromAttributeProto(attr_proto);
    if (value == nullptr) {
      MS_LOG(ERROR) << "Failed set func_graph attr to func_graph";
      return false;
    }
    outputFuncGraph->set_attr(attr_proto.name(), value);
  }
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
  if (importProto.has_bprop_hash()) {
    outputFuncGraph->set_bprop_hash(importProto.bprop_hash());
  }

  if (importProto.has_bprop_filepath()) {
    outputFuncGraph->set_bprop_filepath(importProto.bprop_filepath());
  }

  if (!BuildAttrForFuncGraph(outputFuncGraph, importProto)) {
    MS_LOG(ERROR) << "Build attribute for graph fail!";
  }

  if (!ImportParametersForGraph(outputFuncGraph, importProto)) {
    MS_LOG(ERROR) << "Import parameters for graph fail!";
    return false;
  }
  if (!ImportMapParametersForGraph(outputFuncGraph, importProto)) {
    MS_LOG(ERROR) << "Import map parameters for graph failed!";
    return false;
  }
  if (ImportNodesForGraph(outputFuncGraph, importProto)) {
    MS_LOG(DEBUG) << "Success to parse graph: " << outputFuncGraph->ToString() << ": " << outputFuncGraph.get();
    return true;
  }
  MS_LOG(ERROR) << "Failed to parse nodes. ";
  return false;
}

bool MSANFModelParser::MSANFParseModelConfigureInfo(const mind_ir::ModelProto &model_proto) {
  if (!model_proto.has_producer_name()) {
    MS_LOG(ERROR) << "Parse model producer name from pb file failed!";
    return false;
  }
  producer_name_ = model_proto.producer_name();
  MS_LOG(INFO) << "producer_name: " << producer_name_;

  if (!model_proto.has_model_version()) {
    MS_LOG(ERROR) << "Parse model producer version from pb file failed!";
    return false;
  }
  model_version_ = model_proto.model_version();
  MS_LOG(INFO) << "producer_version: " << model_version_;

  int64 mind_ir_version = 0;
  if (model_proto.has_mind_ir_version()) {
    mind_ir_version = model_proto.mind_ir_version();
  }
  if (!mind_ir::Version_IsValid(mind_ir_version)) {
    MS_LOG(EXCEPTION) << "This software can only support the maximum mind ir version: " << mind_ir::Version_MAX
                      << ", please install the latest version to support the mind ir version: " << mind_ir_version;
  }

  return true;
}

bool MSANFModelParser::SetValueForTopGraphParameter(const FuncGraphPtr &topGraph,
                                                    const std::map<std::string, ValuePtr> &weights) {
  size_t fv_param_count = 0;
  auto parameters = topGraph->parameters();
  for (int64_t i = SizeToLong(parameters.size()) - 1; i >= 0; --i) {
    auto parameter = parameters[i]->cast<ParameterPtr>();
    if (parameter == nullptr) {
      MS_LOG(ERROR) << "AnfNode " << parameters[i]->DebugString() << " should be Parameter.";
      return false;
    }
    auto type = parameter->Type();
    if (type == nullptr) {
      MS_LOG(ERROR) << "Parameter " << parameter->DebugString() << " has no type.";
      return false;
    }
    if (!type->isa<RefType>()) {
      break;
    }
    auto parameter_name = parameter->name();
    auto weights_iter = weights.find(parameter_name);
    if (weights_iter == weights.end()) {
      MS_LOG(ERROR) << "Find initial weight value for " << parameter_name << " failed.";
      return false;
    }
    parameter->set_default_param(weights_iter->second);
    fv_param_count++;
  }
  topGraph->set_fv_param_count(fv_param_count);
  return true;
}

FuncGraphPtr MSANFModelParser::Parse(const mind_ir::ModelProto &model_proto,
                                     const std::map<std::string, ValuePtr> &weights) {
  if (IsLite()) {
    abstract_valid_ = true;
  }

  if (!MSANFParseModelConfigureInfo(model_proto)) {
    MS_LOG(ERROR) << "Parse configuration info for pb file failed!";
  }

  if (model_proto.has_little_endian()) {
    if (model_proto.little_endian() != this->little_endian()) {
      MS_LOG(ERROR) << "The byte order of export MindIr device and load MindIr device is not same!";
      return nullptr;
    }
  }

  FuncGraphPtr dstGraph = std::make_shared<FuncGraph>();

  for (int i = 0; i < model_proto.primitives_size(); ++i) {
    if (!BuildPrimitiveNode(model_proto.primitives(i))) {
      MS_LOG(ERROR) << "Parse primitives info for pb file failed! " << model_proto.primitives(i).DebugString();
      return nullptr;
    }
  }

  const mind_ir::GraphProto &graphBuild = model_proto.graph();

  // Forward declare FuncGraph name
  // Compatible with the previous proto.
  if (graphBuild.has_name()) {
    anfnode_build_map_[graphBuild.name()] = NewValueNodeWithAbstract(dstGraph);
  }
  for (int i = 0; i < model_proto.functions_size(); ++i) {
    FuncGraphPtr graph = std::make_shared<FuncGraph>();
    const auto &graph_proto = model_proto.functions(i);
    if (!graph_proto.has_name()) {
      MS_LOG(ERROR) << "The function has not a name. Please export mindIR again. ";
      return nullptr;
    }
    if (anfnode_build_map_.count(graph_proto.name()) > 0) {
      MS_LOG(ERROR) << "There is a duplication function graph name: " << graph_proto.name();
      return nullptr;
    }
    auto debug_info = graph->debug_info();
    debug_info->set_name(graph_proto.name());
    anfnode_build_map_[graph_proto.name()] = NewValueNodeWithAbstract(graph);
  }

  // Parser the proto.
  if (!BuildFuncGraph(dstGraph, graphBuild)) {
    MS_LOG(ERROR) << "Build funcgraph failed!";
    return nullptr;
  }

  if (!weights.empty()) {
    if (!SetValueForTopGraphParameter(dstGraph, weights)) {
      MS_LOG(ERROR) << "Set value for top graph fail.";
      return nullptr;
    }
  }

  MS_LOG(DEBUG) << "Parse pb to build FuncGraph Success! graph: " << graphBuild.name() << ": " << dstGraph.get();
  top_graph_ = dstGraph;
  for (int i = 0; i < model_proto.functions_size(); ++i) {
    const auto &graph_proto = model_proto.functions(i);
    FuncGraphPtr graph = GetValueNode<FuncGraphPtr>(anfnode_build_map_[graph_proto.name()]);
    if (!BuildFuncGraph(graph, graph_proto)) {
      MS_LOG(ERROR) << "Build funcgraph failed!";
      return nullptr;
    }
    MS_LOG(DEBUG) << "Parse pb to build FuncGraph Success! graph: " << graph_proto.name() << ": " << graph.get();
  }

  // Release resource
  anfnode_build_map_.clear();

  // Correct the null abstract for compatibility with previous versions.
  if (!abstract_valid_ && weights.empty()) {
    CorrectFuncGraph(dstGraph);
  }
  return dstGraph;
}

const LayoutMap MSANFModelParser::ParseLayout(const mind_ir::ModelProto &model_proto) {
  LayoutMap ret;
  mind_ir::ParallelProto parallel_proto = model_proto.parallel();
  for (int i = 0; i < parallel_proto.layout_size(); ++i) {
    const mind_ir::LayoutProto &layout_proto = parallel_proto.layout(i);
    LayoutPtr cur_layout = std::make_shared<Layout>();
    const std::string name = layout_proto.name();
    std::vector<int64_t> device_arrangement;
    for (int num = 0; num < layout_proto.device_arrangement_int_size(); ++num) {
      (void)device_arrangement.emplace_back(layout_proto.device_arrangement_int(num));
    }
    std::vector<int64_t> tensor_map;
    for (int num = 0; num < layout_proto.tensor_map_int_size(); ++num) {
      (void)tensor_map.emplace_back(layout_proto.tensor_map_int(num));
    }
    std::vector<int64_t> slice_shape;
    for (int num = 0; num < layout_proto.slice_shape_int_size(); ++num) {
      (void)slice_shape.emplace_back(layout_proto.slice_shape_int(num));
    }
    int64_t field_size = layout_proto.field_size();
    bool uniform_spilt = layout_proto.uniform_split();
    const std::string opt_shard_group = layout_proto.opt_shard_group();

    cur_layout->set_device_arrangement(device_arrangement);
    cur_layout->set_tensor_map(tensor_map);
    cur_layout->set_slice_shape(slice_shape);
    cur_layout->set_field_size(field_size);
    cur_layout->set_uniform_split(uniform_spilt);
    cur_layout->set_opt_shard_group(opt_shard_group);

    ret[name] = cur_layout;
  }
  return ret;
}

AnfNodePtr MSANFModelParser::GetAnfNode(const std::string &node_name) {
  if (node_name.find("MetaFuncGraph::") == 0) {
    auto fg_name = node_name.substr(std::string("MetaFuncGraph::").length());
    auto mindir_meta_fg = std::make_shared<MindIRMetaFuncGraph>(fg_name);
    return NewValueNode(mindir_meta_fg);
  }
  if (node_name.find("ClassType::") == 0) {
    auto class_type = node_name.substr(std::string("ClassType::").length());
    auto mindir_class_type = std::make_shared<MindIRClassType>(class_type);
    return NewValueNode(mindir_class_type);
  }
  auto it = anfnode_build_map_.find(node_name);
  if (it == anfnode_build_map_.end()) {
    return nullptr;
  }
  FuncGraphPtr func_graph_ptr = GetValueNode<FuncGraphPtr>(it->second);
  if (func_graph_ptr != nullptr) {
    auto node = NewValueNode(func_graph_ptr);
    node->set_abstract(it->second->abstract());
    return node;
  } else {
    return it->second;
  }
}

bool MSANFModelParser::BuildPrimitiveNode(const mind_ir::PrimitiveProto &primitive_proto) {
  static auto op_primc_fns = ops::OpPrimCRegister::GetInstance().GetPrimCMap();
  auto &prim_type = primitive_proto.op_type();
  std::shared_ptr<Primitive> prim;

  auto it = op_primc_fns.find(prim_type);
  if (it != op_primc_fns.end()) {
    prim = it->second();
  } else {
    if (prim_type.compare(0, strlen(kDoSignaturePrimitivePrefix), kDoSignaturePrimitivePrefix) == 0) {
      auto op_name = prim_type.substr(strlen(kDoSignaturePrimitivePrefix));
      prim = std::make_shared<prim::DoSignaturePrimitive>(op_name, std::make_shared<Primitive>(op_name));
    } else {
      MS_LOG(DEBUG) << "Special node_type: " << prim_type;
      prim = std::make_shared<Primitive>(prim_type);
    }
  }

  if (primitive_proto.has_instance_name()) {
    prim->set_instance_name(primitive_proto.instance_name());
  }

  // Set primitive attributes
  auto prim_to_add_attr = GetValueWithoutDoSignature(prim)->cast<PrimitivePtr>();
  MS_EXCEPTION_IF_NULL(prim_to_add_attr);
  prim_to_add_attr->set_attr("is_load", MakeValue(true));
  for (int i = 0; i < primitive_proto.attribute_size(); ++i) {
    const mind_ir::AttributeProto &attr_proto = primitive_proto.attribute(i);
    if (!SetPrimitiveAttrWithType(prim_to_add_attr, attr_proto)) {
      MS_LOG(ERROR) << "Parse prim: " << prim->ToString() << " attributes error: " << attr_proto.DebugString();
      return false;
    }
  }
  if (anfnode_build_map_.count(primitive_proto.name()) > 0) {
    MS_LOG(ERROR) << "There is a duplication primitive instance name: " << primitive_proto.name();
    return false;
  }
  anfnode_build_map_[primitive_proto.name()] = NewValueNodeWithAbstract(prim);
  return true;
}

abstract::AbstractBasePtr MSANFModelParser::BuildAbstractFunction(const mind_ir::AttributeProto &attr_proto) {
  switch (attr_proto.type()) {
    case mind_ir::AttributeProto_AttributeType_PRIMITIVECLOSURE:
    case mind_ir::AttributeProto_AttributeType_FUNCGRAPHCLOSURE: {
      auto func_node = GetAnfNode(attr_proto.s());
      if (func_node == nullptr) {
        MS_LOG(WARNING) << "Failed to get function graph closure: " << attr_proto.DebugString();
        return nullptr;
      }
      return func_node->abstract();
    }
    case mind_ir::AttributeProto_AttributeType_PARTIALCLOSURE: {
      auto anf_node = GetAnfNode(attr_proto.s());
      MS_EXCEPTION_IF_NULL(anf_node);
      auto partial_node = anf_node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(partial_node);
      AbstractBasePtrList args_spec_list;
      auto &inputs = partial_node->inputs();
      const size_t kPartial_args_begin_pos = 2;
      const size_t kPartial_fn_pos = 1;
      if (inputs.size() <= kPartial_args_begin_pos) {
        MS_LOG(ERROR) << "Partial node input size is wrong.";
        return nullptr;
      }
      (void)std::transform(inputs.begin() + kPartial_args_begin_pos, inputs.end(), std::back_inserter(args_spec_list),
                           [](const AnfNodePtr &arg) -> AbstractBasePtr { return arg->abstract(); });
      auto &op_node = inputs[kPartial_fn_pos];
      abstract::AbstractFuncAtomPtr fn;
      if (op_node->abstract() != nullptr) {
        fn = op_node->abstract()->cast<abstract::AbstractFuncAtomPtr>();
        if (fn == nullptr) {
          MS_LOG(ERROR) << "Can't get the abstract of partial node: " << op_node->ToString();
          return nullptr;
        }
      } else {
        MS_LOG(WARNING) << "Can't get the abstract of partial node: " << op_node->ToString();
        return nullptr;
      }
      return std::make_shared<abstract::PartialAbstractClosure>(fn, args_spec_list, partial_node);
    }
    case mind_ir::AttributeProto_AttributeType_UNIONFUNCCLOSURE: {
      abstract::AbstractFuncAtomPtrList func_list;
      for (int index = 0; index < attr_proto.values_size(); index++) {
        auto &item_proto = attr_proto.values(index);
        auto item_abstract = BuildAbstractFunction(item_proto);
        if (item_abstract == nullptr) {
          MS_LOG(WARNING) << "Can't get the abstract of function union closure: " << item_proto.DebugString();
          return nullptr;
        }
        (void)func_list.emplace_back(item_abstract->cast<abstract::AbstractFuncAtomPtr>());
      }
      return std::make_shared<abstract::AbstractFuncUnion>(func_list);
    }
    default: {
      MS_LOG(ERROR) << "Not support function abstract: " << attr_proto.DebugString();
      return nullptr;
    }
  }
}

void MSANFModelParser::CorrectFuncGraph(const FuncGraphPtr &root) {
  MS_LOG(DEBUG) << "Begin to correct the funcgraph.";
  MS_EXCEPTION_IF_NULL(root);
  auto inputs = root->get_inputs();
  auto valid =
    std::all_of(inputs.begin(), inputs.end(), [](const AnfNodePtr &arg) -> bool { return arg->abstract() != nullptr; });
  if (valid) {
    (void)ValidMindir(root);
  } else {
    MS_LOG(INFO) << "There are some nullptr of abstract in the top function graph parameters." << root->DumpText();
  }
  MS_LOG(DEBUG) << "End to correct the funcgraph.";
}

bool MSANFModelParser::BuildAttrForCNode(const CNodePtr &cnode, const mind_ir::NodeProto &node_proto) {
  for (auto i = 0; i < node_proto.node_attr_size(); ++i) {
    const auto &attr_proto = node_proto.node_attr(i);
    auto value = GetValueFromAttributeProto(attr_proto);
    if (value == nullptr) {
      MS_LOG(ERROR) << "Failed set func_graph attr to func_graph";
      return false;
    }
    cnode->AddAttr(attr_proto.name(), value);
  }
  for (auto i = 0; i < node_proto.primal_attr_size(); ++i) {
    const auto &attr_proto = node_proto.primal_attr(i);
    auto value = GetValueFromAttributeProto(attr_proto);
    if (value == nullptr) {
      MS_LOG(ERROR) << "Failed set func_graph attr to func_graph";
      return false;
    }
    cnode->AddPrimalAttr(attr_proto.name(), value);
  }
  return true;
}
}  // namespace mindspore

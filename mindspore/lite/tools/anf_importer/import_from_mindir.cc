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

#include "tools/anf_importer/import_from_mindir.h"
#include <unistd.h>
#include <map>
#include <memory>
#include <stack>
#include <unordered_map>
#include <vector>
#include <algorithm>

#include "src/ops/primitive_c.h"
#include "frontend/operator/ops.h"
#include "include/errorcode.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "schema/inner/model_generated.h"
#include "securec/include/securec.h"
#include "src/tensor.h"
#include "src/param_value_lite.h"
#include "proto/onnx.pb.h"
#include "src/common/log_adapter.h"
#include "tools/common/protobuf_utils.h"
#include "tools/common/graph_util.h"
#include "load_mindir/load_model.h"

using string = std::string;
using int32 = int32_t;
using int64 = int64_t;
using uint64 = uint64_t;

namespace mindspore::lite {
static constexpr char kConstantValueNode[] = "Constant";

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
  std::stack<std::string> rules;
  std::stack<ValuePtr> value;
  int num = 0, count = 0;
  for (size_t i = 0; i < str.length(); i++) {
    if (str[i] == '[') {
      rules.push("[");
    } else if (str[i] == ']') {
      // rules
      std::vector<ValuePtr> vec;
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
      auto vt = std::make_shared<ValueTuple>(vec);
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
        num++;
      }
    }
  }
  return {};
}

std::shared_ptr<abstract::AbstractTuple> ParserAttrShape(
  const std::string &attr_name, const std::unordered_map<string, abstract::AbstractTensorPtr> &kv) {
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
  std::stack<std::string> rules;
  std::stack<abstract::AbstractBasePtr> value;
  int num = 0, count = 0;
  for (size_t i = 0; i < str.length(); i++) {
    if (str[i] == '[') {
      rules.push("[");
    } else if (str[i] == ']') {
      // rules
      std::vector<abstract::AbstractBasePtr> vec;
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
      auto vt = std::make_shared<abstract::AbstractTuple>(vec);
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
        num++;
      }
    }
  }
  return {};
}

#define PARSE_ONNXATTR_IN_SCALAR_FORM(type, valuetype)                                    \
  ValuePtr ParseAttrInScalar_##type##_##valuetype(const onnx::TensorProto &attr_tensor) { \
    if (attr_tensor.type##_data_size() == 1) {                                            \
      auto value = static_cast<valuetype>(attr_tensor.type##_data(0));                    \
      return MakeValue<valuetype>(value);                                                 \
    } else {                                                                              \
      MS_LOG(ERROR) << "size of scalar tensor doesn't equal 1!";                          \
    }                                                                                     \
    return {};                                                                            \
  }

PARSE_ONNXATTR_IN_SCALAR_FORM(double, double)
PARSE_ONNXATTR_IN_SCALAR_FORM(float, float)
PARSE_ONNXATTR_IN_SCALAR_FORM(string, string)
PARSE_ONNXATTR_IN_SCALAR_FORM(int32, int32)
PARSE_ONNXATTR_IN_SCALAR_FORM(int32, bool)
PARSE_ONNXATTR_IN_SCALAR_FORM(int64, int64)
PARSE_ONNXATTR_IN_SCALAR_FORM(uint64, uint64)

int AnfImporterFromMindir::BuildParameterForFuncGraph(const ParameterPtr &node,
                                                      const onnx::ValueInfoProto &value_proto) {
  if (node == nullptr) {
    return RET_NULL_PTR;
  }
  if (!value_proto.has_type() || !value_proto.has_name()) {
    MS_LOG(ERROR) << "onnx ValueInfoProto has no type or name! ";
    return RET_PARAM_INVALID;
  }
  node->set_name(value_proto.name());
  const auto &type_proto = value_proto.type();
  if (!type_proto.has_tensor_type()) {
    MS_LOG(ERROR) << "onnx TypeProto has no tensor_type! ";
    return RET_PARAM_INVALID;
  }
  const onnx::TypeProto_Tensor &tensor_typeproto = type_proto.tensor_type();
  if (!tensor_typeproto.has_elem_type() || !tensor_typeproto.has_shape()) {
    MS_LOG(ERROR) << "onnx TypeProto_Tensor has no elem_type or shape! ";
    return RET_INPUT_TENSOR_ERROR;
  }
  const onnx::TensorShapeProto &tensor_shape = tensor_typeproto.shape();
  std::vector<int> shape;
  for (int i = 0; i < tensor_shape.dim_size(); ++i) {
    shape.push_back(tensor_shape.dim(i).dim_value());
  }

  if (kDefaultValueSwitchMap.find(tensor_typeproto.elem_type()) == kDefaultValueSwitchMap.end()) {
    MS_LOG(ERROR) << "onnx TypeProto_Tensor  elem_type is not support yet!";
    return RET_PARAM_INVALID;
  }

  auto type_ptr = TypeIdToType(kDefaultValueSwitchMap[tensor_typeproto.elem_type()]);
  std::vector<int64_t> shape_vector;
  (void)std::transform(shape.begin(), shape.end(), std::back_inserter(shape_vector),
                       [](const int32_t &value) { return static_cast<int64_t>(value); });
  auto abstract_tensor = std::make_shared<abstract::AbstractTensor>(type_ptr, shape_vector);
  node->set_abstract(abstract_tensor);

  if (default_para_map_.find(value_proto.name()) != default_para_map_.end()) {
    auto *tensor_info = new (std::nothrow) Tensor(kDefaultValueSwitchMap[tensor_typeproto.elem_type()], shape);
    if (tensor_info == nullptr) {
      return RET_MEMORY_FAILED;
    }
    tensor_info->MallocData();
    const onnx::TensorProto initialize_proto = default_para_map_[value_proto.name()];
    std::string initial_data = initialize_proto.raw_data();
    auto *tensor_data_buf = reinterpret_cast<uint8_t *>(tensor_info->MutableData());
    if (tensor_data_buf == nullptr) {
      delete tensor_info;
      return RET_MEMORY_FAILED;
    }
    tensor_info->set_data(nullptr);
    auto ret = memcpy_s(tensor_data_buf, tensor_info->Size(), initial_data.data(), initial_data.size());
    if (EOK != ret) {
      MS_LOG(ERROR) << "memcpy_s error";
      delete tensor_data_buf;
      delete tensor_info;
      return RET_MEMORY_FAILED;
    }

    ParamValueLitePtr param_value = std::make_shared<ParamValueLite>();
    if (param_value == nullptr) {
      delete tensor_info;
      return RET_NULL_PTR;
    }
    param_value->SetTensorData(tensor_data_buf, tensor_info->Size());
    param_value->set_tensor_type(tensor_info->data_type());
    param_value->set_tensor_shape(tensor_info->shape());
    node->set_default_param(param_value);
    delete tensor_info;
  }
  anfnode_build_map_[value_proto.name()] = node;
  return RET_OK;
}

int AnfImporterFromMindir::ImportParametersForGraph(const FuncGraphPtr &outputFuncGraph,
                                                    const onnx::GraphProto &importProto) {
  if (outputFuncGraph == nullptr) {
    return RET_NULL_PTR;
  }
  MS_LOG(INFO) << "Parameters had default paramerer size is: " << importProto.initializer_size();

  for (int i = 0; i < importProto.initializer_size(); ++i) {
    const onnx::TensorProto &initializer_proto = importProto.initializer(i);
    if (!initializer_proto.has_name()) {
      MS_LOG(ERROR) << "initializer vector of onnx GraphProto has no name at index: " << i;
      return RET_PARAM_INVALID;
    }
    default_para_map_[initializer_proto.name()] = initializer_proto;
  }

  int status = RET_OK;
  MS_LOG(INFO) << "all parameters size: " << importProto.input_size();
  for (int i = 0; i < importProto.input_size(); ++i) {
    const onnx::ValueInfoProto &input_proto = importProto.input(i);
    status = BuildParameterForFuncGraph(outputFuncGraph->add_parameter(), input_proto);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Build parameter for funcgraph fail at index: " << i;
      break;
    }
  }
  return status;
}

bool AnfImporterFromMindir::ObtainCNodeAttrInTypeForm(const PrimitivePtr &prim, const std::string &attr_name,
                                                      const onnx::TensorProto &attr_tensor) {
  if (prim == nullptr) {
    return false;
  }
  const int attr_tensor_type = attr_tensor.data_type();
  if (kDefaultValueSwitchMap.find(attr_tensor_type) == kDefaultValueSwitchMap.end()) {
    MS_LOG(ERROR) << "Obtain attr in type-form has not support input type:" << attr_tensor_type;
    return false;
  }
  prim->AddAttr(attr_name, TypeIdToType(kDefaultValueSwitchMap[attr_tensor_type]));
  return true;
}

ValuePtr AnfImporterFromMindir::ObtainCNodeAttrInScalarForm(const onnx::TensorProto &attr_tensor) {
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
}

bool AnfImporterFromMindir::ObtainCNodeAttrInTensorForm(const PrimitivePtr &prim, const std::string &attr_name,
                                                        const onnx::TensorProto &attr_tensor) {
  if (prim == nullptr) {
    return false;
  }
  const int attr_tensor_type = attr_tensor.data_type();
  const std::string &tensor_buf = attr_tensor.raw_data();
  std::vector<int> shape;
  auto ret = EOK;
  if (attr_tensor.dims_size() != 0) {
    for (int i = 0; i < attr_tensor.dims_size(); ++i) {
      shape.push_back(attr_tensor.dims(i));
    }
    std::vector<int64_t> shape_vector;
    (void)std::transform(shape.begin(), shape.end(), std::back_inserter(shape_vector),
                         [](const int32_t &value) { return static_cast<int64_t>(value); });
    tensor::TensorPtr tensor_info =
      std::make_shared<tensor::Tensor>(kDefaultValueSwitchMap[attr_tensor_type], shape_vector);
    auto *tensor_data_buf = reinterpret_cast<uint8_t *>(tensor_info->data_c());
    ret = memcpy_s(tensor_data_buf, tensor_info->Size(), tensor_buf.data(), tensor_buf.size());
    if (EOK != ret) {
      MS_LOG(ERROR) << "memcpy_s error";
      return false;
    }
    prim->set_attr(attr_name, MakeValue(tensor_info));
  } else {
    if (attr_tensor_type == onnx::TensorProto_DataType_DOUBLE) {
      size_t data_size = sizeof(double);
      double attr_value = 0.0;
      ret = memcpy_s(&attr_value, data_size, tensor_buf.data(), tensor_buf.size());
      if (EOK != ret) {
        MS_LOG(ERROR) << "memcpy_s error";
        return false;
      }
      prim->set_attr(attr_name, MakeValue<double>(attr_value));
    } else if (attr_tensor_type == onnx::TensorProto_DataType_INT64) {
      size_t data_size = sizeof(int64_t);
      int64_t attr_value = 0;
      ret = memcpy_s(&attr_value, data_size, tensor_buf.data(), tensor_buf.size());
      if (EOK != ret) {
        MS_LOG(ERROR) << "memcpy_s error";
        return false;
      }
      prim->set_attr(attr_name, MakeValue<int64_t>(attr_value));
    } else if (attr_tensor_type == onnx::TensorProto_DataType_BOOL) {
      size_t data_size = sizeof(bool);
      bool attr_value = false;
      ret = memcpy_s(&attr_value, data_size, tensor_buf.data(), tensor_buf.size());
      if (EOK != ret) {
        MS_LOG(ERROR) << "memcpy_s error";
        return false;
      }
      prim->set_attr(attr_name, MakeValue<bool>(attr_value));
    }
  }
  return ret == EOK;
}

bool AnfImporterFromMindir::GetAttrValueForCNode(const PrimitivePtr &prim, const onnx::AttributeProto &attr_proto) {
  if (prim == nullptr) {
    return false;
  }
  const std::string &attr_name = attr_proto.name();
  if (!attr_proto.has_ref_attr_name()) {
    MS_LOG(ERROR) << "CNode parse attr type has no ref_attr_name";
    return false;
  }
  const std::string &ref_attr_name = attr_proto.ref_attr_name();
  if (ref_attr_name.empty()) {
    MS_LOG(ERROR) << "ref_attr_name is empty";
    return false;
  }
  string type = "";
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
        return ObtainCNodeAttrInTypeForm(prim, attr_name, attr_tensor);
      }
      case FORM_PARSE_SCALAR: {
        auto res = ObtainCNodeAttrInScalarForm(attr_tensor);
        kv.insert(std::pair<string, ValuePtr>(attr_tensor.name(), res));
        break;
      }
      case FORM_PARSE_TENSOR: {
        return ObtainCNodeAttrInTensorForm(prim, attr_name, attr_tensor);
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

bool AnfImporterFromMindir::ObtainValueNodeInTensorForm(const std::string &value_node_name,
                                                        const onnx::TensorProto &attr_tensor) {
  const int attr_tensor_type = attr_tensor.data_type();
  std::vector<int> shape;
  for (int i = 0; i < attr_tensor.dims_size(); ++i) {
    shape.push_back(attr_tensor.dims(i));
  }
  std::vector<int> shape_vector;
  (void)std::transform(shape.begin(), shape.end(), std::back_inserter(shape_vector),
                       [](const int32_t &value) { return static_cast<int>(value); });
  ParamValueLitePtr param_value = std::make_shared<ParamValueLite>();
  param_value->set_tensor_shape(shape_vector);
  param_value->set_tensor_type(kDefaultValueSwitchMap[attr_tensor_type]);
  const std::string &tensor_buf = attr_tensor.raw_data();
  auto tensor_data = new (std::nothrow) char[tensor_buf.size()];
  if (tensor_data == nullptr) {
    MS_LOG(ERROR) << "Tensor_data is nullptr";
    return false;
  }
  auto ret = memcpy_s(tensor_data, tensor_buf.size(), tensor_buf.data(), tensor_buf.size());
  if (ret != EOK) {
    delete[] tensor_data;
    MS_LOG(ERROR) << "Memcpy error: " << ret;
    return false;
  }
  param_value->SetTensorData(tensor_data, tensor_buf.size());
  auto new_value_node = NewValueNode(MakeValue(param_value));
  if (new_value_node == nullptr) {
    MS_LOG(ERROR) << "Make valuenode fail";
    return false;
  }
  auto type_ptr = TypeIdToType(kDefaultValueSwitchMap[attr_tensor_type]);
  std::vector<int64_t> shape_vector_int64;
  (void)std::transform(shape.begin(), shape.end(), std::back_inserter(shape_vector_int64),
                       [](const int32_t &value) { return static_cast<int64_t>(value); });
  auto abstract_tensor = std::make_shared<abstract::AbstractTensor>(type_ptr, shape_vector_int64);
  new_value_node->set_abstract(abstract_tensor);
  anfnode_build_map_[value_node_name] = new_value_node;
  return true;
}

bool AnfImporterFromMindir::ObtainValueNodeInTypeForm(const std::string &value_node_name,
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

bool AnfImporterFromMindir::GetAttrValueForValueNode(const std::string &value_node_name,
                                                     const onnx::AttributeProto &attr_proto) {
  if (!attr_proto.has_ref_attr_name()) {
    MS_LOG(ERROR) << "CNode parse attr type has no ref_attr_name";
    return false;
  }
  const std::string &ref_attr_name = attr_proto.ref_attr_name();
  if (ref_attr_name.empty()) {
    MS_LOG(ERROR) << "ref_attr_name is empty";
    return false;
  }
  string type = "";
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

bool AnfImporterFromMindir::BuildValueNodeForFuncGraph(const onnx::NodeProto &node_proto) {
  const std::string &value_node_name = node_proto.output(0);
  const onnx::AttributeProto &attr_proto = node_proto.attribute(0);
  if (!attr_proto.has_ref_attr_name()) {
    MS_LOG(ERROR) << "parse ValueNode  don't have ref_attr_name";
    return false;
  }
  return GetAttrValueForValueNode(value_node_name, attr_proto);
}

std::unordered_map<std::string, abstract::AbstractTensorPtr> AnfImporterFromMindir::GetAbstractForCNode(
  const onnx::AttributeProto &attr_proto) {
  std::unordered_map<std::string, abstract::AbstractTensorPtr> kv;
  for (int i = 0; i < attr_proto.tensors_size(); i++) {
    std::vector<int> shape;
    const onnx::TensorProto &attr_tensor = attr_proto.tensors(i);
    for (int j = 0; j < attr_tensor.dims_size(); ++j) {
      shape.push_back(attr_tensor.dims(j));
    }
    std::vector<int64_t> shape_vector;
    (void)std::transform(shape.begin(), shape.end(), std::back_inserter(shape_vector),
                         [](const int32_t &value) { return static_cast<int64_t>(value); });
    auto type_ptr = TypeIdToType(kDefaultValueSwitchMap[attr_tensor.data_type()]);
    auto abstract_tensor = std::make_shared<abstract::AbstractTensor>(type_ptr, shape_vector);
    kv.insert(std::pair<string, abstract::AbstractTensorPtr>(attr_tensor.name(), abstract_tensor));
  }
  return kv;
}

CNodePtr AnfImporterFromMindir::BuildCNodeForFuncGraph(const FuncGraphPtr &outputFuncGraph,
                                                       const onnx::NodeProto &node_proto,
                                                       const schema::QuantType &quantType) {
  static bool interrupt = false;
  if (outputFuncGraph == nullptr) {
    MS_LOG(ERROR) << "output funcgraph is nullptr";
    return nullptr;
  }
  if (!node_proto.has_op_type()) {
    MS_LOG(ERROR) << "Get CNode op_type failed!";
    return nullptr;
  }
  const std::string &node_name = node_proto.output(0);
  const std::string &fullname_with_scope = node_proto.domain();
  const std::string &node_type = node_proto.op_type();
  PrimitivePtr prim = std::make_shared<mindspore::Primitive>(node_type);
  if (prim == nullptr) {
    MS_LOG(ERROR) << "new primitive failed";
    return nullptr;
  }
  prim->set_instance_name(node_type);
  std::unordered_map<std::string, abstract::AbstractTensorPtr> kv;
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
  for (int i = 0; i < node_proto.input_size(); ++i) {
    const std::string &input_name = node_proto.input(i);
    if (anfnode_build_map_.find(input_name) == anfnode_build_map_.end()) {
      if (!interrupt) {
        MS_LOG(ERROR) << node_name << " input " << i << input_name << "can't find in nodes have parsed";
        interrupt = true;
      }
      inputs.push_back(nullptr);
    } else {
      inputs.push_back(anfnode_build_map_[input_name]);
    }
  }
  auto primitivec_ptr = PrimitiveC::Create(*prim, inputs, quantType);
  if (primitivec_ptr == nullptr || interrupt) {
    interrupt = true;
    if (primitivec_ptr == nullptr) {
      NoSupportOp::GetInstance()->InsertOp(prim->name());
    }
    return nullptr;
  }
  inputs.insert(inputs.begin(), NewValueNode(primitivec_ptr));
  CNodePtr cnode_ptr = outputFuncGraph->NewCNode(inputs);
  if (cnode_ptr == nullptr) {
    interrupt = true;
    MS_LOG(ERROR) << "funcgraph new cnode failed";
    return nullptr;
  }
  if (kv.empty()) {
    AbstractBasePtrList elem;
    for (size_t index = 1; index < cnode_ptr->inputs().size(); ++index) {
      elem.push_back(cnode_ptr->input(index)->abstract());
    }
    cnode_ptr->set_abstract(std::make_shared<abstract::AbstractTuple>(elem));
  } else if (1 == kv.size()) {
    auto iter = kv.begin();
    cnode_ptr->set_abstract(iter->second);
  } else {
    auto abstract = ParserAttrShape(shape_ref_attr_name, kv);
    cnode_ptr->set_abstract(abstract);
  }

  cnode_ptr->set_fullname_with_scope(fullname_with_scope);
  anfnode_build_map_[node_name] = cnode_ptr;
  return cnode_ptr;
}

bool AnfImporterFromMindir::BuildReturnForFuncGraph(const FuncGraphPtr &outputFuncGraph,
                                                    const onnx::GraphProto &importProto, const CNodePtr &cnode_ptr) {
  if (outputFuncGraph == nullptr || cnode_ptr == nullptr) {
    MS_LOG(ERROR) << "output funcgraph or cnode is nullptr";
    return false;
  }
  std::vector<AnfNodePtr> inputs;
  if (importProto.output_size() > 1) {
    inputs.clear();
    auto primitiveT = std::make_unique<schema::PrimitiveT>();
    MS_ASSERT(primitiveT != nullptr);
    primitiveT->value.type = schema::PrimitiveType_MakeTuple;
    std::shared_ptr<PrimitiveC> primitivec_ptr = std::make_shared<PrimitiveC>(primitiveT.release());
    MS_ASSERT(primitivec_ptr != nullptr);
    inputs.push_back(NewValueNode(primitivec_ptr));
    AbstractBasePtrList elem;
    for (int out_size = 0; out_size < importProto.output_size(); ++out_size) {
      const onnx::ValueInfoProto &output_node = importProto.output(out_size);
      const std::string &out_tuple = output_node.name();
      inputs.push_back(anfnode_build_map_[out_tuple]);
      if (anfnode_build_map_[out_tuple] == nullptr) {
        MS_LOG(ERROR) << "AnfNode is nullptr";
        return false;
      }
      elem.push_back(anfnode_build_map_[out_tuple]->abstract());
    }
    auto maketuple_ptr = outputFuncGraph->NewCNode(inputs);
    if (maketuple_ptr == nullptr) {
      MS_LOG(ERROR) << "maketuple_ptr is nullptr";
      return false;
    }
    maketuple_ptr->set_abstract(std::make_shared<abstract::AbstractTuple>(elem));
    inputs.clear();
    auto primReturn = std::make_unique<schema::PrimitiveT>();
    MS_ASSERT(primReturn != nullptr);
    primReturn->value.type = schema::PrimitiveType_Return;
    std::shared_ptr<PrimitiveC> primitive_return_value_ptr = std::make_shared<PrimitiveC>(primReturn.release());
    MS_ASSERT(primitive_return_value_ptr != nullptr);
    inputs.push_back(NewValueNode(primitive_return_value_ptr));
    inputs.push_back(maketuple_ptr);
    auto return_node = outputFuncGraph->NewCNode(inputs);
    if (return_node == nullptr) {
      MS_LOG(ERROR) << "funcgraph new cnode failed";
      return false;
    }
    outputFuncGraph->set_return(return_node);
    MS_LOG(INFO) << "Construct funcgraph finined, all success.";
  } else {
    const onnx::ValueInfoProto &output_node = importProto.output(0);
    const onnx::TypeProto &output_typeproto = output_node.type();
    int output_type = output_typeproto.tensor_type().elem_type();
    std::vector<int> output_shape;
    for (int i = 0; i < output_typeproto.tensor_type().shape().dim_size(); ++i) {
      output_shape.push_back(output_typeproto.tensor_type().shape().dim(i).dim_value());
    }
    std::vector<int64_t> shape_vector;
    (void)std::transform(output_shape.begin(), output_shape.end(), std::back_inserter(shape_vector),
                         [](const int32_t &value) { return static_cast<int64_t>(value); });
    auto type_ptr = TypeIdToType(kDefaultValueSwitchMap[output_type]);
    auto abstract_tensor = std::make_shared<abstract::AbstractTensor>(type_ptr, shape_vector);
    inputs.clear();
    auto primReturn = std::make_unique<schema::PrimitiveT>();
    MS_ASSERT(primReturn != nullptr);
    primReturn->value.type = schema::PrimitiveType_Return;
    std::shared_ptr<PrimitiveC> primitiveTReturnValuePtr = std::make_shared<PrimitiveC>(primReturn.release());
    MS_ASSERT(primitiveTReturnValuePtr != nullptr);
    inputs.push_back(NewValueNode(primitiveTReturnValuePtr));
    inputs.push_back(cnode_ptr);
    auto return_node = outputFuncGraph->NewCNode(inputs);
    if (return_node == nullptr) {
      MS_LOG(ERROR) << "funcgraph new cnode failed";
      return false;
    }
    return_node->set_abstract(abstract_tensor);
    outputFuncGraph->set_return(return_node);
    MS_LOG(INFO) << "Construct funcgraph finined, all success!";
  }
  return true;
}

int AnfImporterFromMindir::ImportNodesForGraph(const FuncGraphPtr &outputFuncGraph, const onnx::GraphProto &importProto,
                                               const schema::QuantType &quantType) {
  if (outputFuncGraph == nullptr) {
    MS_LOG(ERROR) << "funcgraph is nullptr";
    return RET_NULL_PTR;
  }
  MS_LOG(INFO) << "The CNdoe size : " << importProto.node_size();
  CNodePtr cnode_ptr = nullptr;
  CNodePtr last_cnode_ptr = nullptr;
  int status = RET_OK;
  NoSupportOp::GetInstance()->SetFmkType("MINDIR");
  for (int i = 0; i < importProto.node_size(); ++i) {
    const onnx::NodeProto &node_proto = importProto.node(i);
    const std::string &node_type = node_proto.op_type();
    if (node_type == kConstantValueNode) {
      if (status == RET_OK && !BuildValueNodeForFuncGraph(node_proto)) {
        MS_LOG(ERROR) << "Build ValueNode for funcgraph fail at index: : " << i;
        status = RET_ERROR;
      }
      continue;
    }
    cnode_ptr = BuildCNodeForFuncGraph(outputFuncGraph, node_proto, quantType);
    if (cnode_ptr == nullptr) {
      MS_LOG(ERROR) << "Build CNode for funcgraph fail at index: : " << i;
      return RET_ERROR;
    }

    auto primitive_c = GetValueNode<std::shared_ptr<PrimitiveC>>(cnode_ptr->input(0));
    if (primitive_c == nullptr) {
      MS_LOG(ERROR) << "primitive_c is nullptr";
      return RET_ERROR;
    }
  }
  if (status != RET_OK) {
    return status;
  }
  if (!BuildReturnForFuncGraph(outputFuncGraph, importProto, cnode_ptr)) {
    MS_LOG(ERROR) << "Build ReturnNode for funcgraph failed";
    status = RET_ERROR;
  }
  return status;
}

int AnfImporterFromMindir::BuildFuncGraph(const FuncGraphPtr &outputFuncGraph, const onnx::GraphProto &importProto,
                                          const schema::QuantType &quantType) {
  if (outputFuncGraph == nullptr) {
    MS_LOG(ERROR) << "fundgraph is nullptr";
    return RET_NULL_PTR;
  }
  GraphDebugInfoPtr debug_info_ptr = outputFuncGraph->debug_info();
  if (debug_info_ptr == nullptr) {
    MS_LOG(ERROR) << "funcgraph's debug info is nullptr";
    return RET_NULL_PTR;
  }
  if (importProto.has_name()) {
    debug_info_ptr->set_name(importProto.name());
  } else {
    MS_LOG(INFO) << "FuncGraph under converting has not name!";
  }

  auto status = ImportParametersForGraph(outputFuncGraph, importProto);
  if (status != RET_OK) {
    return status;
  }
  return ImportNodesForGraph(outputFuncGraph, importProto, quantType);
}

int AnfImporterFromMindir::ParseModelConfigureInfo(const onnx::ModelProto &model_proto) {
  if (!model_proto.has_producer_name()) {
    MS_LOG(ERROR) << "Parse model producer name from pb file failed!";
    return RET_GRAPH_FILE_ERR;
  }
  producer_name_ = model_proto.producer_name();

  if (!model_proto.has_model_version()) {
    MS_LOG(ERROR) << "Parse model producer version from pb file failed!";
    return RET_GRAPH_FILE_ERR;
  }
  model_version_ = model_proto.model_version();

  if (!model_proto.has_ir_version()) {
    MS_LOG(ERROR) << "Parse model version from pb file failed!";
    return RET_GRAPH_FILE_ERR;
  }
  ir_version_ = model_proto.ir_version();
  return RET_OK;
}

int AnfImporterFromMindir::Import(const converter::Flags *flag) {
  if (flag->trainModel) {
    func_graph_ = LoadMindIR(flag->modelFile, true);
    if (func_graph_ != nullptr) {
      return RET_OK;
    } else {
      MS_LOG(ERROR) << "Parse new mind_ir proto failed, Trying old onnx format";
    }
  }
  onnx_model_ = ReadOnnxFromBinary(flag->modelFile);
  if (onnx_model_ == nullptr) {
    MS_LOG(DEBUG) << "Parse model failed, which is not an old mindir model";
    func_graph_ = LoadMindIR(flag->modelFile, true);
    if (func_graph_ == nullptr) {
      MS_LOG(ERROR) << "The mindir model cannot be parsed, which may not match proto file.";
      return RET_GRAPH_FILE_ERR;
    }
    return RET_OK;
  }
  FuncGraphPtr dstGraph = std::make_shared<mindspore::FuncGraph>();
  if (dstGraph == nullptr) {
    MS_LOG(ERROR) << "funcgraph is nullptr";
    return RET_NULL_PTR;
  }
  int status = ParseModelConfigureInfo(*onnx_model_);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Parse configuration info for pb file failed!";
    return status;
  }
  auto quantType = flag->quantType;
  const onnx::GraphProto &graphBuild = onnx_model_->graph();
  status = BuildFuncGraph(dstGraph, graphBuild, quantType);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Build funcgraph failed!";
    func_graph_ = nullptr;
    return status;
  }
  func_graph_ = dstGraph;
  MS_LOG(INFO) << "Parse pb to build FuncGraph Success!";
  return RET_OK;
}

onnx::ModelProto *AnfImporterFromMindir::ReadOnnxFromBinary(const std::string &model_path) {
  auto onnx_model = new (std::nothrow) onnx::ModelProto;
  if (onnx_model == nullptr) {
    MS_LOG(ERROR) << "New onnx ModelProto failed!";
    return nullptr;
  }
  if (RET_OK != ValidateFileStr(model_path, ".mindir")) {
    MS_LOG(ERROR) << "INPUT ILLEGAL: modelFile must be *.mindir";
    delete (onnx_model);
    return nullptr;
  }
  if (ReadProtoFromBinaryFile((const char *)model_path.c_str(), onnx_model) != RET_OK) {
    MS_LOG(ERROR) << "Read onnx model file failed, which is not a matched onnx model";
    delete (onnx_model);
    return nullptr;
  }
  return onnx_model;
}

FuncGraphPtr AnfImporterFromMindir::GetResult() { return this->func_graph_; }
}  // namespace mindspore::lite

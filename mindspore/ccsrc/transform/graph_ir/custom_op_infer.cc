/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include <string>
#include <vector>
#ifdef MSLITE_ENABLE_GRAPH_KERNEL
#include <fstream>
#include <memory>
#include <unordered_map>
#include "nlohmann/json.hpp"
#include "transform/graph_ir/transform_util.h"
#include "backend/common/graph_kernel/model/op_register.h"
#include "backend/common/graph_kernel/core/value_depend_op_utils.h"
#endif
#include "utils/log_adapter.h"
#include "graph/operator.h"

namespace mindspore {
namespace transform {
namespace {
bool InferOffline(const ge::Operator &op, std::vector<ge::TensorDesc> *outputs_info) {
  if (outputs_info == nullptr) {
    return false;
  }

  // output_shapes
  std::vector<std::vector<int64_t>> output_shapes;
  if (op.GetAttr("output_shapes", output_shapes) != ge::GRAPH_SUCCESS) {
    return false;
  }

  // output_formats
  std::vector<int32_t> output_formats;
  if (op.GetAttr("output_formats", output_formats) != ge::GRAPH_SUCCESS ||
      output_formats.size() != output_shapes.size()) {
    return false;
  }

  // output_types
  std::vector<int32_t> output_types;
  if (op.GetAttr("output_types", output_types) != ge::GRAPH_SUCCESS || output_types.size() != output_shapes.size()) {
    return false;
  }

  for (size_t i = 0; i < output_shapes.size(); ++i) {
    (void)outputs_info->emplace_back(ge::Shape(output_shapes[i]), static_cast<ge::Format>(output_formats[i]),
                                     static_cast<ge::DataType>(output_types[i]));
  }
  return true;
}

std::string GetCustomOpName(const ge::Operator &op) {
  std::string res;
  ge::AscendString op_name;
  if (op.GetName(op_name) != ge::GRAPH_SUCCESS) {
    return res;
  }
  return op_name.GetString();
}

std::string GetCustomOpType(const ge::Operator &op) {
  std::string res;
  ge::AscendString op_type;
  if (op.GetOpType(op_type) != ge::GRAPH_SUCCESS) {
    return res;
  }
  return op_type.GetString();
}

std::string GetCustomOpKey(const ge::Operator &op) {
  auto op_name = GetCustomOpName(op);
  auto op_type = GetCustomOpType(op);
  auto op_key = op_name + "(" + op_type + ")";
  return op_key;
}
}  // namespace

#ifdef MSLITE_ENABLE_GRAPH_KERNEL
using mindspore::graphkernel::inner::ConstTensorNode;
using mindspore::graphkernel::inner::DAttrs;
using mindspore::graphkernel::inner::Node;
using mindspore::graphkernel::inner::NodeBase;
using mindspore::graphkernel::inner::NodePtr;
using mindspore::graphkernel::inner::NodePtrList;

namespace {
TypeId ConvertGeDataType(const ge::DataType &type) {
  static std::unordered_map<ge::DataType, TypeId> ge_ms_type = {
    {ge::DataType::DT_FLOAT16, TypeId::kNumberTypeFloat16}, {ge::DataType::DT_FLOAT, TypeId::kNumberTypeFloat32},
    {ge::DataType::DT_DOUBLE, TypeId::kNumberTypeFloat64},  {ge::DataType::DT_INT8, TypeId::kNumberTypeInt8},
    {ge::DataType::DT_INT16, TypeId::kNumberTypeInt16},     {ge::DataType::DT_INT32, TypeId::kNumberTypeInt32},
    {ge::DataType::DT_INT64, TypeId::kNumberTypeInt64},     {ge::DataType::DT_UINT8, TypeId::kNumberTypeUInt8},
    {ge::DataType::DT_UINT16, TypeId::kNumberTypeUInt16},   {ge::DataType::DT_UINT32, TypeId::kNumberTypeUInt32},
    {ge::DataType::DT_UINT64, TypeId::kNumberTypeUInt64},   {ge::DataType::DT_BOOL, TypeId::kNumberTypeBool},
    {ge::DataType::DT_STRING, TypeId::kObjectTypeString},   {ge::DataType::DT_BF16, TypeId::kNumberTypeBFloat16}};
  auto iter = ge_ms_type.find(type);
  if (iter != ge_ms_type.end()) {
    return iter->second;
  }
  return TypeId::kTypeUnknown;
}

using ConvertFunc = std::function<tensor::TensorPtr(const nlohmann::json &)>;

template <typename T, TypeId type_id>
tensor::TensorPtr ConvertSingleValueHelper(const nlohmann::json &value) {
  T v = value;
  ShapeVector shape = {1};
  return std::make_shared<tensor::Tensor>(type_id, shape, &v, type_id);
}

tensor::TensorPtr ConvertSingleValue(const nlohmann::json &value, const std::string &type) {
  // reverse function of SetSingleValue
  static std::unordered_map<std::string, ConvertFunc> convertTable = {
    {"bool", ConvertSingleValueHelper<bool, TypeId::kNumberTypeBool>},
    {"int8", ConvertSingleValueHelper<int8_t, TypeId::kNumberTypeInt8>},
    {"int16", ConvertSingleValueHelper<int16_t, TypeId::kNumberTypeInt16>},
    {"int32", ConvertSingleValueHelper<int32_t, TypeId::kNumberTypeInt32>},
    {"int64", ConvertSingleValueHelper<int64_t, TypeId::kNumberTypeInt64>},
    {"uint8", ConvertSingleValueHelper<uint8_t, TypeId::kNumberTypeUInt8>},
    {"uint16", ConvertSingleValueHelper<uint16_t, TypeId::kNumberTypeUInt16>},
    {"uint32", ConvertSingleValueHelper<uint32_t, TypeId::kNumberTypeUInt32>},
    {"uint64", ConvertSingleValueHelper<uint64_t, TypeId::kNumberTypeUInt64>},
    {"float16", ConvertSingleValueHelper<float, TypeId::kNumberTypeFloat16>},
    {"float32", ConvertSingleValueHelper<float, TypeId::kNumberTypeFloat32>},
    {"float64", ConvertSingleValueHelper<double, TypeId::kNumberTypeFloat64>}};
  if (convertTable.count(type) > 0) {
    ConvertFunc convertFunc = convertTable[type];
    return convertFunc(value);
  }
  MS_LOG(WARNING) << "Fail to convert the single value: " << value << ". type is: " << type;
  return nullptr;
}

tensor::TensorPtr ConvertListValue(const nlohmann::json &value, const std::string &type, const ShapeVector &shape) {
  // reverse function of SetValueList
  tensor::TensorPtr res = nullptr;
  if (type == "int32") {
    std::vector<int32_t> v = value;
    res = std::make_shared<tensor::Tensor>(TypeId::kNumberTypeInt32, shape, &v[0], TypeId::kNumberTypeInt32);
  } else if (type == "int64") {
    std::vector<int64_t> v = value;
    res = std::make_shared<tensor::Tensor>(TypeId::kNumberTypeInt64, shape, &v[0], TypeId::kNumberTypeInt64);
  }
  MS_LOG(WARNING) << "Fail to convert the list value: " << value << ". type is: " << type;
  return res;
}

NodePtrList GetOpInputs(const nlohmann::json &op_desc, const std::unordered_map<std::string, NodePtr> &all_tensors,
                        const std::string &op_name) {
  const auto &value_depend_op_info = graphkernel::ValueDependOpUtils::GetOpIndexInfo();
  bool is_depend_value = value_depend_op_info.find(op_name) != value_depend_op_info.end();
  NodePtrList res;
  for (const auto &input_desc : op_desc["input_desc"]) {
    for (const auto &item : input_desc) {
      std::string name = item["tensor_name"];
      auto iter = all_tensors.find(name);
      if (iter != all_tensors.end()) {
        res.push_back(iter->second);
      } else {
        // const value input
        if (is_depend_value && item.find("value") != item.end()) {
          auto value = item["value"];
          std::string type = item["data_type"];
          ShapeVector shape = item["shape"];
          auto tensor = value.is_array() ? ConvertListValue(value, type, shape) : ConvertSingleValue(value, type);
          if (tensor != nullptr) {
            res.push_back(std::make_shared<ConstTensorNode>(tensor));
            continue;
          }
          MS_LOG(WARNING) << "Fail to parse the const value of tensor [" << name << "]. tensor json is: " << item;
        }
        std::string format = item["format"];
        NodeBase n{ShapeVector(item["shape"]), StringToTypeId(item["data_type"]), format};
        res.push_back(std::make_shared<Node>(n));
      }
    }
  }
  return res;
}

DAttrs GetOpAttr(const nlohmann::json &op_desc) {
  DAttrs res;
  // no attr
  if (op_desc.find("attr") == op_desc.end() || op_desc["attr"].is_null()) {
    return res;
  }
  for (const auto &item : op_desc["attr"]) {
    std::string name = item["name"];
    std::string type = item["data_type"];
    ValuePtr attr_value = nullptr;
    if (type == "str") {
      std::string value = item["value"];
      attr_value = (name == "dst_type" && op_desc["name"] == "Cast") ? StringToType(value) : MakeValue(value);
    } else if (type == "int") {
      int64_t value = item["value"];
      attr_value = MakeValue(value);
    } else if (type == "bool") {
      bool value = item["value"];
      attr_value = MakeValue(value);
    } else if (type == "float") {
      float value = item["value"];
      attr_value = MakeValue(value);
    } else if (type == "listInt") {
      std::vector<int64_t> value = item["value"];
      attr_value = MakeValue(value);
    } else if (type == "listStr") {
      std::vector<std::string> value = item["value"];
      attr_value = MakeValue(value);
    } else {
      MS_LOG(WARNING) << "Fail to parse attr [" << name << "] because its type: " << type
                      << " is not in supported list: [str, int, bool, float, listInt, listStr]. attr json is: " << item;
    }
    if (attr_value != nullptr) {
      res[name] = attr_value;
    }
  }
  return res;
}

bool InferOnline(const ge::Operator &op, const nlohmann::json &js, std::vector<ge::TensorDesc> *outputs_info) {
  if (outputs_info == nullptr) {
    return false;
  }
  std::unordered_map<std::string, NodePtr> all_tensors;
  // iter input_desc: inputs info use the real info pass by GE
  std::vector<nlohmann::json> input_desc = js["input_desc"];
  for (size_t i = 0; i < input_desc.size(); ++i) {
    const auto &item = input_desc[i][0];
    std::string input_name = "x" + std::to_string(i);
    auto ge_desc = op.GetInputDescByName(input_name.c_str());
    std::string format = item["format"];
    NodeBase n{ge_desc.GetShape().GetDims(), ConvertGeDataType(ge_desc.GetDataType()), format};
    MS_LOG(DEBUG) << "input[" << i << "]: " << n.shape << " " << TypeIdToString(n.type);
    all_tensors[item["tensor_name"]] = std::make_shared<Node>(n);
  }

  // iter op_desc: infer each op
  for (const auto &op_desc : js["op_desc"]) {
    std::string op_name = op_desc["name"];
    auto op_ptr = mindspore::graphkernel::inner::OpRegistry::Instance().NewOp(op_name);
    auto op_inputs = GetOpInputs(op_desc, all_tensors, op_name);
    auto op_attr = GetOpAttr(op_desc);
    auto infer_res = op_ptr->Infer(op_inputs, op_attr);
    std::vector<nlohmann::json> op_output_desc = op_desc["output_desc"];
    if (infer_res.size() != op_output_desc.size()) {
      MS_LOG(ERROR) << "For op [" << op_name
                    << "], the length of inferred output shape list is not equal to the length of output_desc list: "
                    << infer_res.size() << " vs " << op_output_desc.size();
      return false;
    }
    for (size_t i = 0; i < op_output_desc.size(); ++i) {
      std::string name = op_output_desc[i]["tensor_name"];
      all_tensors[name] = std::make_shared<Node>(infer_res[i]);
    }
  }

  // iter output_desc: combine the outputs info
  std::vector<nlohmann::json> output_desc = js["output_desc"];
  // format not need infer
  std::vector<int32_t> output_formats;
  if (op.GetAttr("output_formats", output_formats) != ge::GRAPH_SUCCESS ||
      output_formats.size() != output_desc.size()) {
    return false;
  }

  for (size_t i = 0; i < output_desc.size(); ++i) {
    std::string name = output_desc[i]["tensor_name"];
    auto iter = all_tensors.find(name);
    if (iter == all_tensors.end()) {
      MS_LOG(ERROR) << "Tensor [" << name << "] not found in op_desc";
      return false;
    }
    auto shape = iter->second->shape;
    (void)outputs_info->emplace_back(ge::Shape(shape), static_cast<ge::Format>(output_formats[i]),
                                     TransformUtil::ConvertDataType(iter->second->type));
    MS_LOG(DEBUG) << "output[" << i << "]: " << shape << " " << TypeIdToString(iter->second->type);
  }
  return true;
}

bool InputsInfoNotChanged(const ge::Operator &op, const nlohmann::json &js) {
  std::vector<nlohmann::json> input_desc = js["input_desc"];
  for (size_t i = 0; i < input_desc.size(); ++i) {
    std::string input_name = "x" + std::to_string(i);
    auto ge_desc = op.GetInputDescByName(input_name.c_str());
    auto ge_shape = ge_desc.GetShape().GetDims();
    auto ge_type = ge_desc.GetDataType();
    const auto &item = input_desc[i][0];
    ShapeVector ms_shape = item["shape"];
    auto ms_type = StringToTypeId(item["data_type"]);
    if (ge_shape != ms_shape || ConvertGeDataType(ge_type) != ms_type) {
      return false;
    }
  }
  return true;
}

bool Infer(const ge::Operator &op, const std::string &op_key, const std::string &info_path,
           std::vector<ge::TensorDesc> *outputs_info) {
  if (outputs_info == nullptr) {
    return false;
  }

  // read akg info and parse it to json format
  std::ifstream info_str(info_path);
  if (!info_str.is_open()) {
    return false;
  }
  nlohmann::json js;
  info_str >> js;
  info_str.close();

  // 1) if input information not changed, reuse the outputs info saved in op attr 2) else infer online
  if (InputsInfoNotChanged(op, js)) {
    MS_LOG(INFO) << "Infer shape offline for op " << op_key;
    return InferOffline(op, outputs_info);
  }
  MS_LOG(INFO) << "Infer shape online for op " << op_key;
  return InferOnline(op, js, outputs_info);
}
}  // namespace

ge::graphStatus CustomAkgOpInferFunc(ge::Operator &op) {
  auto op_key = GetCustomOpKey(op);
  MS_LOG(INFO) << "Start infer shape for op " << op_key;

  // get akg info path of current op
  std::string info_path;
  auto status = op.GetAttr("info_path", info_path);
  if (status != ge::GRAPH_SUCCESS) {
    return status;
  }

  // infer shape
  std::vector<ge::TensorDesc> outputs_info;
  try {
    if (!Infer(op, op_key, info_path, &outputs_info)) {
      MS_LOG(ERROR) << "Failed infer shape for op " << op_key << ", akg info path: " << info_path;
      return ge::GRAPH_FAILED;
    }
  } catch (std::exception &e) {
    MS_LOG(ERROR) << "Failed infer shape for op " << op_key << ", akg info path: " << info_path
                  << " error message: " << e.what();
    return ge::GRAPH_FAILED;
  }

  // update output desc
  for (size_t i = 0; i < outputs_info.size(); ++i) {
    std::string output_name = "y" + std::to_string(i);
    (void)op.UpdateOutputDesc(output_name, outputs_info[i]);
  }
  MS_LOG(INFO) << "End infer shape for op " << op_key;
  return ge::GRAPH_SUCCESS;
}
#else
ge::graphStatus CustomAkgOpInferFunc(ge::Operator &) { return ge::GRAPH_SUCCESS; }
#endif

ge::graphStatus CustomTbeAicpuOpInferFunc(ge::Operator &op) {
  auto op_key = GetCustomOpKey(op);
  MS_LOG(INFO) << "Start infer shape for op " << op_key;
  std::vector<ge::TensorDesc> outputs_info;
  if (!InferOffline(op, &outputs_info)) {
    MS_LOG(ERROR) << "Failed infer shape for op " << op_key;
    return ge::GRAPH_FAILED;
  }
  // update output desc
  std::vector<std::string> output_names;
  if (op.GetAttr("output_names", output_names) != ge::GRAPH_SUCCESS || output_names.size() != outputs_info.size()) {
    MS_LOG(ERROR) << "For op " << op_key
                  << ", attr 'output_names' size is not equal to outputs_info size: " << output_names.size() << " vs "
                  << outputs_info.size();
    return ge::GRAPH_FAILED;
  }
  for (size_t i = 0; i < outputs_info.size(); ++i) {
    (void)op.UpdateOutputDesc(output_names[i], outputs_info[i]);
  }
  MS_LOG(INFO) << "End infer shape for op " << op_key;
  return ge::GRAPH_SUCCESS;
}
}  // namespace transform
}  // namespace mindspore

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

#include "tools/converter/parser/tf/tf_util.h"
#include <string>
#include <vector>
#include <string_view>
#include <regex>
#include <unordered_map>
#include "src/common/log_adapter.h"
#include "schema/inner/model_generated.h"

namespace mindspore {
namespace lite {
static const std::unordered_map<int, mindspore::TypeId> TF_TYPE_MAP = {
  {tensorflow::DT_INT8, mindspore::kNumberTypeInt8},
  {tensorflow::DT_UINT8, mindspore::kNumberTypeUInt8},
  {tensorflow::DT_INT16, mindspore::kNumberTypeInt16},
  {tensorflow::DT_UINT16, mindspore::kNumberTypeUInt16},
  {tensorflow::DT_INT32, mindspore::kNumberTypeInt32},
  {tensorflow::DT_INT64, mindspore::kNumberTypeInt64},
  {tensorflow::DT_HALF, mindspore::kNumberTypeFloat16},
  {tensorflow::DT_FLOAT, mindspore::kNumberTypeFloat32},
  {tensorflow::DT_DOUBLE, mindspore::kNumberTypeFloat64},
  {tensorflow::DT_COMPLEX64, mindspore::kNumberTypeComplex64},
  {tensorflow::DT_BOOL, mindspore::kNumberTypeBool},
  {tensorflow::DT_STRING, mindspore::kObjectTypeString},
  {tensorflow::DT_VARIANT, mindspore::kObjectTypeTensorType}};

TypeId TensorFlowUtils::GetTFDataType(const tensorflow::DataType &tf_data_type) {
  auto iter = TF_TYPE_MAP.find(tf_data_type);
  if (iter == TF_TYPE_MAP.end()) {
    MS_LOG(WARNING) << "unsupported TF data type: " << tf_data_type;
    return kTypeUnknown;
  }
  return iter->second;
}

bool TensorFlowUtils::FindAttrValue(const tensorflow::NodeDef &node_def, const std::string &attr_name,
                                    tensorflow::AttrValue *attr_value) {
  const google::protobuf::Map<std::string, tensorflow::AttrValue> &attr = node_def.attr();
  const google::protobuf::Map<std::string, tensorflow::AttrValue>::const_iterator it = attr.find(attr_name);
  if (it != attr.end()) {
    *attr_value = it->second;
    return true;
  }
  return false;
}

TypeId TensorFlowUtils::ParseAttrDataType(const tensorflow::NodeDef &node_def, const std::string &attr_name) {
  tensorflow::AttrValue attr_value;
  if (!FindAttrValue(node_def, attr_name, &attr_value)) {
    MS_LOG(ERROR) << "Find attr failed: " << attr_name;
    return kTypeUnknown;
  }
  return GetTFDataType(attr_value.type());
}

bool TensorFlowUtils::DecodeInt64(std::string_view *str_view, uint64_t *value) {
  if (str_view == nullptr || value == nullptr) {
    *value = 0;
    MS_LOG(ERROR) << "str_view or value is nullptr";
    return false;
  }
  auto data = str_view->data();
  const auto end = data + str_view->size();

  const char *next = nullptr;
  uint64_t result = 0;
  for (uint32_t shift = 0; shift <= 63 && data < end; shift += 7) {
    uint64_t byte = *(reinterpret_cast<const unsigned char *>(data));
    data++;
    if (byte & 128) {
      result |= ((byte & 127) << shift);
    } else {
      result |= (byte << shift);
      *value = result;
      next = reinterpret_cast<const char *>(data);
      break;
    }
  }

  if (next == nullptr) {
    return false;
  } else {
    *str_view = std::string_view(next, end - next);
    return true;
  }
}

// convert input_arg in subgraph to node_name[:index] format
std::string TensorFlowUtils::GetFlattenNodeName(const std::string &input_name) {
  std::regex re("\\:+");
  std::vector<std::string> input_splits(std::sregex_token_iterator(input_name.begin(), input_name.end(), re, -1),
                                        std::sregex_token_iterator());
  std::string ret = input_name;
  if (input_splits.size() == 3) {
    if (input_splits[2] == "0") {
      ret = input_splits[0];
    } else {
      ret = input_splits[0] + ":" + input_splits[2];  // multi output node
    }
  }
  return ret;
}

// get referenced node name from input name
std::string TensorFlowUtils::GetNodeName(const std::string &input_name) {
  std::regex re("\\:+");
  std::vector<std::string> input_splits(std::sregex_token_iterator(input_name.begin(), input_name.end(), re, -1),
                                        std::sregex_token_iterator());
  if (input_splits.size() > 1) {
    return input_splits[0];
  }
  return input_name;
}

mindspore::Format TensorFlowUtils::ParseNodeFormat(const tensorflow::NodeDef &node_def) {
  tensorflow::AttrValue attr_value;
  if (!FindAttrValue(node_def, "data_format", &attr_value)) {
    MS_LOG(ERROR) << "Find attr data_format failed";
    return mindspore::Format::NCHW;
  }
  if (attr_value.s() == "NHWC") {
    return mindspore::Format::NHWC;
  }
  return mindspore::Format::NCHW;
}
}  // namespace lite
}  // namespace mindspore

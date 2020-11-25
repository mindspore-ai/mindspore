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
#include <unordered_map>
#include "src/common/log_adapter.h"
#include "schema/inner/model_generated.h"

namespace mindspore {
namespace lite {
static const std::unordered_map<int, mindspore::TypeId> TF_TYPE_MAP = {
  {tensorflow::DT_INT8, mindspore::kNumberTypeInt8},      {tensorflow::DT_UINT8, mindspore::kNumberTypeUInt8},
  {tensorflow::DT_INT16, mindspore::kNumberTypeInt16},    {tensorflow::DT_UINT16, mindspore::kNumberTypeUInt16},
  {tensorflow::DT_INT32, mindspore::kNumberTypeInt32},    {tensorflow::DT_INT64, mindspore::kNumberTypeInt64},
  {tensorflow::DT_HALF, mindspore::kNumberTypeFloat16},   {tensorflow::DT_FLOAT, mindspore::kNumberTypeFloat32},
  {tensorflow::DT_DOUBLE, mindspore::kNumberTypeFloat64}, {tensorflow::DT_COMPLEX64, mindspore::kNumberTypeComplex64},
  {tensorflow::DT_BOOL, mindspore::kNumberTypeBool},      {tensorflow::DT_STRING, mindspore::kObjectTypeString}};

TypeId TensorFlowUtils::GetTFDataType(const tensorflow::DataType &tf_data_type) {
  auto iter = TF_TYPE_MAP.find(tf_data_type);
  if (iter == TF_TYPE_MAP.end()) {
    MS_LOG(ERROR) << "unsupported TF data type: " << tf_data_type;
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
schema::Format TensorFlowUtils::ParseNodeFormat(const tensorflow::NodeDef &node_def) {
  tensorflow::AttrValue attr_value;
  if (!FindAttrValue(node_def, "data_format", &attr_value)) {
    MS_LOG(ERROR) << "Find attr data_format failed";
    return schema::Format_NUM_OF_FORMAT;
  }
  if (attr_value.s() == "NHWC") {
    return schema::Format_NHWC;
  } else if (attr_value.s() == "NCHW") {
    return schema::Format_NCHW;
  }
  return schema::Format_NUM_OF_FORMAT;
}
}  // namespace lite
}  // namespace mindspore

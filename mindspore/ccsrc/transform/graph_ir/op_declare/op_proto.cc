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

#include "transform/graph_ir/op_declare/op_proto.h"
#include <string>
#include <limits>
#include <utility>
#include "graph/types.h"
#include "utils/log_adapter.h"

using ge::DT_BF16;
using ge::DT_BOOL;
using ge::DT_COMPLEX128;
using ge::DT_COMPLEX64;
using ge::DT_DOUBLE;
using ge::DT_DUAL;
using ge::DT_DUAL_SUB_INT8;
using ge::DT_DUAL_SUB_UINT8;
using ge::DT_FLOAT;
using ge::DT_FLOAT16;
using ge::DT_INT16;
using ge::DT_INT2;
using ge::DT_INT32;
using ge::DT_INT4;
using ge::DT_INT64;
using ge::DT_INT8;
using ge::DT_MAX;
using ge::DT_QINT16;
using ge::DT_QINT32;
using ge::DT_QINT8;
using ge::DT_QUINT16;
using ge::DT_QUINT8;
using ge::DT_RESOURCE;
using ge::DT_STRING;
using ge::DT_STRING_REF;
using ge::DT_UINT1;
using ge::DT_UINT16;
using ge::DT_UINT2;
using ge::DT_UINT32;
using ge::DT_UINT64;
using ge::DT_UINT8;
using ge::DT_UNDEFINED;
using ge::DT_VARIANT;

namespace mindspore::transform {
namespace {
std::vector<std::string> SplitString(const std::string &input) {
  std::vector<std::string> words;
  std::string word = "";
  for (char c : input) {
    if (std::isalnum(c) || c == '_') {
      word += c;
    } else if (!word.empty()) {
      words.push_back(word);
      word = "";
    }
  }
  if (!word.empty()) {
    words.push_back(word);
  }
  return words;
}

std::vector<enum ge::DataType> ParseGeTypes(const std::string &tensor_types) {
  static HashMap<std::string, std::vector<enum ge::DataType>> kGeTypeMap = {
    {"DT_BF16", {DT_BF16}},
    {"DT_BOOL", {DT_BOOL}},
    {"DT_COMPLEX128", {DT_COMPLEX128}},
    {"DT_COMPLEX64", {DT_COMPLEX64}},
    {"DT_DOUBLE", {DT_DOUBLE}},
    {"DT_DUAL", {DT_DUAL}},
    {"DT_DUAL_SUB_INT8", {DT_DUAL_SUB_INT8}},
    {"DT_DUAL_SUB_UINT8", {DT_DUAL_SUB_UINT8}},
    {"DT_FLOAT", {DT_FLOAT}},
    {"DT_FLOAT16", {DT_FLOAT16}},
    {"DT_INT16", {DT_INT16}},
    {"DT_INT2", {DT_INT2}},
    {"DT_INT32", {DT_INT32}},
    {"DT_INT4", {DT_INT4}},
    {"DT_INT64", {DT_INT64}},
    {"DT_INT8", {DT_INT8}},
    {"DT_MAX", {DT_MAX}},
    {"DT_QINT16", {DT_QINT16}},
    {"DT_QINT32", {DT_QINT32}},
    {"DT_QINT8", {DT_QINT8}},
    {"DT_QUINT16", {DT_QUINT16}},
    {"DT_QUINT8", {DT_QUINT8}},
    {"DT_RESOURCE", {DT_RESOURCE}},
    {"DT_STRING", {DT_STRING}},
    {"DT_STRING_REF", {DT_STRING_REF}},
    {"DT_UINT1", {DT_UINT1}},
    {"DT_UINT16", {DT_UINT16}},
    {"DT_UINT2", {DT_UINT2}},
    {"DT_UINT32", {DT_UINT32}},
    {"DT_UINT64", {DT_UINT64}},
    {"DT_UINT8", {DT_UINT8}},
    {"DT_UNDEFINED", {DT_UNDEFINED}},
    {"DT_VARIANT", {DT_VARIANT}},
    {"TensorType", {}},
    {"ALL", {DT_BOOL,   DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16,  DT_INT32,
             DT_INT64,  DT_INT8,       DT_QINT16,    DT_QINT32, DT_QINT8, DT_QUINT16, DT_QUINT8, DT_RESOURCE,
             DT_STRING, DT_UINT16,     DT_UINT32,    DT_UINT64, DT_UINT8, DT_BF16}},
    {"BasicType",
     {DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64, DT_INT8, DT_QINT16,
      DT_QINT32, DT_QINT8, DT_QUINT16, DT_QUINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_UINT8, DT_BF16}},
    {"NumberType",
     {DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64, DT_INT8, DT_QINT16,
      DT_QINT32, DT_QINT8, DT_QUINT16, DT_QUINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_UINT8, DT_BF16}},
    {"RealNumberType",
     {DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64, DT_INT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_UINT8,
      DT_BF16}},
    {"IntegerDataType", {DT_INT16, DT_INT32, DT_INT64, DT_INT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_UINT8}},
    {"UnaryDataType", {DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_BF16}},
    {"FloatingDataType", {DT_DOUBLE, DT_FLOAT, DT_FLOAT16}},
    {"IndexNumberType", {DT_INT32, DT_INT64}},
    // ACL: CANN BUGs
    {"REALNUMBERTYPE",
     {DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_INT64, DT_INT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_UINT8,
      DT_BF16}},
    {"DT_FLOAT32", {DT_FLOAT}},
    {"DT_FLOAT64", {DT_DOUBLE}},
    {"ListInt", {}},
    {"DI_UINT16", {DT_UINT16}},
    {"int32", {DT_INT32}},
    {"int8", {DT_INT8}},
  };
  std::vector<enum ge::DataType> result;
  auto split_tensor_types = SplitString(tensor_types);
  for (const auto &n : split_tensor_types) {
    auto iter = kGeTypeMap.find(n);
    if (iter == kGeTypeMap.end()) {
      MS_LOG(WARNING) << "Unknown data type: " << n;
      continue;
    }
    auto &v = iter->second;
    result.insert(result.end(), v.begin(), v.end());
  }
  return result;
}
}  // namespace

OpProto::OpProto(const std::string &name) : name_(name) {}

OpProto &OpProto::SetInput(const std::string &name, const std::string &tensor_type, bool is_optional) {
  input_names_.emplace_back(name);
  input_optional_flags_.push_back(is_optional);
  input_types_.emplace(name, ParseGeTypes(tensor_type));
  return *this;
}

OpProto &OpProto::SetOutput(const std::string &name, const std::string &tensor_type) {
  output_names_.emplace_back(name);
  output_types_.emplace(name, ParseGeTypes(tensor_type));
  return *this;
}

OpProto &OpProto::SetAttr(const std::string &name, bool is_optional) {
  attr_optional_flags_[name] = is_optional;
  return *this;
}

OpProto &OpProto::DoNothing() { return *this; }

size_t OpProto::GetInputIndexByName(const std::string &name) const {
  auto iter = std::find(input_names_.begin(), input_names_.end(), name);
  if (iter != input_names_.end()) {
    return iter - input_names_.begin();
  }
  MS_LOG(WARNING) << "CANN op " << name_ << " cannot find input " << name;
  return std::numeric_limits<size_t>::max();
}

size_t OpProto::GetOutputIndexByName(const std::string &name) const {
  auto iter = std::find(output_names_.begin(), output_names_.end(), name);
  if (iter != output_names_.end()) {
    return iter - output_names_.begin();
  }
  MS_LOG(WARNING) << "CANN op " << name_ << " cannot find output " << name;
  return std::numeric_limits<size_t>::max();
}

bool OpProto::IsInputOptionalTypeByName(const std::string &name) const {
  auto index = GetInputIndexByName(name);
  if (index == std::numeric_limits<size_t>::max()) {
    return false;
  }
  return input_optional_flags_[index];
}

bool OpProto::IsAttrOptionalTypeByName(const std::string &name) const {
  auto iter = attr_optional_flags_.find(name);
  if (iter != attr_optional_flags_.end()) {
    return iter->second;
  }
  MS_LOG(WARNING) << "CANN op " << name_ << " cannot find attr " << name;
  return true;
}

std::vector<enum ge::DataType> OpProto::GetInputTypesByName(const std::string &name) const {
  auto iter = input_types_.find(name);
  if (iter != input_types_.end()) {
    return iter->second;
  }
  MS_LOG(WARNING) << "CANN op " << name_ << " cannot find input " << name;
  return {};
}

std::vector<enum ge::DataType> OpProto::GetOutputTypesByName(const std::string &name) const {
  auto iter = output_types_.find(name);
  if (iter != output_types_.end()) {
    return iter->second;
  }
  MS_LOG(WARNING) << "CANN op " << name_ << " cannot find output " << name;
  return {};
}

OpProtoStorage &OpProtoStorage::GetInstance() {
  static OpProtoStorage instance = {};
  return instance;
}

OpProto &OpProtoStorage::GetOpProto(const std::string &name) {
  auto it = op_proto_map_.find(name);
  if (it == op_proto_map_.end()) {
    it = op_proto_map_.emplace(name, OpProto(name)).first;
  }
  return it->second;
}
}  //  namespace mindspore::transform

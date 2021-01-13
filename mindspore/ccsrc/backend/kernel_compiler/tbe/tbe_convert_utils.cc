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

#include "backend/kernel_compiler/tbe/tbe_convert_utils.h"

#include <unordered_map>
#include <map>
#include <string>

#include "backend/session/anf_runtime_algorithm.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
namespace tbe {
const std::unordered_map<std::string, TypeId> type_str_id_maps = {
  {"float", TypeId::kNumberTypeFloat32},   {"float16", TypeId::kNumberTypeFloat16},
  {"float32", TypeId::kNumberTypeFloat32}, {"float64", TypeId::kNumberTypeFloat64},
  {"int", TypeId::kNumberTypeInt},         {"int8", TypeId::kNumberTypeInt8},
  {"int16", TypeId::kNumberTypeInt16},     {"int32", TypeId::kNumberTypeInt32},
  {"int64", TypeId::kNumberTypeInt64},     {"uint", TypeId::kNumberTypeUInt},
  {"uint8", TypeId::kNumberTypeUInt8},     {"uint16", TypeId::kNumberTypeUInt16},
  {"uint32", TypeId::kNumberTypeUInt32},   {"uint64", TypeId::kNumberTypeUInt64},
  {"bool", TypeId::kNumberTypeBool},       {"", TypeId::kMetaTypeNone},
};

const std::map<TypeId, std::string> type_id_str_maps = {
  {TypeId::kNumberTypeFloat32, "float32"}, {TypeId::kNumberTypeFloat16, "float16"},
  {TypeId::kNumberTypeFloat, "float"},     {TypeId::kNumberTypeFloat64, "float64"},
  {TypeId::kNumberTypeInt, "int"},         {TypeId::kNumberTypeInt8, "int8"},
  {TypeId::kNumberTypeInt16, "int16"},     {TypeId::kNumberTypeInt32, "int32"},
  {TypeId::kNumberTypeInt64, "int64"},     {TypeId::kNumberTypeUInt, "uint"},
  {TypeId::kNumberTypeUInt8, "uint8"},     {TypeId::kNumberTypeUInt16, "uint16"},
  {TypeId::kNumberTypeUInt32, "uint32"},   {TypeId::kNumberTypeUInt64, "uint64"},
  {TypeId::kNumberTypeBool, "int8"},       {TypeId::kMetaTypeNone, ""},
};

const std::map<std::string, std::string> type_str_maps = {
  {"Float32", "float32"}, {"Float16", "float16"}, {"Int8", "int8"},   {"Int16", "int16"},
  {"UInt16", "uint16"},   {"UInt8", "uint8"},     {"Int32", "int32"}, {"UInt32", "uint32"},
  {"Int64", "int64"},     {"UInt64", "uint64"},   {"Bool", "int8"},   {"Float64", "float64"},
};

const std::unordered_map<std::string, size_t> type_nbyte_maps = {
  {"float16", sizeof(float) / 2}, {"float32", sizeof(float)},  {"float64", sizeof(float) * 2},
  {"int8", sizeof(int) / 4},      {"int16", sizeof(int) / 2},  {"int32", sizeof(int)},
  {"int64", sizeof(int) * 2},     {"uint8", sizeof(int) / 4},  {"uint16", sizeof(int) / 2},
  {"uint32", sizeof(int)},        {"uint64", sizeof(int) * 2}, {"bool", sizeof(char)},
};

const std::unordered_map<std::string, FusionType> fusion_type_maps = {
  {"CONVLUTION", FusionType::CONVLUTION}, {"ELEMWISE", FusionType::ELEMWISE}, {"COMMREDUCE", FusionType::COMMREDUCE},
  {"SEGMENT", FusionType::SEGMENT},       {"OPAQUE", FusionType::OPAQUE},
};

TypeId DtypeToTypeId(const std::string &dtypes) {
  auto iter = type_str_id_maps.find(dtypes);
  if (iter == type_str_id_maps.end()) {
    MS_LOG(EXCEPTION) << "Illegal input device dtype: " << dtypes;
  }
  return iter->second;
}

std::string TypeIdToString(TypeId type_id) {
  auto iter = type_id_str_maps.find(type_id);
  if (iter == type_id_str_maps.end()) {
    MS_LOG(EXCEPTION) << "Illegal input dtype: " << TypeIdLabel(type_id);
  }
  return iter->second;
}

size_t GetDtypeNbyte(const std::string &dtypes) {
  auto iter = type_nbyte_maps.find(dtypes);
  if (iter == type_nbyte_maps.end()) {
    MS_LOG(EXCEPTION) << "Illegal input dtype: " << dtypes;
  }
  return iter->second;
}

FusionType GetFusionType(const std::string &pattern) {
  auto iter = fusion_type_maps.find(pattern);
  if (iter == fusion_type_maps.end()) {
    MS_LOG(INFO) << "Illegal fusion pattern: " << pattern;
    return UNKNOWN_FUSION_TYPE;
  }
  return iter->second;
}

std::string GetProcessor(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  std::string device;
  switch (AnfAlgo::GetProcessor(anf_node)) {
    case Processor::AICORE:
      device = kProcessorAiCore;
      break;
    default:
      MS_LOG(INFO) << "Unknown processor type." << anf_node->fullname_with_scope();
      break;
  }
  return device;
}
}  // namespace tbe
}  // namespace kernel
}  // namespace mindspore

/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/hal/device/ge_types_convert.h"
#include "graph/utils/type_utils.h"

namespace {
constexpr auto kInvalidFormat = "RESERVED";
}
namespace mindspore {
namespace device {
namespace ascend {
ge::proto::DataType GeTypesConvert::GetGeDataType(TypeId type_id) {
  static const std::map<TypeId, ge::proto::DataType> data_type_map = {
    {TypeId::kTypeUnknown, ge::proto::DT_UNDEFINED},     {TypeId::kNumberTypeFloat32, ge::proto::DT_FLOAT},
    {TypeId::kNumberTypeFloat16, ge::proto::DT_FLOAT16}, {TypeId::kNumberTypeInt8, ge::proto::DT_INT8},
    {TypeId::kNumberTypeUInt8, ge::proto::DT_UINT8},     {TypeId::kNumberTypeInt16, ge::proto::DT_INT16},
    {TypeId::kNumberTypeUInt16, ge::proto::DT_UINT16},   {TypeId::kNumberTypeInt32, ge::proto::DT_INT32},
    {TypeId::kNumberTypeInt64, ge::proto::DT_INT64},     {TypeId::kNumberTypeUInt32, ge::proto::DT_UINT32},
    {TypeId::kNumberTypeUInt64, ge::proto::DT_UINT64},   {TypeId::kNumberTypeBool, ge::proto::DT_BOOL},
    {TypeId::kNumberTypeFloat64, ge::proto::DT_DOUBLE},  {TypeId::kObjectTypeString, ge::proto::DT_STRING},
  };
  MS_LOG(INFO) << "Vm origin type_id:" << type_id << ": " << TypeIdLabel(type_id);
  auto iter = data_type_map.find(type_id);
  if (iter == data_type_map.end()) {
    MS_LOG(EXCEPTION) << "MindSpore data type:" << TypeIdLabel(type_id) << " can't been found in GE.";
  }
  return iter->second;
}

ge::proto::DataType GeTypesConvert::TransHcclDataTypeToGeDataType(HcclDataType dtype) {
  static const std::map<HcclDataType, ge::proto::DataType> data_type_map = {
    {HcclDataType::HCCL_DATA_TYPE_INT8, ge::proto::DT_INT8},
    {HcclDataType::HCCL_DATA_TYPE_INT16, ge::proto::DT_INT16},
    {HcclDataType::HCCL_DATA_TYPE_INT32, ge::proto::DT_INT32},
    {HcclDataType::HCCL_DATA_TYPE_FP16, ge::proto::DT_FLOAT16},
    {HcclDataType::HCCL_DATA_TYPE_FP32, ge::proto::DT_FLOAT},
    {HcclDataType::HCCL_DATA_TYPE_INT64, ge::proto::DT_INT64},
    {HcclDataType::HCCL_DATA_TYPE_UINT64, ge::proto::DT_UINT64},
    {HcclDataType::HCCL_DATA_TYPE_RESERVED, ge::proto::DT_UNDEFINED},
  };
  MS_LOG(INFO) << "Origin HcclDataType: " << dtype;
  auto iter = data_type_map.find(dtype);
  if (iter == data_type_map.end()) {
    MS_LOG(EXCEPTION) << "HcclDataType: " << dtype << " can't been found in GE.";
  }
  return iter->second;
}

ge::DataType GeTypesConvert::TransTypeIdToGeDataType(TypeId type_id) {
  static const std::map<TypeId, ge::DataType> data_type_map = {
    {TypeId::kNumberTypeFloat, ge::DataType::DT_FLOAT},     {TypeId::kNumberTypeFloat32, ge::DataType::DT_FLOAT},
    {TypeId::kNumberTypeFloat16, ge::DataType::DT_FLOAT16}, {TypeId::kNumberTypeInt8, ge::DataType::DT_INT8},
    {TypeId::kNumberTypeInt16, ge::DataType::DT_INT16},     {TypeId::kNumberTypeUInt16, ge::DataType::DT_UINT16},
    {TypeId::kNumberTypeUInt8, ge::DataType::DT_UINT8},     {TypeId::kNumberTypeInt32, ge::DataType::DT_INT32},
    {TypeId::kNumberTypeInt, ge::DataType::DT_INT32},       {TypeId::kNumberTypeInt64, ge::DataType::DT_INT64},
    {TypeId::kNumberTypeUInt32, ge::DataType::DT_UINT32},   {TypeId::kNumberTypeUInt, ge::DataType::DT_UINT32},
    {TypeId::kNumberTypeUInt64, ge::DataType::DT_UINT64},   {TypeId::kNumberTypeBool, ge::DataType::DT_BOOL},
    {TypeId::kNumberTypeFloat64, ge::DataType::DT_DOUBLE},  {TypeId::kTypeUnknown, ge::DataType::DT_UNDEFINED}};
  auto iter = data_type_map.find(type_id);
  if (iter == data_type_map.end()) {
    MS_LOG(EXCEPTION) << "Invalid data type:" << type_id << ": " << TypeIdLabel(type_id);
  }
  return iter->second;
}

ge::Format GeTypesConvert::GetGeFormat(const std::string &format, size_t shape_size) {
  static constexpr size_t k4dSize = 4;
  static const std::map<std::string, ge::Format> format_map = {
    // default format: nchw, fractal_nz?
    {kOpFormat_DEFAULT, ge::Format::FORMAT_NCHW},
    {kOpFormat_NC1KHKWHWC0, ge::Format::FORMAT_NC1KHKWHWC0},
    {kOpFormat_ND, ge::Format::FORMAT_ND},
    {kOpFormat_NCHW, ge::Format::FORMAT_NCHW},
    {kOpFormat_NHWC, ge::Format::FORMAT_NHWC},
    {kOpFormat_HWCN, ge::Format::FORMAT_HWCN},
    {kOpFormat_NC1HWC0, ge::Format::FORMAT_NC1HWC0},
    {kOpFormat_FRAC_Z, ge::Format::FORMAT_FRACTAL_Z},
    {kOpFormat_FRAC_NZ, ge::Format::FORMAT_FRACTAL_NZ},
    {kOpFormat_C1HWNCoC0, ge::Format::FORMAT_C1HWNCoC0},
    {kOpFormat_NC1HWC0_C04, ge::Format::FORMAT_NC1HWC0_C04},
    {kOpFormat_FRACTAL_Z_C04, ge::Format::FORMAT_FRACTAL_Z_C04},
    {kOpFormat_NDHWC, ge::Format::FORMAT_NDHWC},
    {kOpFormat_NCDHW, ge::Format::FORMAT_NCDHW},
    {kOpFormat_DHWNC, ge::Format::FORMAT_DHWNC},
    {kOpFormat_DHWCN, ge::Format::FORMAT_DHWCN},
    {kOpFormat_NDC1HWC0, ge::Format::FORMAT_NDC1HWC0},
    {kOpFormat_FRACTAL_Z_3D, ge::Format::FORMAT_FRACTAL_Z_3D},
    {kOpFormat_FRACTAL_ZN_LSTM, ge::Format::FORMAT_FRACTAL_ZN_LSTM},
    {kOpFormat_ND_RNN_BIAS, ge::Format::FORMAT_ND_RNN_BIAS},
    {kOpFormat_FRACTAL_ZN_RNN, ge::Format::FORMAT_FRACTAL_ZN_RNN}};
  if (format == kOpFormat_DEFAULT) {
    return shape_size == k4dSize ? ge::Format::FORMAT_NCHW : ge::Format::FORMAT_ND;
  }
  auto iter = format_map.find(format);
  if (iter == format_map.end()) {
    MS_LOG(EXCEPTION) << "Invalid format:" << format;
  }
  return iter->second;
}

std::string GeTypesConvert::GetGeTilingFormat(ge::Format ge_format) {
  auto format_str = ge::TypeUtils::FormatToSerialString(ge_format);
  if (format_str == kInvalidFormat) {
    MS_LOG(EXCEPTION) << "Not support format:" << ge_format;
  }
  return format_str;
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

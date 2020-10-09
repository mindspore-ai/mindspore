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

#include "runtime/device/ascend/ge_types_convert.h"

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
    {TypeId::kNumberTypeFloat64, ge::proto::DT_DOUBLE},
  };
  MS_LOG(INFO) << "Vm origin type_id:" << type_id;
  auto iter = data_type_map.find(type_id);
  if (iter == data_type_map.end()) {
    MS_LOG(EXCEPTION) << "Invalid data type:" << type_id;
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
    {TypeId::kNumberTypeInt64, ge::DataType::DT_DOUBLE},    {TypeId::kTypeUnknown, ge::DataType::DT_UNDEFINED}};
  auto iter = data_type_map.find(type_id);
  if (iter == data_type_map.end()) {
    MS_LOG(EXCEPTION) << "Invalid data type:" << type_id;
  }
  return iter->second;
}

GeFormat GeTypesConvert::GetGeFormat(const std::string &format, size_t shape_size) {
  static const std::map<std::string, GeFormat> format_map = {
    // default format: nchw, fractal_nz?
    {kOpFormat_DEFAULT, kFormat_NCHW},
    {kOpFormat_NC1KHKWHWC0, kFormat_NC1KHKWHWC0},
    {kOpFormat_ND, kFormat_ND},
    {kOpFormat_NCHW, kFormat_NCHW},
    {kOpFormat_NHWC, kFormat_NHWC},
    {kOpFormat_HWCN, kFormat_HWCN},
    {kOpFormat_NC1HWC0, kFormat_NC1HWC0},
    {kOpFormat_FRAC_Z, kFormat_FRACTAL_Z},
    {kOpFormat_FRAC_NZ, kFormat_FRACTAL_NZ},
    {kOpFormat_C1HWNCoC0, kFormat_C1HWNCoC0},
    {kOpFormat_NC1HWC0_C04, kFormat_NC1HWC0_C04},
    {kOpFormat_FRACTAL_Z_C04, kFormat_FRACTAL_Z_C04},
    {kOpFormat_NDHWC, kFormat_NDHWC},
  };
  MS_LOG(INFO) << "GetGeFormat format:" << format << " shape_size:" << shape_size;
  if (format == kOpFormat_DEFAULT) {
    return shape_size == 4 ? kFormat_NCHW : kFormat_ND;
  }
  auto iter = format_map.find(format);
  if (iter == format_map.end()) {
    MS_LOG(EXCEPTION) << "Invalid format:" << format;
  }
  return iter->second;
}

std::string GeTypesConvert::GetGeTilingFormat(GeFormat ge_format) {
  static const std::map<GeFormat, std::string> kFormatToStringMap = {
    {kFormat_NCHW, "NCHW"},
    {kFormat_NHWC, "NHWC"},
    {kFormat_ND, "ND"},
    {kFormat_NC1HWC0, "NC1HWC0"},
    {kFormat_FRACTAL_Z, "FRACTAL_Z"},
    {kFormat_NC1C0HWPAD, "NC1C0HWPAD"},
    {kFormat_NHWC1C0, "NHWC1C0"},
    {kFormat_FSR_NCHW, "FSR_NCHW"},
    {kFormat_FRACTAL_DECONV, "FRACTAL_DECONV"},
    {kFormat_C1HWNC0, "C1HWNC0"},
    {kFormat_FRACTAL_DECONV_TRANSPOSE, "FRACTAL_DECONV_TRANSPOSE"},
    {kFormat_FRACTAL_DECONV_SP_STRIDE_TRANS, "FRACTAL_DECONV_SP_STRIDE_TRANS"},
    {kFormat_NC1HWC0_C04, "NC1HWC0_C04"},
    {kFormat_FRACTAL_Z_C04, "FRACTAL_Z_C04"},
    {kFormat_CHWN, "CHWN"},
    {kFormat_FRACTAL_DECONV_SP_STRIDE8_TRANS, "DECONV_SP_STRIDE8_TRANS"},
    {kFormat_NC1KHKWHWC0, "NC1KHKWHWC0"},
    {kFormat_BN_WEIGHT, "BN_WEIGHT"},
    {kFormat_FILTER_HWCK, "FILTER_HWCK"},
    {kFormat_HWCN, "HWCN"},
    {kFormat_HASHTABLE_LOOKUP_LOOKUPS, "LOOKUP_LOOKUPS"},
    {kFormat_HASHTABLE_LOOKUP_KEYS, "LOOKUP_KEYS"},
    {kFormat_HASHTABLE_LOOKUP_VALUE, "LOOKUP_VALUE"},
    {kFormat_HASHTABLE_LOOKUP_OUTPUT, "LOOKUP_OUTPUT"},
    {kFormat_HASHTABLE_LOOKUP_HITS, "LOOKUP_HITS"},
    {kFormat_MD, "MD"},
    {kFormat_NDHWC, "NDHWC"},
    {kFormat_NCDHW, "NCDHW"},
    {kFormat_DHWCN, "DHWCN"},
    {kFormat_DHWNC, "DHWNC"},
    {kFormat_NDC1HWC0, "NDC1HWC0"},
    {kFormat_FRACTAL_Z_3D, "FRACTAL_Z_3D"},
    {kFormat_FRACTAL_Z_3D_TRANSPOSE, "FRACTAL_Z_3D_TRANSPOSE"},
    {kFormat_C1HWNCoC0, "C1HWNCoC0"},
    {kFormat_FRACTAL_NZ, "FRACTAL_NZ"},
    {kFormat_CN, "CN"},
    {kFormat_NC, "NC"},
    {kFormat_FRACTAL_ZN_LSTM, "FRACTAL_ZN_LSTM"},
    {kFormat_FRACTAL_Z_G, "FRACTAL_Z_G"},
    {kFormat_RESERVED, "FORMAT_RESERVED"},
    {kFormat_ALL, "ALL"}};

  auto iter = kFormatToStringMap.find(ge_format);
  if (iter == kFormatToStringMap.end()) {
    MS_LOG(EXCEPTION) << "Invalid ge_format:" << ge_format;
  }
  return iter->second;
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

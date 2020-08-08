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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_DUMP_GE_DUMP_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_DUMP_GE_DUMP_H_

#include <map>
#include <string>
#include "proto/ge_dtype.pb.h"
#include "ir/dtype/type_id.h"
#include "utils/utils.h"

namespace mindspore {
namespace device {
namespace ascend {
static ge::proto::DataType GetGeDataType(TypeId type_id) {
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

enum GeFormat {
  kFormat_NCHW = 0,   // NCHW
  kFormat_NHWC,       // NHWC
  kFormat_ND,         // Nd Tensor
  kFormat_NC1HWC0,    // NC1HWC0
  kFormat_FRACTAL_Z,  // FRACTAL_Z
  kFormat_NC1C0HWPAD,
  kFormat_NHWC1C0,
  kFormat_FSR_NCHW,
  kFormat_FRACTAL_DECONV,
  kFormat_C1HWNC0,
  kFormat_FRACTAL_DECONV_TRANSPOSE,
  kFormat_FRACTAL_DECONV_SP_STRIDE_TRANS,
  kFormat_NC1HWC0_C04,    // NC1HWC0, C0 =4
  kFormat_FRACTAL_Z_C04,  // FRACZ, C0 =4
  kFormat_CHWN,
  kFormat_FRACTAL_DECONV_SP_STRIDE8_TRANS,
  kFormat_HWCN,
  kFormat_NC1KHKWHWC0,  // KH,KW kernel h& kernel w maxpooling max output format
  kFormat_BN_WEIGHT,
  kFormat_FILTER_HWCK,  // filter input tensor format
  kFormat_HASHTABLE_LOOKUP_LOOKUPS = 20,
  kFormat_HASHTABLE_LOOKUP_KEYS,
  kFormat_HASHTABLE_LOOKUP_VALUE,
  kFormat_HASHTABLE_LOOKUP_OUTPUT,
  kFormat_HASHTABLE_LOOKUP_HITS = 24,
  kFormat_C1HWNCoC0,
  kFormat_MD,
  kFormat_NDHWC,
  kFormat_FRACTAL_ZZ,
  kFormat_FRACTAL_NZ,
  kFormat_NCDHW,
  kFormat_DHWCN,  // 3D filter input tensor format
  kFormat_NDC1HWC0,
  kFormat_FRACTAL_Z_3D,
  kFormat_CN,
  kFormat_NC,
  kFormat_DHWNC,
  kFormat_FRACTAL_Z_3D_TRANSPOSE,  // 3D filter(transpose) input tensor format
  kFormat_RESERVED,
  kFormat_ALL
};

static GeFormat GetGeFormat(const std::string &format, size_t shape_size) {
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
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_DUMP_GE_DUMP_H_

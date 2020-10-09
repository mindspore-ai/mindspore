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
#include "external/graph/types.h"

namespace mindspore {
namespace device {
namespace ascend {
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
  kFormat_FRACTAL_ZN_LSTM,
  kFormat_FRACTAL_Z_G,
  kFormat_RESERVED,
  kFormat_ALL
};

class GeTypesConvert {
 public:
  GeTypesConvert() = default;
  ~GeTypesConvert() = default;
  static ge::proto::DataType GetGeDataType(TypeId type_id);
  static GeFormat GetGeFormat(const std::string &format, size_t shape_size);
  static std::string GetGeTilingFormat(GeFormat ge_format);
  static ge::DataType TransTypeIdToGeDataType(TypeId type_id);
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_DUMP_GE_DUMP_H_

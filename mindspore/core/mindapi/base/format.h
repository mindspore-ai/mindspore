/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_MINDAPI_BASE_FORMAT_H_
#define MINDSPORE_CORE_MINDAPI_BASE_FORMAT_H_

#include <cstdint>
#include <vector>
#include <string>

namespace mindspore {
enum Format : int64_t {
  DEFAULT_FORMAT = -1,
  NCHW = 0,
  NHWC,
  NHWC4,
  HWKC,
  HWCK,
  KCHW,
  CKHW,
  KHWC,
  CHWK,
  HW,
  HW4,
  NC,
  NC4,
  NC4HW4,
  NCDHW,
  NWC,
  NCW,
  NDHWC,
  NC8HW8,
  FRACTAL_NZ,
  ND,         // Nd Tensor
  NC1HWC0,    // NC1HWC0
  FRACTAL_Z,  // FRACTAL_Z
  NC1C0HWPAD,
  NHWC1C0,
  FSR_NCHW,
  FRACTAL_DECONV,
  C1HWNC0,
  FRACTAL_DECONV_TRANSPOSE,
  FRACTAL_DECONV_SP_STRIDE_TRANS,
  NC1HWC0_C04,    // NC1HWC0, C0 is 4
  FRACTAL_Z_C04,  // FRACZ, C0 is 4
  CHWN,
  FRACTAL_DECONV_SP_STRIDE8_TRANS,
  HWCN,
  NC1KHKWHWC0,  // KH,KW kernel h& kernel w maxpooling max output format
  BN_WEIGHT,
  FILTER_HWCK,  // filter input tensor format
  LOOKUP_LOOKUPS,
  LOOKUP_KEYS,
  LOOKUP_VALUE,
  LOOKUP_OUTPUT,
  LOOKUP_HITS,
  C1HWNCoC0,
  MD,
  FRACTAL_ZZ,
  DHWCN,  // 3D filter input tensor format
  NDC1HWC0,
  FRACTAL_Z_3D,
  CN,
  DHWNC,
  FRACTAL_Z_3D_TRANSPOSE,  // 3D filter(transpose) input tensor format
  FRACTAL_ZN_LSTM,
  FRACTAL_Z_G,
  ND_RNN_BIAS,
  FRACTAL_ZN_RNN,
  NYUV,
  NYUV_A,
  NCL,
  NUM_OF_FORMAT
};

inline const std::vector<std::string> &GetFormatNames() {
  static std::vector<std::string> names = {
    "NCHW",
    "NHWC",
    "NHWC4",
    "HWKC",
    "HWCK",
    "KCHW",
    "CKHW",
    "KHWC",
    "CHWK",
    "HW",
    "HW4",
    "NC",
    "NC4",
    "NC4HW4",
    "NCDHW",
    "NWC",
    "NCW",
    "NDHWC",
    "NC8HW8",
    "FRACTAL_NZ",
    "ND",
    "NC1HWC0",
    "FRACTAL_Z",
    "NC1C0HWPAD",
    "NHWC1C0",
    "FSR_NCHW",
    "FRACTAL_DECONV",
    "C1HWNC0",
    "FRACTAL_DECONV_TRANSPOSE",
    "FRACTAL_DECONV_SP_STRIDE_TRANS",
    "NC1HWC0_C04",
    "FRACTAL_Z_C04",
    "CHWN",
    "FRACTAL_DECONV_SP_STRIDE8_TRANS",
    "HWCN",
    "NC1KHKWHWC0",
    "BN_WEIGHT",
    "FILTER_HWCK",
    "LOOKUP_LOOKUPS",
    "LOOKUP_KEYS",
    "LOOKUP_VALUE",
    "LOOKUP_OUTPUT",
    "LOOKUP_HITS",
    "C1HWNCoC0",
    "MD",
    "FRACTAL_ZZ",
    "DHWCN",
    "NDC1HWC0",
    "FRACTAL_Z_3D",
    "CN",
    "DHWNC",
    "FRACTAL_Z_3D_TRANSPOSE",
    "FRACTAL_ZN_LSTM",
    "FRACTAL_Z_G",
    "ND_RNN_BIAS",
    "FRACTAL_ZN_RNN",
    "NYUV",
    "NYUV_A",
    "NCL",
  };
  return names;
}

inline std::string FormatEnumToString(mindspore::Format format) {
  const auto &names = GetFormatNames();
  if (format == mindspore::Format::DEFAULT_FORMAT) {
    return "DefaultFormat";
  }
  if (format < mindspore::NCHW || format >= mindspore::NUM_OF_FORMAT) {
    return "";
  }
  return names[format];
}
}  // namespace mindspore
#endif  // MINDSPORE_CORE_MINDAPI_BASE_FORMAT_H_

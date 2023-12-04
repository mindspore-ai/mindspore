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

#include "kernel/format_utils.h"
#include <algorithm>
#include <map>
#include "utils/log_adapter.h"

namespace mindspore {
namespace kernel {
const std::map<std::string, Format> format_relation_map = {
  {"DefaultFormat", Format::DEFAULT_FORMAT},
  {"NCHW", Format::NCHW},
  {"NHWC", Format::NHWC},
  {"NHWC4", Format::NHWC4},
  {"HWKC", Format::HWKC},
  {"HWCK", Format::HWCK},
  {"KCHW", Format::KCHW},
  {"CKHW", Format::CKHW},
  {"KHWC", Format::KHWC},
  {"CHWK", Format::CHWK},
  {"HW", Format::HW},
  {"HW4", Format::HW4},
  {"NC", Format::NC},
  {"NC4", Format::NC4},
  {"NC4HW4", Format::NC4HW4},
  {"NCDHW", Format::NCDHW},
  {"NWC", Format::NWC},
  {"NCW", Format::NCW},
  {"NDHWC", Format::NDHWC},
  {"NC8HW8", Format::NC8HW8},
  {"FRACTAL_NZ", Format::FRACTAL_NZ},
  {"ND", Format::ND},
  {"NC1HWC0", Format::NC1HWC0},
  {"FRACTAL_Z", Format::FRACTAL_Z},
  {"NC1C0HWPAD", Format::NC1C0HWPAD},
  {"NHWC1C0", Format::NHWC1C0},
  {"FSR_NCHW", Format::FSR_NCHW},
  {"FRACTAL_DECONV", Format::FRACTAL_DECONV},
  {"C1HWNC0", Format::C1HWNC0},
  {"FRACTAL_DECONV_TRANSPOSE", Format::FRACTAL_DECONV_TRANSPOSE},
  {"FRACTAL_DECONV_SP_STRIDE_TRANS", Format::FRACTAL_DECONV_SP_STRIDE_TRANS},
  {"NC1HWC0_C04", Format::NC1HWC0_C04},
  {"FRACTAL_Z_C04", Format::FRACTAL_Z_C04},
  {"CHWN", Format::CHWN},
  {"FRACTAL_DECONV_SP_STRIDE8_TRANS", Format::FRACTAL_DECONV_SP_STRIDE8_TRANS},
  {"HWCN", Format::HWCN},
  {"NC1KHKWHWC0", Format::NC1KHKWHWC0},
  {"BN_WEIGHT", Format::BN_WEIGHT},
  {"FILTER_HWCK", Format::FILTER_HWCK},
  {"LOOKUP_LOOKUPS", Format::LOOKUP_LOOKUPS},
  {"LOOKUP_KEYS", Format::LOOKUP_KEYS},
  {"LOOKUP_VALUE", Format::LOOKUP_VALUE},
  {"LOOKUP_OUTPUT", Format::LOOKUP_OUTPUT},
  {"LOOKUP_HITS", Format::LOOKUP_HITS},
  {"C1HWNCoC0", Format::C1HWNCoC0},
  {"MD", Format::MD},
  {"FRACTAL_ZZ", Format::FRACTAL_ZZ},
  {"DHWCN", Format::DHWCN},
  {"NDC1HWC0", Format::NDC1HWC0},
  {"FRACTAL_Z_3D", Format::FRACTAL_Z_3D},
  {"CN", Format::CN},
  {"DHWNC", Format::DHWNC},
  {"FRACTAL_Z_3D_TRANSPOSE", Format::FRACTAL_Z_3D_TRANSPOSE},
  {"FRACTAL_ZN_LSTM", Format::FRACTAL_ZN_LSTM},
  {"FRACTAL_Z_G", Format::FRACTAL_Z_G},
  {"ND_RNN_BIAS", Format::ND_RNN_BIAS},
  {"FRACTAL_ZN_RNN", Format::FRACTAL_ZN_RNN},
  {"NYUV", Format::NYUV},
  {"NYUV_A", Format::NYUV_A},
  {"NCL", Format::NCL}};

Format GetFormatFromStrToEnum(const std::string &format_str) {
  auto iter = format_relation_map.find(format_str);
  if (iter != format_relation_map.end()) {
    return iter->second;
  }
  MS_LOG(DEBUG) << "The data format " << format_str << " can not be converted to enum.";
  return Format::DEFAULT_FORMAT;
}

std::string GetFormatFromEnumToStr(Format format) {
  const std::string format_str = "DefaultFormat";
  auto iter = std::find_if(format_relation_map.begin(), format_relation_map.end(),
                           [format](auto item) { return item.second == format; });
  if (iter != format_relation_map.end()) {
    return iter->first;
  }
  MS_LOG(WARNING) << "The data format " << format << " can not be converted to string.";
  return format_str;
}
}  // namespace kernel
}  // namespace mindspore

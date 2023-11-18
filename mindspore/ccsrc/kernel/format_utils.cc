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
const std::map<std::string, Format> format_relation_map = {{"DefaultFormat", Format::DEFAULT_FORMAT},
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
                                                           {"NUM_OF_FORMAT", Format::NUM_OF_FORMAT},
                                                           {"NCDHW", Format::NCDHW},
                                                           {"NWC", Format::NWC},
                                                           {"NCW", Format::NCW},
                                                           {"NDHWC", Format::NDHWC}};

Format GetFormatFromStrToEnum(const std::string &format_str) {
  auto iter = format_relation_map.find(format_str);
  if (iter != format_relation_map.end()) {
    return iter->second;
  }
  MS_LOG(DEBUG) << "The data format " << format_str << " can not be converted to enum.";
  return Format::DEFAULT_FORMAT;
}

std::string GetFormatFromEnumToStr(Format format) {
  std::string format_str = "DefaultFormat";
  auto iter = std::find_if(format_relation_map.begin(), format_relation_map.end(),
                           [format](auto item) { return item.second == format; });
  if (iter != format_relation_map.end()) {
    return iter->first;
  }
  return format_str;
}
}  // namespace kernel
}  // namespace mindspore

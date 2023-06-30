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
  NHWC = 1,
  NHWC4 = 2,
  HWKC = 3,
  HWCK = 4,
  KCHW = 5,
  CKHW = 6,
  KHWC = 7,
  CHWK = 8,
  HW = 9,
  HW4 = 10,
  NC = 11,
  NC4 = 12,
  NC4HW4 = 13,
  NUM_OF_FORMAT = 14,
  NCDHW = 15,
  NWC = 16,
  NCW = 17,
  NDHWC = 18,
  NC8HW8 = 19
};

inline std::string FormatEnumToString(mindspore::Format format) {
  static std::vector<std::string> names = {
    "NCHW", "NHWC", "NHWC4", "HWKC", "HWCK",   "KCHW",          "CKHW",  "KHWC", "CHWK",
    "HW",   "HW4",  "NC",    "NC4",  "NC4HW4", "NUM_OF_FORMAT", "NCDHW", "NWC",  "NCW",
  };
  if (format == mindspore::Format::DEFAULT_FORMAT) {
    return "DefaultFormat";
  }
  if (format < mindspore::NCHW || format > mindspore::NCW) {
    return "";
  }
  return names[format];
}
}  // namespace mindspore
#endif  // MINDSPORE_CORE_MINDAPI_BASE_FORMAT_H_

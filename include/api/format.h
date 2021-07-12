/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
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

#ifndef MINDSPORE_INCLUDE_API_FORMAT_H
#define MINDSPORE_INCLUDE_API_FORMAT_H

#include <cstdint>

namespace mindspore {
enum Format : int64_t {
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
  NCW = 17
};
}  // namespace mindspore
#endif  // MINDSPORE_INCLUDE_API_FORMAT_H

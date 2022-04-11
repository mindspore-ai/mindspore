/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_SHAPE_UTILS_INFO_H_
#define MINDSPORE_SHAPE_UTILS_INFO_H_

#include "mindapi/base/shape_vector.h"

namespace mindspore {
inline size_t SizeOf(const ShapeVector &shape) {
  int64_t data_size = 1;
  for (auto dim : shape) {
    if (dim < 0) {
      // For dynamic shape which has negative dimensions, data size should be zero.
      return 0;
    }
    data_size *= dim;
  }
  return static_cast<size_t>(data_size);
}
}  // namespace mindspore

#endif  // MINDSPORE_SHAPE_UTILS_INFO_H_

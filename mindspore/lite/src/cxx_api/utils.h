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
#include <limits.h>
#include <vector>
#include "src/tensor.h"

namespace mindspore {
static std::vector<int32_t> TruncateShape(const std::vector<int64_t> &shape, enum TypeId type, size_t data_len) {
  std::vector<int32_t> empty;
  std::vector<int32_t> truncated_shape;
  size_t element_size = lite::DataTypeSize(type);
  for (auto i : shape) {
    if (i < 0 || i > INT_MAX || element_size > INT_MAX / static_cast<size_t>(i)) {
      MS_LOG(ERROR) << "Invalid shape.";
      return empty;
    } else {
      element_size *= static_cast<size_t>(i);
      truncated_shape.push_back(static_cast<int32_t>(i));
    }
  }
  if (element_size != data_len) {
    MS_LOG(ERROR) << "Invalid data size.";
    return empty;
  }
  return truncated_shape;
}

}  // namespace mindspore

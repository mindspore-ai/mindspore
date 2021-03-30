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

#ifndef MINDSPORE_LITE_SRC_CXX_API_TENSOR_UTILS_H
#define MINDSPORE_LITE_SRC_CXX_API_TENSOR_UTILS_H

#include <limits.h>
#include <vector>
#include "ir/dtype/type_id.h"

namespace mindspore {
static std::vector<int32_t> TruncateShape(const std::vector<int64_t> &shape, enum TypeId type, size_t data_len,
                                          bool verify_size) {
  std::vector<int32_t> empty;
  if (shape.empty()) {
    return empty;
  }
  std::vector<int32_t> truncated_shape;
  truncated_shape.resize(shape.size());
  size_t element_size = lite::DataTypeSize(type);
  for (size_t i = 0; i < shape.size(); i++) {
    auto dim = shape[i];
    if (dim < 0 || dim > INT_MAX || element_size > INT_MAX / static_cast<size_t>(dim)) {
      MS_LOG(ERROR) << "Invalid shape.";
      return empty;
    } else {
      element_size *= static_cast<size_t>(dim);
      truncated_shape[i] = static_cast<int32_t>(dim);
    }
  }
  if (verify_size) {
    if (element_size != data_len) {
      MS_LOG(ERROR) << "Invalid data size.";
      return empty;
    }
  }
  return truncated_shape;
}
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_CXX_API_TENSOR_UTILS_H

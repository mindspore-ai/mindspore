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
#include "src/cxx_api/tensor/tensor_impl.h"
#include <numeric>
#include <memory>
#include <algorithm>
#include <string>
#include <vector>
#include <functional>
#include "include/api/types.h"
#include "include/api/status.h"
#include "src/cxx_api/utils.h"
#include "src/common/log_adapter.h"

namespace mindspore {
MSTensor::Impl::Impl(const std::string &name, enum DataType type, const std::vector<int64_t> &shape, const void *data,
                     size_t data_len) {
  std::vector<int32_t> truncated_shape = TruncateShape(shape, static_cast<enum TypeId>(type), data_len, true);
  if (truncated_shape.empty() && !(shape.empty())) {
    lite_tensor_ = nullptr;
  } else {
    lite_tensor_ = new (std::nothrow) lite::Tensor(name, static_cast<enum TypeId>(type), truncated_shape, data);
  }
}

}  // namespace mindspore

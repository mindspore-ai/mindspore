/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CORE_UTILS_TENSOR_CONSTRUCT_UTILS_H_
#define MINDSPORE_CORE_UTILS_TENSOR_CONSTRUCT_UTILS_H_
#include <vector>
#include "ir/tensor.h"
namespace mindspore {
template <typename T>
void SetTensorData(void *data, T num, size_t data_length) {
  MS_EXCEPTION_IF_NULL(data);
  auto tensor_data = reinterpret_cast<T *>(data);
  MS_EXCEPTION_IF_NULL(tensor_data);
  for (size_t index = 0; index < data_length; ++index) {
    *tensor_data = num;
    ++tensor_data;
  }
}
class TensorConstructUtils {
 public:
  static tensor::TensorPtr CreateZerosTensor(TypeId type, const std::vector<int64_t> &shape);
  static tensor::TensorPtr CreateOnesTensor(TypeId type, const std::vector<int64_t> &shape);
  static tensor::TensorPtr CreateTensor(TypeId type, const std::vector<int64_t> &shape, void *data);
};
}  // namespace mindspore
#endif  // MINDSPORE_CORE_UTILS_TENSOR_CONSTRUCT_UTILS_H_

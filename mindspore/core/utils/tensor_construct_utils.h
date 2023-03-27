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
#include <algorithm>
#include "ir/tensor.h"
#include "ir/dtype/type_id.h"
namespace mindspore {
template <typename T>
void SetTensorData(void *data, const T &num, size_t data_length) {
  MS_EXCEPTION_IF_NULL(data);
  auto tensor_data = reinterpret_cast<T *>(data);
  std::fill(tensor_data, tensor_data + data_length, num);
}
class MS_CORE_API TensorConstructUtils {
 public:
  static tensor::TensorPtr CreateZerosTensor(const TypePtr &type, const std::vector<int64_t> &shape);
  static tensor::TensorPtr CreateOnesTensor(const TypePtr &type, const std::vector<int64_t> &shape,
                                            bool skip_exception = false);
  static tensor::TensorPtr CreateTensor(const TypePtr &type, const std::vector<int64_t> &shape, void *data);

 private:
  static TypeId ExtractTypeId(const TypePtr &type);
};
}  // namespace mindspore
#endif  // MINDSPORE_CORE_UTILS_TENSOR_CONSTRUCT_UTILS_H_

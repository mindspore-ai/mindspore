/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_UTILS_TENSOR_UTILS_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_UTILS_TENSOR_UTILS_H_

#include <vector>
#include <string>

#include "include/api/types.h"
#include "ir/tensor.h"

namespace mindspore {
class TensorUtils {
 public:
  // MSTensor <-> TensorPtr
  static std::vector<mindspore::tensor::TensorPtr> MSTensorToTensorPtr(const std::vector<MSTensor> &ms_tensors);
  static std::vector<MSTensor> TensorPtrToMSTensor(std::vector<mindspore::tensor::TensorPtr> tensor_ptrs,
                                                   const std::vector<std::string> &tensor_names);

  // TensorPtr <-> Tensor
  static std::vector<mindspore::tensor::TensorPtr> TensorToTensorPtr(
    const std::vector<mindspore::tensor::Tensor> &tensors);
  static std::vector<mindspore::tensor::Tensor> TensorPtrToTensor(
    const std::vector<mindspore::tensor::TensorPtr> &tensor_ptrs);
};
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_EXTENDRT_UTILS_TENSOR_UTILS_H_

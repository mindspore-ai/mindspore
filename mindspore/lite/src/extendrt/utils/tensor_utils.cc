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

#include <memory>
#include <algorithm>

#include "extendrt/utils/tensor_utils.h"

namespace mindspore {
std::vector<mindspore::tensor::TensorPtr> TensorUtils::MSTensorToTensorPtr(const std::vector<MSTensor> &ms_tensors) {
  std::vector<mindspore::tensor::TensorPtr> tensor_ptrs;

  for (auto ms_tensor : ms_tensors) {
    auto data_type = ms_tensor.DataType();
    auto type_id = static_cast<mindspore::TypeId>(data_type);
    auto shape = ms_tensor.Shape();
    auto data = ms_tensor.MutableData();
    auto data_size = ms_tensor.DataSize();
    auto tensor_ptr = std::make_shared<mindspore::tensor::Tensor>(type_id, shape, data, data_size);
    tensor_ptrs.push_back(tensor_ptr);
  }
  return tensor_ptrs;
}

std::vector<MSTensor> TensorUtils::TensorPtrToMSTensor(std::vector<mindspore::tensor::TensorPtr> tensor_ptrs,
                                                       const std::vector<std::string> &tensor_names) {
  std::vector<MSTensor> ms_tensors;

  for (size_t i = 0; i < tensor_ptrs.size(); i++) {
    auto graph_tensor = tensor_ptrs[i];
    std::string graph_tensor_name = tensor_names[i];
    auto type_id = graph_tensor->data_type_c();
    auto data_type = static_cast<mindspore::DataType>(type_id);
    auto ms_tensor_ptr = MSTensor::CreateRefTensor(graph_tensor_name, data_type, graph_tensor->shape_c(),
                                                   graph_tensor->data_c(), graph_tensor->Size());
    if (ms_tensor_ptr == nullptr) {
      MS_LOG_WARNING << "Failed to create input tensor ";
      return {};
    }
    ms_tensors.push_back(*ms_tensor_ptr);
    delete ms_tensor_ptr;
  }

  return ms_tensors;
}

std::vector<mindspore::tensor::TensorPtr> TensorUtils::TensorToTensorPtr(
  const std::vector<mindspore::tensor::Tensor> &tensors) {
  std::vector<mindspore::tensor::TensorPtr> tensor_ptrs;
  for (auto &tensor : tensors) {
    auto type_id = static_cast<TypeId>(tensor.data_type_c());
    auto shape = tensor.shape_c();
    auto data = tensor.data_c();
    auto data_size = tensor.Size();
    auto tensor_ptr = std::make_shared<mindspore::tensor::Tensor>(type_id, shape, data, data_size);
    tensor_ptrs.push_back(tensor_ptr);
  }
  return tensor_ptrs;
}

std::vector<mindspore::tensor::Tensor> TensorUtils::TensorPtrToTensor(
  const std::vector<mindspore::tensor::TensorPtr> &tensor_ptrs) {
  std::vector<mindspore::tensor::Tensor> tensors;
  std::transform(tensor_ptrs.begin(), tensor_ptrs.end(), std::back_inserter(tensors),
                 [](mindspore::tensor::TensorPtr tensor_ptr) { return mindspore::tensor::Tensor(*tensor_ptr); });
  return tensors;
}
}  // namespace mindspore

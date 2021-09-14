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

#include "src/cxx_api/tensor_utils.h"
#include "src/common/log_adapter.h"
#include "src/tensor.h"

namespace mindspore {
std::vector<int32_t> TruncateShape(const std::vector<int64_t> &shape, enum TypeId type, size_t data_len,
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
    if (dim < 0 || dim > INT_MAX || (dim != 0 && element_size > INT_MAX / static_cast<size_t>(dim))) {
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
Status LiteTensorToMSTensor(tensor::MSTensor *srcTensor, MSTensor *dstTensor, bool fromSession) {
  auto impl = std::make_shared<MSTensor::Impl>(srcTensor);
  if (impl == nullptr || impl->lite_tensor() == nullptr) {
    MS_LOG(ERROR) << "Create tensor failed.";
    return kLiteError;
  }
  impl->set_from_session(fromSession);
  auto tensor = MSTensor(impl);
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "Create tensor failed.";
    return kLiteError;
  }
  *dstTensor = tensor;
  return kSuccess;
}

std::vector<MSTensor> LiteTensorsToMSTensors(const std::vector<mindspore::tensor::MSTensor *> &srcTensors,
                                             bool fromSession) {
  std::vector<MSTensor> dstTensors;
  dstTensors.reserve(srcTensors.size());
  for (auto inTensor : srcTensors) {
    MSTensor tensor;
    auto status = LiteTensorToMSTensor(inTensor, &tensor, fromSession);
    if (status != kSuccess) {
      return {};
    }
    dstTensors.emplace_back(tensor);
  }
  return dstTensors;
}
}  // namespace mindspore

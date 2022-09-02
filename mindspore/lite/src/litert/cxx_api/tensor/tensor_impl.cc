/**
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

#include "src/litert/cxx_api/tensor/tensor_impl.h"
#include <cstddef>
#include <numeric>
#include <memory>
#include <algorithm>
#include <string>
#include <vector>
#include <functional>
#include "src/litert/cxx_api/tensor_utils.h"
#include "src/tensor.h"
#include "src/common/string_utils.h"

namespace mindspore {
using mindspore::lite::RET_OK;

std::shared_ptr<LiteTensorImpl> LiteTensorImpl::CreateTensorImpl(const std::string &name, enum DataType type,
                                                                 const std::vector<int64_t> &shape, const void *data,
                                                                 size_t data_len) {
  std::vector<int32_t> truncated_shape;
  if (data_len == 0) {
    truncated_shape = TruncateShape(shape, static_cast<enum TypeId>(type), data_len, false);
  } else {
    truncated_shape = TruncateShape(shape, static_cast<enum TypeId>(type), data_len, true);
  }
  if (truncated_shape.empty() && !(shape.empty())) {
    MS_LOG(ERROR) << "Invalid shape for creating tensor.";
    return nullptr;
  }
  auto lite_tensor = lite::Tensor::CreateTensor(name, static_cast<enum TypeId>(type), truncated_shape, data, data_len);
  if (lite_tensor == nullptr) {
    MS_LOG(ERROR) << "Failed to allocate lite tensor.";
    return nullptr;
  }
  auto impl = std::make_shared<LiteTensorImpl>(lite_tensor);
  if (impl == nullptr) {
    MS_LOG(ERROR) << "Failed to allocate tensor impl.";
    return nullptr;
  }
  impl->set_from_session(false);
  return impl;
}

std::shared_ptr<LiteTensorImpl> LiteTensorImpl::CreateTensorImplByDeepCopy(const std::string &name, enum DataType type,
                                                                           const std::vector<int64_t> &shape,
                                                                           const void *data, size_t data_len) {
  std::vector<int32_t> truncated_shape;
  truncated_shape = TruncateShape(shape, static_cast<enum TypeId>(type), data_len, false);
  auto lite_tensor =
    lite::Tensor::CreateTensorByDeepCopy(name, static_cast<enum TypeId>(type), truncated_shape, data, data_len);
  if (lite_tensor == nullptr) {
    MS_LOG(ERROR) << "Failed to allocate lite tensor.";
    return nullptr;
  }
  auto impl = std::make_shared<LiteTensorImpl>(lite_tensor);
  if (impl == nullptr) {
    MS_LOG(ERROR) << "Failed to allocate tensor impl.";
    return nullptr;
  }
  impl->set_from_session(false);
  impl->set_own_data(true);
  return impl;
}

void LiteTensorImpl::SetDeviceData(void *data) {
  if (lite_tensor_ == nullptr) {
    MS_LOG(ERROR) << "Invalid tensor.";
    return;
  }
  lite_tensor_->set_device_data(data);
}

void *LiteTensorImpl::GetDeviceData() {
  if (lite_tensor_ == nullptr) {
    MS_LOG(ERROR) << "Invalid tensor.";
    return nullptr;
  }
  return lite_tensor_->device_data();
}

#ifndef STRING_KERNEL_CLIP
std::shared_ptr<LiteTensorImpl> LiteTensorImpl::StringsToTensorImpl(const std::string &name,
                                                                    const std::vector<std::string> &str) {
  auto lite_tensor = new (std::nothrow) lite::Tensor();
  if (lite_tensor == nullptr) {
    MS_LOG(ERROR) << "Failed to allocate lite tensor.";
    return nullptr;
  }
  lite_tensor->set_tensor_name(name);
  std::vector<lite::StringPack> all_pack;
  for (auto &input : str) {
    lite::StringPack pack = {static_cast<int>(input.length()), input.data()};
    all_pack.push_back(pack);
  }
  auto ret = lite::WriteStringsToTensor(lite_tensor, all_pack);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Convert strings to tensor failed.";
    delete lite_tensor;
    return nullptr;
  }
  auto impl = std::make_shared<LiteTensorImpl>(lite_tensor);
  if (impl == nullptr) {
    delete lite_tensor;
    MS_LOG(ERROR) << "Failed to allocate tensor impl.";
    return nullptr;
  }
  impl->set_own_data(true);
  impl->set_from_session(false);
  return impl;
}

std::vector<std::string> LiteTensorImpl::TensorImplToStrings(const std::shared_ptr<LiteTensorImpl> &impl) {
  std::vector<std::string> empty;
  auto lite_tensor = impl->lite_tensor();
  if (lite_tensor == nullptr) {
    MS_LOG(ERROR) << "Invalid tensor impl.";
    return empty;
  }
  const void *ptr = lite_tensor->data();
  std::vector<lite::StringPack> all_pack = lite::ParseStringBuffer(ptr);
  std::vector<std::string> result(all_pack.size());
  std::transform(all_pack.begin(), all_pack.end(), result.begin(), [](lite::StringPack &pack) {
    std::string str(pack.data, pack.len);
    return str;
  });
  return result;
}
#endif
}  // namespace mindspore

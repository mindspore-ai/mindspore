/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_TENSOR_FUTURE_H_
#define MINDSPORE_MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_TENSOR_FUTURE_H_

#include <utility>
#include <memory>
#include "ir/tensor.h"
#include "include/utils/visible.h"

namespace mindspore {
namespace pynative {
using DeviceAddress = device::DeviceAddress;
using DeviceAddressPtr = device::DeviceAddressPtr;
using DeviceAddressFutureData = tensor::FutureData<device::DeviceAddress>;
using DeviceAddressFutureDataPtr = std::shared_ptr<DeviceAddressFutureData>;

class COMMON_EXPORT DeviceAddressFuture : public tensor::FutureBase<device::DeviceAddress> {
 public:
  explicit DeviceAddressFuture(std::future<DeviceAddressFutureDataPtr> future)
      : tensor::FutureBase<device::DeviceAddress>(std::move(future)) {}
  ~DeviceAddressFuture() override;
  std::shared_ptr<device::DeviceAddress> Get() override;

 private:
  std::once_flag once_flag_;
};

class COMMON_EXPORT DeviceAddressPromise {
 public:
  explicit DeviceAddressPromise(std::promise<DeviceAddressFutureDataPtr> promise) : promise_(std::move(promise)) {}
  ~DeviceAddressPromise() = default;

  void SetValue(const DeviceAddressFutureDataPtr &data);
  std::future<DeviceAddressFutureDataPtr> GetFuture();

 private:
  std::promise<DeviceAddressFutureDataPtr> promise_;
  std::once_flag once_flag_;
};
using DeviceAddressPromisePtr = std::shared_ptr<DeviceAddressPromise>;
}  // namespace pynative
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_TENSOR_FUTURE_H_

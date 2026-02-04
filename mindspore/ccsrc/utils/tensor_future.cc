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

#include "include/utils/tensor_future.h"
#include "pybind_api/gil_scoped_long_running.h"

namespace mindspore {
namespace pynative {
DeviceAddressFuture::~DeviceAddressFuture() {
  if (future_.valid()) {
    try {
      (void)future_.get();
    } catch (const std::exception &e) {
      MS_LOG(INFO) << "Find error and ignore when destroy future:" << e.what();
    }
  }
}

std::shared_ptr<DeviceAddress> DeviceAddressFuture::Get() {
  std::call_once(once_flag_, [this]() {
    if (future_.valid()) {
      // cppcheck-suppress unreadVariable
      GilReleaseWithCheck gil_release;
      future_data_ = future_.get();
    }
  });

  if (future_data_ != nullptr) {
    if (future_data_->GetException() != nullptr) {
      MS_LOG(DEBUG) << "Found exception in future data. Rethrow the exception.";
      std::rethrow_exception(future_data_->GetException());
    }
    return future_data_->GetData();
  } else {
    MS_LOG(ERROR) << "The future data is null";
    return nullptr;
  }
}

void DeviceAddressPromise::SetValue(const DeviceAddressFutureDataPtr &data) {
  std::call_once(once_flag_, [this, &data]() { promise_.set_value(data); });
}

std::future<DeviceAddressFutureDataPtr> DeviceAddressPromise::GetFuture() { return promise_.get_future(); }
}  // namespace pynative
}  // namespace mindspore

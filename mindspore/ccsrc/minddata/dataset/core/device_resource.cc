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

#include "minddata/dataset/core/device_resource.h"

namespace mindspore {
namespace dataset {

Status DeviceResource::InitResource(uint32_t) {
  return Status(StatusCode::kMDUnexpectedError,
                "Is this a valid device? If yes, please implement this InitResource() in the derived class.");
}

Status DeviceResource::FinalizeResource() {
  return Status(StatusCode::kMDUnexpectedError,
                "Is this a valid device? If yes, please implement this FinalizeResource() in the derived class.");
}

Status DeviceResource::Sink(const mindspore::MSTensor &host_input, std::shared_ptr<DeviceTensor> *device_input) {
  return Status(StatusCode::kMDUnexpectedError,
                "Is this a valid device whose device memory is available? If yes, please implement this Sink() in the "
                "derived class.");
}

Status DeviceResource::Pop(const std::shared_ptr<DeviceTensor> &device_output, std::shared_ptr<Tensor> *host_output) {
  return Status(StatusCode::kMDUnexpectedError,
                "Is this a valid device whose device memory is available? If yes, please implement this Pop() in the "
                "derived class.");
}

Status DeviceResource::DeviceDataRelease() {
  return Status(
    StatusCode::kMDUnexpectedError,
    "Is this a valid device whose device memory is available? If yes, please implement this DeviceDataRelease() in the "
    "derived class.");
}

std::shared_ptr<void> DeviceResource::GetInstance() {
  MS_LOG(ERROR) << "Is this a device which contains a processor object? If yes, please implement this GetInstance() in "
                   "the derived class";
  return nullptr;
}

void *DeviceResource::GetContext() {
  MS_LOG(ERROR)
    << "Is this a device which contains context resource? If yes, please implement GetContext() in the derived class";
  return nullptr;
}

void *DeviceResource::GetStream() {
  MS_LOG(ERROR)
    << "Is this a device which contains stream resource? If yes, please implement GetContext() in the derived class";
  return nullptr;
}

}  // namespace dataset
}  // namespace mindspore

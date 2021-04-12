/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/core/de_tensor.h"
#include "minddata/dataset/core/device_tensor.h"
#include "minddata/dataset/include/constants.h"
#include "minddata/dataset/core/data_type.h"
#include "minddata/dataset/core/type_id.h"
#include "mindspore/core/ir/dtype/type_id.h"
#include "utils/hashing.h"
#ifndef ENABLE_ANDROID
#include "utils/log_adapter.h"
#define EXCEPTION_IF_NULL(ptr) MS_EXCEPTION_IF_NULL(ptr)
#else
#include "mindspore/lite/src/common/log_adapter.h"
#define EXCEPTION_IF_NULL(ptr) MS_ASSERT((ptr) != nullptr)
#endif

namespace mindspore {
namespace dataset {

DETensor::DETensor(std::shared_ptr<dataset::Tensor> tensor_impl)
    : tensor_impl_(tensor_impl),
      name_("MindDataTensor"),
      type_(static_cast<mindspore::DataType>(DETypeToMSType(tensor_impl_->type()))),
      shape_(tensor_impl_->shape().AsVector()),
      is_device_(false) {}

#ifndef ENABLE_ANDROID
DETensor::DETensor(std::shared_ptr<dataset::DeviceTensor> device_tensor_impl, bool is_device)
    : device_tensor_impl_(device_tensor_impl), name_("MindDataDeviceTensor"), is_device_(is_device) {
  // The sequence of shape_ is (width, widthStride, height, heightStride) in Dvpp module
  // We need to add [1]widthStride and [3]heightStride, which are actual YUV image shape, into shape_ attribute
  uint8_t flag = 0;
  for (auto &i : device_tensor_impl->GetYuvStrideShape()) {
    if (flag % 2 == 1) {
      int64_t j = static_cast<int64_t>(i);
      shape_.emplace_back(j);
    }
    ++flag;
  }
  std::reverse(shape_.begin(), shape_.end());
  MS_LOG(INFO) << "This is a YUV420 format image, one pixel takes 1.5 bytes. Therefore, the shape of"
               << " image is in (H, W) format. You can search for more information about YUV420 format";
}
#endif

const std::string &DETensor::Name() const { return name_; }

enum mindspore::DataType DETensor::DataType() const {
#ifndef ENABLE_ANDROID
  if (is_device_) {
    EXCEPTION_IF_NULL(device_tensor_impl_);
    return static_cast<mindspore::DataType>(DETypeToMSType(device_tensor_impl_->DeviceDataType()));
  }
#endif
  EXCEPTION_IF_NULL(tensor_impl_);
  return static_cast<mindspore::DataType>(DETypeToMSType(tensor_impl_->type()));
}

size_t DETensor::DataSize() const {
#ifndef ENABLE_ANDROID
  if (is_device_) {
    EXCEPTION_IF_NULL(device_tensor_impl_);
    return device_tensor_impl_->DeviceDataSize();
  }
#endif
  EXCEPTION_IF_NULL(tensor_impl_);
  return static_cast<uint32_t>(tensor_impl_->SizeInBytes());
}

const std::vector<int64_t> &DETensor::Shape() const { return shape_; }

std::shared_ptr<const void> DETensor::Data() const {
#ifndef ENABLE_ANDROID
  if (is_device_) {
    EXCEPTION_IF_NULL(device_tensor_impl_);
    return std::shared_ptr<const void>(device_tensor_impl_->GetHostBuffer(), [](const void *) {});
  }
#endif
  return std::shared_ptr<const void>(tensor_impl_->GetBuffer(), [](const void *) {});
}

void *DETensor::MutableData() {
#ifndef ENABLE_ANDROID
  if (is_device_) {
    EXCEPTION_IF_NULL(device_tensor_impl_);
    return static_cast<void *>(device_tensor_impl_->GetDeviceMutableBuffer());
  }
#endif
  EXCEPTION_IF_NULL(tensor_impl_);
  return static_cast<void *>(tensor_impl_->GetMutableBuffer());
}

bool DETensor::IsDevice() const { return is_device_; }

std::shared_ptr<mindspore::MSTensor::Impl> DETensor::Clone() const {
#ifndef ENABLE_ANDROID
  if (is_device_) {
    EXCEPTION_IF_NULL(device_tensor_impl_);
    return std::make_shared<DETensor>(device_tensor_impl_, is_device_);
  }
#endif
  return std::make_shared<DETensor>(tensor_impl_);
}
}  // namespace dataset
}  // namespace mindspore

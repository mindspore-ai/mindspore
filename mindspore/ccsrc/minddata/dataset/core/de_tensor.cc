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

#include "minddata/dataset/core/de_tensor.h"
#include "minddata/dataset/core/constants.h"
#include "minddata/dataset/core/data_type.h"
#include "minddata/dataset/include/type_id.h"
#include "mindspore/core/ir/dtype/type_id.h"
#include "utils/hashing.h"
#ifndef ENABLE_ANDROID
#include "utils/log_adapter.h"
#define CHECK_IF_NULL(ptr) MS_EXCEPTION_IF_NULL(ptr)
#else
#include "mindspore/lite/src/common/log_adapter.h"
#define CHECK_IF_NULL(ptr) MS_ASSERT((ptr) != nullptr)
#endif

namespace mindspore {
namespace dataset {

DETensor::DETensor(std::shared_ptr<dataset::Tensor> tensor_impl)
    : tensor_impl_(tensor_impl),
      name_("MindDataTensor"),
      type_(static_cast<mindspore::DataType>(DETypeToMSType(tensor_impl_->type()))),
      shape_(tensor_impl_->shape().AsVector()) {}

const std::string &DETensor::Name() const { return name_; }

enum mindspore::DataType DETensor::DataType() const {
  CHECK_IF_NULL(tensor_impl_);
  return static_cast<mindspore::DataType>(DETypeToMSType(tensor_impl_->type()));
}

size_t DETensor::DataSize() const {
  CHECK_IF_NULL(tensor_impl_);
  return static_cast<size_t>(tensor_impl_->SizeInBytes());
}

const std::vector<int64_t> &DETensor::Shape() const { return shape_; }

std::shared_ptr<const void> DETensor::Data() const {
  return std::shared_ptr<const void>(tensor_impl_->GetBuffer(), [](const void *) {});
}

void *DETensor::MutableData() {
  CHECK_IF_NULL(tensor_impl_);
  return tensor_impl_->GetMutableBuffer();
}

bool DETensor::IsDevice() const { return false; }

std::shared_ptr<mindspore::MSTensor::Impl> DETensor::Clone() const { return std::make_shared<DETensor>(tensor_impl_); }
}  // namespace dataset
}  // namespace mindspore

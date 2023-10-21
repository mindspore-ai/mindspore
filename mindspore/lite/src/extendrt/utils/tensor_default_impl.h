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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_UTILS_TENSOR_DEFAULT_IMPL_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_UTILS_TENSOR_DEFAULT_IMPL_H_

#include <vector>
#include <string>
#include <memory>
#include <functional>

#include "include/api/types.h"
#include "ir/tensor.h"
#include "include/backend/device_address.h"
#include "common/utils.h"
#include "common/mutable_tensor_impl.h"
#include "src/extendrt/kernel/ascend/plugin/ascend_allocator_plugin.h"

namespace mindspore {
class TensorDefaultImpl : public MutableTensorImpl {
 public:
  TensorDefaultImpl() = default;
  TensorDefaultImpl(const std::string &name, enum DataType type, const std::vector<int64_t> &shape,
                    bool is_acl_host = false)
      : name_(name), type_(type), shape_(shape) {
    if (!is_acl_host) {
      buffer_.SetData(nullptr, 0);
      data_ = buffer_.Data();
    } else {
      is_acl_host_ = true;
      size_t data_type_size = lite::DataTypeSize(TypeId(type));
      size_t data_buf_size = data_type_size;
      for (auto s : shape) {
        data_buf_size *= s;
      }
      void *data_buf_ptr = kernel::AscendAllocatorPlugin::GetInstance().MallocHost(data_buf_size);
      data_ = data_buf_ptr;
      own_data_ = false;
    }
  }

  TensorDefaultImpl(const std::string &name, enum DataType type, const std::vector<int64_t> &shape, const void *data,
                    size_t data_len, bool ref_data, bool own_data)
      : name_(name), type_(type), shape_(shape) {
    if (ref_data) {
      data_ = data;
      own_data_ = own_data;
    } else {
      if (data == nullptr) {
        data_len = 0;
      }
      buffer_.SetData(data, data_len);
      data_ = buffer_.Data();
    }
  }
  ~TensorDefaultImpl() {
    if (own_data_ && data_ != nullptr && data_ != buffer_.Data()) {
      free(const_cast<void *>(data_));
    }
    if (device_data_ != nullptr && own_data_) {
      MS_LOG(INFO) << "free device data in tensor default impl.";
      kernel::AscendAllocatorPlugin::GetInstance().Free(device_data_, device_id_);
      device_data_ = nullptr;
    }
    if (is_acl_host_ && data_ != nullptr) {
      kernel::AscendAllocatorPlugin::GetInstance().FreeHost(const_cast<void *>(data_));
    }
  }
  void SetShape(const std::vector<int64_t> &shape) override { shape_ = shape; }
  void SetDataType(mindspore::DataType data_type) override { type_ = data_type; }
  void SetName(const std::string &name) override { name_ = name; }

  mindspore::Format Format() const override { return format_; }
  void SetFormat(mindspore::Format format) override { format_ = format; }

  const std::string &Name() const override { return name_; }
  enum DataType DataType() const override { return type_; }
  const std::vector<int64_t> &Shape() const override { return shape_; }

  void SetAllocator(const std::shared_ptr<Allocator> &allocator) override { allocator_ = allocator; }
  std::shared_ptr<Allocator> GetAllocator() const override { return allocator_; }

  std::vector<QuantParam> GetQuantParams() const override { return quant_param_; }
  void SetQuantParams(const std::vector<QuantParam> &quant_param) override { quant_param_ = quant_param; }

  size_t DataSize() const override { return ElementNum() * lite::DataTypeSize(static_cast<enum TypeId>(type_)); }

  std::string GetDevice() const override { return device_; }

  int GetDeviceId() const override { return device_id_; }

  void SetDeviceId(int device_id) override { device_id_ = device_id; }

  void SetDevice(const std::string &device) override { device_ = device; }

  void SetDeviceData(void *data) override {
    if (own_data_ && device_data_ != nullptr) {
      MS_LOG(INFO) << "tensor has own device data, now release device data and set new device data.";
      kernel::AscendAllocatorPlugin::GetInstance().Free(device_data_, device_id_);
    }
    device_data_ = data;
    own_data_ = false;
  }

  void *GetDeviceData() override { return device_data_; }
  bool IsConst() const override { return is_const_; }
  void SetIsConst(bool is_const) { is_const_ = is_const; }

  bool IsDevice() const override { return device_data_ != nullptr; }

  std::shared_ptr<const void> Data() const override {
    ResizeData();
    return std::shared_ptr<const void>(data_, [](const void *) {});
  }

  void SetAclHostData(void *data) {
    if (own_data_ && data_ != nullptr && data_ != buffer_.Data()) {
      free(const_cast<void *>(data_));
    }
    if (is_acl_host_ && data_ != nullptr) {
      kernel::AscendAllocatorPlugin::GetInstance().FreeHost(const_cast<void *>(data_));
      is_acl_host_ = false;
    }
    data_ = data;
    is_acl_host_ = true;
    own_data_ = false;
  }

  void SetData(void *data, bool own_data) override {
    if (own_data_ && data_ != nullptr && data_ != buffer_.Data()) {
      free(const_cast<void *>(data_));
    }
    if (is_acl_host_ && data_ != nullptr) {
      kernel::AscendAllocatorPlugin::GetInstance().FreeHost(const_cast<void *>(data_));
      is_acl_host_ = false;
    }
    data_ = data;
    own_data_ = own_data;
  }

  void *MutableData() override {
    ResizeData();
    return const_cast<void *>(data_);
  }

  std::shared_ptr<Impl> Clone() const override {
    auto impl = std::make_shared<TensorDefaultImpl>(name_, type_, shape_, data_, DataSize(), false, false);
    if (!impl) {
      return nullptr;
    }
    impl->SetFormat(format_);
    impl->SetQuantParams(quant_param_);
    impl->SetDeviceData(device_data_);
    impl->SetAllocator(allocator_);
    return impl;
  }

 protected:
  std::string name_;
  enum DataType type_ = DataType::kTypeUnknown;
  enum Format format_ = mindspore::NCHW;
  std::vector<int64_t> shape_;
  std::shared_ptr<Allocator> allocator_ = nullptr;
  std::vector<QuantParam> quant_param_;
  void *device_data_ = nullptr;

  std::string device_ = "";
  int device_id_ = -1;

  mutable Buffer buffer_;
  mutable const void *data_ = nullptr;
  bool own_data_ = false;

  bool is_const_ = false;
  bool is_acl_host_ = false;

  void ResizeData() const {
    if (data_ != nullptr && data_ != buffer_.Data()) {
      return;
    }
    auto data_size = DataSize();
    if (data_size > buffer_.DataSize()) {
      buffer_.ResizeData(data_size);
    }
    if (data_size == 0) {
      data_ = nullptr;
    } else {
      data_ = buffer_.Data();
    }
  }
};
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_EXTENDRT_UTILS_TENSOR_DEFAULT_IMPL_H_

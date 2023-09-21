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
#include <memory>
#include <functional>

#include "include/api/types.h"
#include "ir/tensor.h"
#include "include/backend/device_address.h"
#include "common/utils.h"
#include "common/mutable_tensor_impl.h"
#include "mindspore/core/ir/tensor.h"
#include "kernel/kernel.h"
#include "src/tensor.h"
#include "infer/tensor.h"
#ifdef ENABLE_CLOUD_INFERENCE
#include "src/extendrt/kernel/ascend/plugin/ascend_allocator_plugin.h"
#endif
namespace mindspore {
class TensorRefData : public tensor::TensorData {
 public:
  TensorRefData(void *data, size_t elem_count, size_t data_size, size_t ndim,
                const std::function<void(uint8_t *)> &deleter = nullptr);
  ~TensorRefData();

  ssize_t size() const override;
  ssize_t itemsize() const override;
  ssize_t nbytes() const override;
  ssize_t ndim() const override;
  void *data() override;
  const void *const_data() const override;
  bool is_sub_data() const override { return false; }
  bool has_sub_data() const override { return false; }
  std::string ToString(TypeId type, const ShapeVector &shape, bool use_comma) const override;

 private:
  void *data_ = nullptr;
  size_t elem_count_ = 0;
  size_t data_size_ = 0;
  size_t ndim_ = 0;
  std::function<void(uint8_t *)> deleter_ = nullptr;
};

constexpr auto kLiteDeviceName = "LiteDevice";

class LiteDeviceAddress : public device::DeviceAddress {
 public:
  LiteDeviceAddress(void *ptr, size_t size) : device::DeviceAddress(ptr, size) { device_name_ = kLiteDeviceName; }
  void SetData(void *data) { set_ptr(data); }

  bool SyncDeviceToHost(const ShapeVector &shape, size_t size, TypeId type, void *host_ptr) const override {
    return false;
  }
  bool SyncHostToDevice(const ShapeVector &shape, size_t size, TypeId type, const void *host_ptr,
                        const std::string &format) const override {
    return false;
  }
  bool SyncHostToDevice(const ShapeVector &shape, size_t size, TypeId type, const void *host_ptr) const override {
    return SyncHostToDevice(shape, size, type, host_ptr, "DefaultFormat");
  }
  void ClearDeviceMemory() override {}
};

class TensorTensorImpl : public MutableTensorImpl {
 public:
  explicit TensorTensorImpl(const tensor::Tensor &tensor) : tensor_(std::make_shared<tensor::Tensor>(tensor)) {}
  explicit TensorTensorImpl(const std::shared_ptr<tensor::Tensor> &tensor) : tensor_(tensor) {}

  void SetData(void *, bool) override { MS_LOG_EXCEPTION << "Cannot set data for TensorTensorImpl"; }

  std::shared_ptr<const void> Data() const override {
    MS_EXCEPTION_IF_NULL(tensor_);
    return std::shared_ptr<const void>(tensor_->data_c(), [](const void *) {});
  }

  void SetDeviceId(int device_id) override {
    MS_EXCEPTION_IF_NULL(tensor_);
    device_id_ = device_id;
  }

  void SetDevice(const std::string &device) override {
    MS_EXCEPTION_IF_NULL(tensor_);
    device_ = device;
  }

  int GetDeviceId() const override {
    MS_EXCEPTION_IF_NULL(tensor_);
    return device_id_;
  }

  std::string GetDevice() const override {
    MS_EXCEPTION_IF_NULL(tensor_);
    return device_;
  }

  void *MutableData() override {
    MS_EXCEPTION_IF_NULL(tensor_);
    return tensor_->data_c();
  }

  void SetDeviceData(void *data) override {
    MS_EXCEPTION_IF_NULL(tensor_);
    auto old_device_data = GetDeviceData();
    MS_LOG(ERROR) << "set device data in tensor utils.";
#ifdef ENABLE_CLOUD_INFERENCE
    if (old_device_data != nullptr && device_own_data_) {
      kernel::AscendAllocatorPlugin::GetInstance().Free(old_device_data, GetDeviceId());
    }
#endif
    auto data_size = DataSize();
    auto device_address = std::make_shared<LiteDeviceAddress>(data, data_size);
    tensor_->set_device_address(device_address);
    device_own_data_ = false;
  }
  void *GetDeviceData() override {
    MS_EXCEPTION_IF_NULL(tensor_);
    auto device_address = tensor_->device_address();
    if (device_address == nullptr) {
      return nullptr;
    }
    return device_address->GetMutablePtr();
  }

  bool IsDevice() const override {
    MS_EXCEPTION_IF_NULL(tensor_);
    return tensor_->device_address() != nullptr;
  }

  bool IsConst() const override { return false; }

  void SetShape(const std::vector<int64_t> &) override { MS_LOG_EXCEPTION << "Cannot set shape for TensorTensorImpl"; }
  void SetDataType(mindspore::DataType) override { MS_LOG_EXCEPTION << "Cannot set data type for TensorTensorImpl"; }
  void SetName(const std::string &name) override {
    MS_EXCEPTION_IF_NULL(tensor_);
    tensor_->set_name(name);
  }

  mindspore::Format Format() const override;

  void SetFormat(mindspore::Format format) override;

  const std::string &Name() const override {
    MS_EXCEPTION_IF_NULL(tensor_);
    return tensor_->name();
  }
  enum DataType DataType() const override {
    MS_EXCEPTION_IF_NULL(tensor_);
    return static_cast<enum DataType>(tensor_->data_type());
  }
  const std::vector<int64_t> &Shape() const override {
    MS_EXCEPTION_IF_NULL(tensor_);
    return tensor_->shape();
  }

  void SetAllocator(const std::shared_ptr<Allocator> &allocator) override {
    MS_EXCEPTION_IF_NULL(tensor_);
    tensor_->set_user_data("allocator", allocator);
  }
  std::shared_ptr<Allocator> GetAllocator() const override {
    MS_EXCEPTION_IF_NULL(tensor_);
    return tensor_->user_data<Allocator>("allocator");
  }

  std::vector<QuantParam> GetQuantParams() const override {
    MS_EXCEPTION_IF_NULL(tensor_);
    auto data = tensor_->user_data<std::vector<QuantParam>>("quant_param");
    return data ? *data : std::vector<QuantParam>();
  }

  void SetQuantParams(const std::vector<QuantParam> &quant_param) override {
    MS_EXCEPTION_IF_NULL(tensor_);
    tensor_->set_user_data("quant_param", std::make_shared<std::vector<QuantParam>>(quant_param));
  }

  size_t DataSize() const override {
    auto elem_num = ElementNum();
    if (elem_num <= 0) {
      return 0;
    }
    return LongToSize(elem_num) * lite::DataTypeSize(static_cast<enum TypeId>(DataType()));
  }

  std::shared_ptr<Impl> Clone() const override { return std::make_shared<TensorTensorImpl>(tensor_); }

 private:
  std::shared_ptr<tensor::Tensor> tensor_ = nullptr;
  std::string device_ = "";
  int device_id_ = -1;
  bool device_own_data_ = true;
};

class TensorUtils {
 public:
  // MSTensor <-> TensorPtr
  static std::vector<mindspore::tensor::TensorPtr> MSTensorToTensorPtr(const std::vector<MSTensor> &ms_tensors);
  static std::vector<MSTensor> TensorPtrToMSTensor(std::vector<mindspore::tensor::TensorPtr> tensor_ptrs,
                                                   const std::vector<std::string> &tensor_names);

  static std::vector<mindspore::tensor::Tensor> MSTensorToTensor(const std::vector<MSTensor> &ms_tensors);
  static std::vector<MSTensor> TensorToMSTensor(std::vector<mindspore::tensor::Tensor> tensors,
                                                const std::vector<std::string> &tensor_names);

  // TensorPtr <-> Tensor
  static std::vector<mindspore::tensor::TensorPtr> TensorToTensorPtr(
    const std::vector<mindspore::tensor::Tensor> &tensors);
  static std::vector<mindspore::tensor::Tensor> TensorPtrToTensor(
    const std::vector<mindspore::tensor::TensorPtr> &tensor_ptrs);
};

class CloudTensorUtils {
 public:
  /* lite tensor ---> Address */
  static kernel::AddressPtr LiteTensorToAddressPtr(const lite::Tensor *lite_tensor);
  static std::vector<mindspore::kernel::AddressPtr> LiteTensorToAddressPtrVec(
    const std::vector<lite::Tensor *> &lite_tensors);

  /* lite tensor ---> kernel tensor */
  static kernel::KernelTensorPtr LiteTensorToKernelTensorPtr(const lite::Tensor *lite_tensor);
  static std::vector<kernel::KernelTensorPtr> LiteTensorToKernelTensorPtrVec(
    const std::vector<lite::Tensor *> &lite_tensors);
};

class AbstractTensorUtils {
 public:
  static std::vector<std::vector<int64_t>> GetTensorListShapes(const std::vector<infer::abstract::Tensor *> &tensors);
  static bool SetTensorListShapse(const std::vector<infer::abstract::Tensor *> &tensors,
                                  const std::vector<std::vector<int64_t>> &shapes);
};
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_EXTENDRT_UTILS_TENSOR_UTILS_H_

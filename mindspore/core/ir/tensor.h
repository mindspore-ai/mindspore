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

#ifndef MINDSPORE_CORE_IR_TENSOR_H_
#define MINDSPORE_CORE_IR_TENSOR_H_

#include <memory>
#include <string>
#include <vector>
#include <numeric>
#include <mutex>
#include <condition_variable>

#include "ir/device_sync.h"
#include "ir/meta_tensor.h"
#include "utils/log_adapter.h"
#include "base/float16.h"
#include "utils/shape_utils.h"
#include "utils/ms_exception.h"
#include "ir/device_event.h"

// brief mindspore namespace.
//
// mindspore namespace is the top level namespace of MindSpore project.
// Other namespace should be a sub namespace of mindspore namespace in the ME project.
namespace mindspore {
// brief mindspore::tensor namespace
enum TensorSyncStatus { kNoNeedSync, kNeedSyncHostToDevice, kNeedSyncDeviceToHost, kNeedSyncDeviceToHostImmediately };
// A sub namespace in ME to support tensor related definition.
namespace tensor {
// Tensor data interface.
class TensorData {
 public:
  /// virtual destructor is required for base classes.
  virtual ~TensorData() = default;
  /// Total number of elements.
  virtual ssize_t size() const = 0;
  /// Byte size of a single element.
  virtual ssize_t itemsize() const = 0;
  /// Total number of bytes.
  virtual ssize_t nbytes() const = 0;
  /// Number of dimensions.
  virtual ssize_t ndim() const = 0;
  /// Data pointer.
  virtual void *data() = 0;
  /// Const Data pointer.
  virtual const void *const_data() const = 0;
  /// Is data equals.
  virtual bool equals(const TensorData &other) const {
    if (this == &other) {
      return true;
    }
    // By default, compare data byte by byte.
    auto this_data = static_cast<const uint8_t *>(const_data());
    auto other_data = static_cast<const uint8_t *>(other.const_data());
    if (this_data == nullptr || other_data == nullptr) {
      // null means data not initialized, compare uninitialized data always return false.
      return false;
    }
    return (this_data == other_data) || (ndim() == other.ndim() && nbytes() == other.nbytes() &&
                                         std::equal(this_data, this_data + nbytes(), other_data));
  }
  /// To string.
  virtual std::string ToString(const TypeId type, const ShapeVector &shape, bool use_comma) const = 0;
};

using TensorDataPtr = std::shared_ptr<TensorData>;

class WaitEvent : public ExceptionListener {
 public:
  void OnException() override { set_need_wait(false); }

  void Wait() const {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!need_wait_) {
      return;
    }
    MsException::Instance().SetExceptionListener(const_cast<WaitEvent *>(this));
    cond_var_.wait(lock, [this] { return !need_wait_; });
    MsException::Instance().SetExceptionListener(nullptr);
    MsException::Instance().CheckException();
  }

  void set_need_wait(bool need_wait) {
    std::unique_lock<std::mutex> lock(mutex_);
    need_wait_ = need_wait;
    if (!need_wait_) {
      cond_var_.notify_all();
    }
  }

  bool need_wait() const { return need_wait_; }

 private:
  bool need_wait_{false};
  mutable std::mutex mutex_;
  mutable std::condition_variable cond_var_;
};

// Tensor entity class
class Tensor : public MetaTensor {
 public:
  abstract::AbstractBasePtr ToAbstract() override;

  // brief Create tensor from another tensor, data is shared.
  //
  // param tensor [Tensor] The input tensor.
  explicit Tensor(const Tensor &tensor);

  // brief Create tensor with given data type from another tensor.
  //
  // param tensor [Tensor] The input tensor.
  // param data_type [TypeId] The new tensor data type.
  Tensor(const Tensor &tensor, TypeId data_type);

  // brief Create tensor with the given shared tensor data.
  //
  // param data_type [TypeId] Data type of the tensor.
  // param shape The shape represented by ShapeVector of the tensor.
  // param data The shared tensor data.
  Tensor(TypeId data_type, const ShapeVector &shape, TensorDataPtr data);

  // brief Create a lazy allocated tensor.
  //
  // param data_type [TypeId] Data type of the tensor.
  // param shape The shape represented by ShapeVector of the tensor.
  Tensor(TypeId data_type, const ShapeVector &shape);

  // brief Create a tensor with input data buffer.
  //
  // param data_type [TypeId] Data type of the tensor.
  // param shape The shape represented by ShapeVector of the tensor.
  // param data The input data to be copied into tensor.
  // param data_len The length of data in bytes.
  Tensor(TypeId data_type, const ShapeVector &shape, void *data, size_t data_len);

  // brief Create a tensor with input data buffer and given source data type.
  //
  // param data_type [TypeId] Data type of the tensor.
  // param shape The shape represented by ShapeVector of the tensor.
  // param data The input data to be copied into tensor.
  // param src_data_type The source data type.
  Tensor(TypeId data_type, const ShapeVector &shape, void *data, TypeId src_data_type);

  // brief Create 1 dimension tensor from an int vector.
  //
  // param input [std::vector<int64_t>] the data for tensor
  // param data_type [TypeId] data type
  explicit Tensor(const std::vector<int64_t> &input, const TypePtr &data_type = nullptr);

  // brief Create 1 dimension tensor from a float vector.
  //
  // param input [std::vector<double>] the data for tensor
  // param data_type [TypeId] data type
  explicit Tensor(const std::vector<double> &input, const TypePtr &data_type = nullptr);

  // brief Create 0 dimension tensor from an int64_t scalar.
  //
  // param input [int64] the data for tensor
  // param data_type [TypeId] data type
  explicit Tensor(int64_t input, const TypePtr &data_type = nullptr);

  // brief Create 0 dimension tensor from a float scalar.
  //
  // param input [double] the data for tensor
  // param data_type [TypeId] data type
  explicit Tensor(double input, const TypePtr &data_type = nullptr);

  // brief Create 0 dimension tensor from a uint scalar.
  //
  // param input [uint] the data for tensor
  // param data_type [TypeId] data type
  explicit Tensor(uint64_t input, const TypePtr &data_type = nullptr);

  // brief Create 0 dimension tensor from a bool scalar.
  //
  // param input [bool] the data for tensor
  // param data_type [TypeId] data type
  explicit Tensor(bool input, const TypePtr &data_type = nullptr);

  ~Tensor() override = default;

  MS_DECLARE_PARENT(Tensor, MetaTensor);

  // brief Compares two Tensor objects.
  //
  // Compare two tensor objects to see if they have same data type, shape and data address.
  //
  // param tensor The Tensor object to be compared.
  // return true: If having same type, shape and data address, return true, or return false.
  bool operator==(const Tensor &tensor) const;

  // It is different from 'operator==' which just compare shape/type/address,
  // it do real value comparison.
  bool ValueEqual(const Tensor &tensor) const;

  // assign value to this tensor
  Tensor &AssignValue(const Tensor &tensor);

  bool operator==(const Value &other) const override {
    if (other.isa<Tensor>()) {
      auto &other_ = static_cast<const Tensor &>(other);
      return *this == other_;
    }
    return false;
  }

  // brief Gets tensor's dimension
  //
  // return The number of dimensions of the tensor data.
  int DataDim() const { return static_cast<int>(data().ndim()); }

  // brief Getting tensor data size
  //
  // return The total number of elements of the tensor data.
  int DataSize() const { return static_cast<int>(data().size()); }

  // brief Get the data type fo the tensor for C++
  //
  // return [int] The tensor's data type will be cast to int to return.
  int data_type_c() const { return static_cast<int>(data_type_); }

  // brief Get the tensor's shape for C++
  //
  // return [ShapeVector]
  ShapeVector shape_c(void) const { return shape(); }

  // brief Get Tensor data pointer for c++ type
  //
  // return The pointer to the object
  void *data_c() { return data().data(); }

  // brief Get Tensor data byte-size for c++ type
  //
  // return byte size of Tensor data
  size_t Size() const { return static_cast<size_t>(data().nbytes()); }

  void *data_c() const { return data_->data(); }

  // brief Sync data with device, need wait data valid.
  void data_sync(bool need_wait = true) const;

  // brief Get the internal data object.
  //
  // return The reference to internal data object.
  TensorData &data() { return *data_; }

  // brief Get the internal data shared pointer.
  //
  // return The reference to internal data object.
  const TensorDataPtr &data_ptr() const { return data_; }

  // brief Get the internal data object.
  //
  // return The reference to internal data object.
  const TensorData &data() const { return *data_; }

  TypeId set_data_type(const TypeId data_type) override;

  std::string GetShapeAndDataTypeInfo() const;

  std::string ToStringInternal(int limit_size) const;

  std::string ToStringNoLimit() const;

  std::string ToString() const override;

  std::string ToStringRepr() const;

  void CheckShape(const ShapeVector &shape) const;

  bool is_init() const { return init_flag_; }
  void set_init_flag(bool flag) { init_flag_ = flag; }

  DeviceSyncPtr device_address() const { return device_sync_; }
  void set_device_address(const DeviceSyncPtr &device_sync) { device_sync_ = device_sync; }
  void set_padding_type(const std::string padding_type) { padding_type_ = padding_type; }
  std::string padding_type() const { return padding_type_; }

  std::string id() const { return id_; }
  TypePtr cast_dtype() { return cast_dtype_; }
  void set_cast_dtype(TypePtr dtype = nullptr) { cast_dtype_ = dtype; }

  // used if cache_enable, in order to update tensor from cache to host
  bool cache_enable() const { return cache_enable_; }
  void set_cache_enable(bool cache_enable = true) { cache_enable_ = cache_enable; }
  std::shared_ptr<Tensor> hashmap_tensor_ptr() const { return hashmap_tensor_ptr_; }
  void set_hashmap_tensor_ptr(std::shared_ptr<Tensor> hashmap_tensor_ptr = nullptr) {
    hashmap_tensor_ptr_ = hashmap_tensor_ptr;
  }
  std::shared_ptr<Tensor> cache_tensor_ptr() const { return cache_tensor_ptr_; }
  void set_cache_tensor_ptr(std::shared_ptr<Tensor> cache_tensor_ptr = nullptr) {
    cache_tensor_ptr_ = cache_tensor_ptr;
  }

  void SetNeedWait(bool need_wait) {
    if (event_ != nullptr) {
      event_->set_need_wait(need_wait);
    } else if (need_wait) {
      event_ = std::make_shared<WaitEvent>();
      event_->set_need_wait(need_wait);
    }
  }

  bool NeedWait() const {
    if (event_ != nullptr) {
      return event_->need_wait();
    }
    return false;
  }

  void Wait() const {
    if (event_ != nullptr) {
      event_->Wait();
    }
    event_ = nullptr;
  }

  void SetDeviceEvent(const std::shared_ptr<DeviceEvent> &device_event) { device_event_ = device_event; }

  void WaitDevice() {
    if (device_event_ != nullptr) {
      device_event_->WaitEvent();
    }
  }

  bool NeedWaitDevice() const {
    if (device_event_ != nullptr) {
      return device_event_->NeedWait();
    }
    return false;
  }

  void set_sync_status(TensorSyncStatus sync_status) { sync_status_ = sync_status; }

  TensorSyncStatus sync_status() const { return sync_status_; }

  bool NeedSyncDeviceToHostImmediately() const { return sync_status_ == kNeedSyncDeviceToHostImmediately; }

  bool NeedSyncDeviceToHost() const { return sync_status_ == kNeedSyncDeviceToHost; }

  bool NeedSyncHostToDevice() const { return sync_status_ == kNeedSyncHostToDevice; }

  bool IsGraphOutput() { return graph_output_; }
  void SetIsGraphOutput() { graph_output_ = true; }

 private:
  bool init_flag_{false};
  TensorDataPtr data_{nullptr};
  std::string id_{""};
  mutable std::shared_ptr<WaitEvent> event_{nullptr};
  mutable TensorSyncStatus sync_status_{kNeedSyncHostToDevice};
  bool graph_output_{false};
  DeviceSyncPtr device_sync_{nullptr};
  bool cache_enable_{false};
  std::shared_ptr<Tensor> cache_tensor_ptr_{nullptr};
  std::shared_ptr<Tensor> hashmap_tensor_ptr_{nullptr};
  std::string padding_type_{""};
  TypePtr cast_dtype_{nullptr};
  std::shared_ptr<DeviceEvent> device_event_{nullptr};
};
using TensorPtr = std::shared_ptr<Tensor>;
using TensorPtrList = std::vector<std::shared_ptr<Tensor>>;
}  // namespace tensor
}  // namespace mindspore

#endif  // MINDSPORE_CORE_IR_TENSOR_H_

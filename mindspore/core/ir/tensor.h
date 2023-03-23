/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include <utility>

#include "ir/device_sync.h"
#include "ir/meta_tensor.h"
#include "utils/log_adapter.h"
#include "base/float16.h"
#include "base/user_data.h"
#include "utils/shape_utils.h"
#include "utils/ms_exception.h"
#include "ir/device_event.h"
#include "utils/os.h"
#include "ir/quantization_param.h"

// brief mindspore namespace.
//
// mindspore namespace is the top level namespace of MindSpore project.
// Other namespace should be a sub namespace of mindspore namespace in the ME project.
namespace mindspore {
// brief mindspore::tensor namespace
enum TensorSyncStatus {
  kNoNeedSync,
  kNeedSyncHostToDevice,
  kNeedSyncHostToDeviceImmediately,
  kNeedSyncDeviceToHost,
  kNeedSyncDeviceToHostImmediately
};

enum TensorCompressionType {
  kNoCompression = 0,
  kIndexing = 1,
  kSparse = 2,
  kFSE = 3,
  kBitPacking = 4,
  kFSEInt = 5,
  kFSEInfer = 6
};

// A sub namespace in ME to support tensor related definition.
namespace tensor {
// Tensor data interface.
class MS_CORE_API TensorData {
 public:
  /// \brief Virtual destructor is required for base classes.
  virtual ~TensorData() = default;

  /// \brief Get total number of elements.
  ///
  /// \return Total number of elements.
  virtual ssize_t size() const = 0;

  /// \brief Get byte size of a single element.
  ///
  /// \return Byte size of a single element.
  virtual ssize_t itemsize() const = 0;

  /// \brief Get total number of bytes.
  ///
  /// \return Total number of bytes.
  virtual ssize_t nbytes() const = 0;

  /// \brief Get number of dimensions.
  ///
  /// \return Number of dimensions.
  virtual ssize_t ndim() const = 0;

  /// \brief Get data pointer.
  ///
  /// \return Data pointer.
  virtual void *data() = 0;

  /// \brief Get const data pointer.
  ///
  /// \return Const data pointer.
  virtual const void *const_data() const = 0;

  /// \brief Get whether this tensor data is sub data.
  ///
  /// \return Whether this tensor data is sub data.
  virtual bool is_sub_data() const = 0;

  /// \brief Check whether this tensor data has sub data.
  ///
  /// \return True if this tensor data has sub data, otherwise false.
  virtual bool has_sub_data() const = 0;

  /// \brief Get whether this tensor data is from numpy.
  ///
  /// \return Whether this tensor data is from numpy.
  virtual bool is_from_numpy() const { return false; }

  /// \brief Get whether this tensor data have use persistent storage to save data.
  ///
  /// \return Whether this tensor data have use persistent storage to save data.
  virtual bool is_persistent_data() const { return false; }

  /// \brief Whether the data are equal.
  ///
  /// \param[in] other Another TensorData.
  /// \return Ture if the two data are equal, otherwise false.
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

  /// \brief Get display information about this TensorData.
  ///
  /// \param[in] type The type of tensor data.
  /// \param[in] shape The shape of tensor data.
  /// \param[in] use_comma Whether to use comma.
  /// \return The display information.
  virtual std::string ToString(TypeId type, const ShapeVector &shape, bool use_comma) const = 0;
};

using TensorDataPtr = std::shared_ptr<TensorData>;

class WaitEvent : public ExceptionListener {
 public:
  ~WaitEvent() = default;

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

class Tensor;
using TensorPtr = std::shared_ptr<Tensor>;
using TensorPtrList = std::vector<std::shared_ptr<Tensor>>;

// Tensor entity class
class MS_CORE_API Tensor : public MetaTensor {
 public:
  abstract::AbstractBasePtr ToAbstract() override;
  Tensor() = default;

  /// \brief Create tensor from another tensor, data is shared.
  ///
  /// \param[in] tensor [Tensor] The input tensor.
  explicit Tensor(const Tensor &tensor);
  /// \brief Create tensor with given data type from another tensor.
  ///
  /// \param[in] tensor [Tensor] The input tensor.
  /// \param[in] data_type [TypeId] The new tensor data type.
  Tensor(const Tensor &tensor, TypeId data_type);

  /// \brief Create tensor with the given shared tensor data.
  ///
  /// \param[in] data_type [TypeId] Data type of the tensor.
  /// \param[in] shape The shape represented by ShapeVector of the tensor.
  /// \param[in] data The shared tensor data.
  Tensor(TypeId data_type, const ShapeVector &shape, TensorDataPtr data);

  /// \brief Create a lazy allocated tensor.
  ///
  /// \param[in] data_type [TypeId] Data type of the tensor.
  /// \param[in] shape The shape represented by ShapeVector of the tensor.
  Tensor(TypeId data_type, const ShapeVector &shape);

  /// \brief Create a tensor with input data buffer.
  ///
  /// \param[in] data_type [TypeId] Data type of the tensor.
  /// \param[in] shape The shape represented by ShapeVector of the tensor.
  /// \param[in] data The input data to be copied into tensor.
  /// \param[in] data_len The length of data in bytes.
  Tensor(TypeId data_type, const ShapeVector &shape, void *data, size_t data_len);

  /// \brief Create a tensor with input data buffer and given source data type.
  ///
  /// \param[in] data_type [TypeId] Data type of the tensor.
  /// \param[in] shape The shape represented by ShapeVector of the tensor.
  /// \param[in] data The input data to be copied into tensor.
  /// \param[in] src_data_type The source data type.
  Tensor(TypeId data_type, const ShapeVector &shape, void *data, TypeId src_data_type);

  /// \brief Create 1 dimension tensor from an int vector.
  ///
  /// \param[in] input [std::vector<int64_t>] the data for tensor.
  /// \param[in] data_type [TypeId] data type.
  explicit Tensor(const std::vector<int64_t> &input, const TypePtr &data_type = nullptr);

  /// \brief Create 1 dimension tensor from an int vector.
  ///
  /// \param[in] input [std::vector<int32_t>] the data for tensor.
  /// \param[in] data_type [TypeId] data type.
  explicit Tensor(const std::vector<int32_t> &input, const TypePtr &data_type = nullptr);

  /// \brief Create 1 dimension tensor from a float vector.
  ///
  /// \param[in] input [std::vector<double>] the data for tensor.
  /// \param[in] data_type [TypeId] data type.
  explicit Tensor(const std::vector<double> &input, const TypePtr &data_type = nullptr);

  /// \brief Create 0 dimension tensor from an int64_t scalar.
  ///
  /// \param[in] input [int64] the data for tensor.
  /// \param[in] data_type [TypeId] data type.
  explicit Tensor(int64_t input, const TypePtr &data_type = nullptr);

  /// \brief Create 0 dimension tensor from an int32_t scalar.
  ///
  /// \param[in] input [int32] the data for tensor.
  /// \param[in] data_type [TypeId] data type.
  explicit Tensor(int32_t input, const TypePtr &data_type = nullptr);

  /// \brief Create 0 dimension tensor from an int16_t scalar.
  ///
  /// \param[in] input [int16] the data for tensor.
  /// \param[in] data_type [TypeId] data type.
  explicit Tensor(int16_t input, const TypePtr &data_type = nullptr);

  /// \brief Create 0 dimension tensor from an int8_t scalar.
  ///
  /// \param[in] input [int8] the data for tensor.
  /// \param[in] data_type [TypeId] data type.
  explicit Tensor(int8_t input, const TypePtr &data_type = nullptr);

  /// \brief Create 0 dimension tensor from a double scalar.
  ///
  /// \param[in] input [double] the data for tensor.
  /// \param[in] data_type [TypeId] data type.
  explicit Tensor(double input, const TypePtr &data_type = nullptr);

  /// \brief Create 0 dimension tensor from a float scalar.
  ///
  /// \param[in] input [float] the data for tensor.
  /// \param[in] data_type [TypeId] data type.
  explicit Tensor(float input, const TypePtr &data_type = nullptr);

  /// \brief Create 0 dimension tensor from a float16 scalar.
  ///
  /// \param[in] input [float16] the data for tensor.
  /// \param[in] data_type [TypeId] data type.
  explicit Tensor(float16 input, const TypePtr &data_type = nullptr);

  /// \brief Create 0 dimension tensor from a uint64 scalar.
  ///
  /// \param[in] input [uint64] the data for tensor.
  /// \param[in] data_type [TypeId] data type.
  explicit Tensor(uint64_t input, const TypePtr &data_type = nullptr);

  /// \brief Create 0 dimension tensor from a uint32 scalar.
  ///
  /// \param[in] input [uint32] the data for tensor.
  /// \param[in] data_type [TypeId] data type.
  explicit Tensor(uint32_t input, const TypePtr &data_type = nullptr);

  /// \brief Create 0 dimension tensor from a uint16 scalar.
  ///
  /// \param[in] input [uint16] the data for tensor.
  /// \param[in] data_type [TypeId] data type.
  explicit Tensor(uint16_t input, const TypePtr &data_type = nullptr);

  /// \brief Create 0 dimension tensor from a uint8 scalar.
  ///
  /// \param[in] input [uint8] the data for tensor.
  /// \param[in] data_type [TypeId] data type.
  explicit Tensor(uint8_t input, const TypePtr &data_type = nullptr);

  /// \brief Create 0 dimension tensor from a bool scalar.
  ///
  /// \param[in] input [bool] the data for tensor.
  /// \param[in] data_type [TypeId] data type.
  explicit Tensor(bool input, const TypePtr &data_type = nullptr);

  /// \brief Create a chunk tensor with the given data size.
  ///
  /// \param[in] data_type [TypeId] Data type of the tensor.
  /// \param[in] data_size The tensor chunk data size in number of elements.
  Tensor(TypeId data_type, size_t data_size);

  /// \brief Create a Tensor which shape and size may be inconsistent, such as Tensor with compression data.
  ///
  /// \param[in] origin_data_type [TypeId] Data type of the origin tensor.
  /// \param[in] shape The shape represented by ShapeVector of the tensor.
  /// \param[in] compression_data_size The compression data buffer size.
  /// \param[in] TensorCompressionType The tensor compression type.
  Tensor(TypeId origin_data_type, const ShapeVector &shape, size_t compression_data_size,
         TensorCompressionType compression_type);

  Tensor &operator=(const Tensor &tensor) {
    if (this == &tensor) {
      return *this;
    }
    init_flag_ = tensor.init_flag_;
    is_forward_output_ = tensor.is_forward_output_;
    data_ = tensor.data_;
    id_ = tensor.id_;
    event_ = tensor.event_;
    need_wait_ = tensor.need_wait_;
    sync_status_ = tensor.sync_status_;
    device_sync_ = tensor.device_sync_;
    need_release_device_mem_ = tensor.need_release_device_mem_;
    cache_enable_ = tensor.cache_enable_;
    base_shape_ptr_ = tensor.base_shape_ptr_;
    cache_tensor_ptr_ = tensor.cache_tensor_ptr_;
    hashmap_tensor_ptr_ = tensor.hashmap_tensor_ptr_;
    padding_type_ = tensor.padding_type();
    device_event_ = tensor.device_event_;
    lazy_callback_ = tensor.lazy_callback_;
    user_data_ = tensor.user_data_;
    compression_type_ = tensor.compression_type_;
    tensor_name_ = tensor.tensor_name_;
    return *this;
  }

  /// Destructor of Tensor.
  ~Tensor() override = default;

  MS_DECLARE_PARENT(Tensor, MetaTensor);

  /// \brief Compare two tensor objects to see if they have same data type, shape and data address.
  ///
  /// \param[in] tensor The Tensor object to be compared.
  /// \return True if having same type, shape and data address, otherwise false.
  bool operator==(const Tensor &tensor) const;

  /// \brief It is different from 'operator==' which just compares shape/type/address,
  /// it does real value comparison.
  ///
  /// \param[in] tensor The Tensor object to be compared.
  /// \return True if it has the same value, otherwise false.
  bool ValueEqual(const Tensor &tensor) const;

  /// \brief Assign value to this tensor.
  ///
  /// \param[in] tensor The input tensor.
  /// \return Tensor with new value.
  Tensor &AssignValue(const Tensor &tensor);

  bool operator==(const Value &other) const override {
    if (other.isa<Tensor>()) {
      auto &other_ = static_cast<const Tensor &>(other);
      return *this == other_;
    }
    return false;
  }

  /// \brief Gets tensor's dimension.
  ///
  /// \return The number of dimensions of the tensor data.
  int DataDim() const { return static_cast<int>(data().ndim()); }

  /// \brief Getting tensor data size.
  ///
  /// \return The total number of elements of the tensor data.
  size_t DataSize() const { return data().size(); }

  /// \brief Get the data type of the tensor for C++
  ///
  /// \return [int] The tensor's data type will be cast to int to return.
  int data_type_c() const { return static_cast<int>(data_type_); }

  /// \brief Get the tensor's shape for C++
  ///
  /// \return [ShapeVector]
  ShapeVector shape_c(void) const { return shape(); }

  /// \brief Get Tensor data pointer for c++ type
  ///
  /// \return The pointer to the object
  void *data_c() { return data().data(); }

  /// \brief Get Tensor data byte-size for c++ type
  ///
  /// \return byte size of Tensor data
  size_t Size() const { return static_cast<size_t>(data().nbytes()); }

  /// \brief The pointer to the object
  void *data_c() const { return data_->data(); }

  /// \brief To synchronize data with the device, you need to wait for the data to be valid.
  ///
  void data_sync(bool need_wait = true) const;

  /// \brief To synchronize data with the device without keeping device address, you need to wait for the data to be
  /// valid.
  ///
  void data_sync_directly(const DeviceSync *const device_sync, bool need_wait = true) const;

  /// \brief Get the internal data object.
  ///
  /// \return The reference to internal data object.
  TensorData &data() {
    MS_EXCEPTION_IF_NULL(data_);
    return *data_;
  }

  /// \brief Get the internal data shared pointer.
  ///
  /// return The reference to internal data object.
  const TensorDataPtr &data_ptr() const { return data_; }

  /// \brief Get the internal data object.
  ///
  /// \return The reference to internal data object.
  const TensorData &data() const { return *data_; }

  TypeId set_data_type(TypeId data_type) override;

  size_t set_shape(const ShapeVector &shape) override;

  /// \brief Get information about shape and data type.
  ///
  /// \return Information about shape and data type.
  std::string GetShapeAndDataTypeInfo() const;

  /// \brief Get display information of limit size.
  ///
  /// \param[in] limit_size The limit size.
  /// \return The display information of limit size.
  std::string ToStringInternal(size_t limit_size) const;

  /// \brief Get display information with unlimited size.
  ///
  /// \return The display information with unlimited size.
  std::string ToStringNoLimit() const;

  /// \brief Get display information of this Tensor.
  ///
  /// \return The display information of this Tensor.
  std::string ToString() const override;

  /// \brief Get display information in repr form.
  ///
  /// \return The display information in repr form.
  std::string ToStringRepr() const;

  /// \brief Check the shape of this Tensor.
  ///
  /// \param[in] shape The input shape.
  void CheckShape(const ShapeVector &shape) const;

  /// \brief Check if this Tensor is initialized.
  ///
  /// \return Whether this Tensor is initialized.
  bool is_init() const { return init_flag_; }

  /// \brief Set the initialization flag of this Tensor.
  ///
  /// \param[in] flag Whether this Tensor is initialized.
  void set_init_flag(bool flag) { init_flag_ = flag; }

  /// \brief Check whether this Tensor needs to be converted.
  ///
  /// \return Whether this Tensor needs to be converted.
  bool is_adapter() const { return adapter_flag_; }

  /// \brief Set the adapter flag of this Tensor.
  ///
  /// \param[in] flag Whether this Tensor needs to be converted.
  void set_adapter_flag(bool flag) { adapter_flag_ = flag; }

  /// \brief Check if this Tensor is forward output.
  ///
  /// \return Whether this Tensor is forward output.
  bool is_forward_output() const { return is_forward_output_; }

  /// \brief Set the forward output flag of this Tensor.
  ///
  /// \param[in] is_forward_output Whether this Tensor is forward output.
  void set_is_forward_output(bool is_forward_output) { is_forward_output_ = is_forward_output; }

  /// \brief Get the device address.
  ///
  /// \return The device address.
  DeviceSyncPtr device_address() const { return device_sync_; }

  /// \brief Set the device address.
  ///
  /// \param[in] device_sync The input Device synchronization.
  /// \param[in] need_update_ref_count If need_update_ref_count is true, the device address cannot be released and
  /// reused, so the feature map should set false when set device address of tensor.
  void set_device_address(const DeviceSyncPtr &device_sync, bool need_update_ref_count = true) {
    device_sync_ = device_sync;
    // To support the old and new runtime coexistence, the output of old runtime may be the input of new runtime, so the
    // device address cannot be released through ref count and set max ref count in this scenario.
    if (need_update_ref_count && (device_sync_ != nullptr)) {
      device_sync_->set_original_ref_count(SIZE_MAX);
      device_sync_->ResetRefCount();
    }
  }

  /// \brief Check whether to release device memory.
  ///
  /// \return Ture if need to release device memory, otherwise false.
  bool need_release_device_mem() const { return need_release_device_mem_; }

  /// \brief Set the flag to determine whether the device memory needs to be released.
  ///
  /// \param[in] release_device_mem If release_device_mem is ture, the device memory will to be released.
  void set_need_release_device_mem(bool release_device_mem) { need_release_device_mem_ = release_device_mem; }

  /// \brief Set the padding type of this Tensor.
  ///
  /// \param[in] padding_type The input padding type.
  void set_padding_type(const std::string padding_type) { padding_type_ = padding_type; }

  /// \brief Get the padding type of this Tensor.
  ///
  /// \return The padding type.
  std::string padding_type() const { return padding_type_; }

  /// \brief Get the id of this Tensor.
  ///
  /// \return The id of this Tensor.
  std::string id() const { return id_; }

  /// \brief Set the id of this Tensor.
  ///
  /// \param[in] id The tensor id.
  void set_id(const std::string &id) { id_ = id; }

  /// \brief Get the cast dtype of this Tensor.
  ///
  /// \return The cast dtype of this Tensor.
  TypePtr cast_dtype() { return cast_dtype_; }

  /// \brief Set the cast dtype of this Tensor.
  ///
  /// \param[in] dtype The input cast dtype.
  void set_cast_dtype(const TypePtr &dtype = nullptr) { cast_dtype_ = dtype; }

  /// \brief Used cache_enable to update the tensor from the cache to the host.
  ///
  /// \return True if caching is enabled, otherwise false.
  bool cache_enable() const { return cache_enable_; }

  /// \brief Set cache_enable.
  ///
  /// \param[in] cache_enable Whether to enable caching.
  void set_cache_enable(bool cache_enable = true) { cache_enable_ = cache_enable; }

  /// \brief Get the pointer of hashmap tensor.
  ///
  /// \return The pointer of hashmap tensor.
  std::shared_ptr<Tensor> hashmap_tensor_ptr() const { return hashmap_tensor_ptr_; }

  /// \brief Set the pointer of hashmap tensor.
  ///
  /// \param[in] hashmap_tensor_ptr The input pointer of hashmap tensor.
  void set_hashmap_tensor_ptr(const std::shared_ptr<Tensor> &hashmap_tensor_ptr = nullptr) {
    hashmap_tensor_ptr_ = hashmap_tensor_ptr;
  }

  /// \brief Get the pointer of cache tensor.
  ///
  /// \return The pointer of cache tensor.
  std::shared_ptr<Tensor> cache_tensor_ptr() const { return cache_tensor_ptr_; }

  /// \brief Set the pointer of cache tensor.
  ///
  /// \param[in] cache_tensor_ptr The input pointer of cache tensor.
  void set_cache_tensor_ptr(const std::shared_ptr<Tensor> &cache_tensor_ptr = nullptr) {
    cache_tensor_ptr_ = cache_tensor_ptr;
  }

  /// \brief Get tensor's BaseShape.
  ///
  /// \return The BaseShape of this tensor.
  const BaseShapePtr &base_shape_ptr() const { return base_shape_ptr_; }

  /// \brief Set tensor's BaseShape.
  ///
  /// \param[in] BaseShapePtr The tensor's BaseShape.
  void set_base_shape(const BaseShapePtr &base_shape) { base_shape_ptr_ = base_shape; }

  /// \brief Set whether the event needs to wait.
  ///
  /// \param[in] need_wait Whether the event needs to wait.
  void SetNeedWait(bool need_wait) {
    need_wait_ = need_wait;
    auto event = event_;
    if (event != nullptr) {
      event->set_need_wait(need_wait);
    } else if (need_wait) {
      event_ = std::make_shared<WaitEvent>();
      event_->set_need_wait(need_wait);
    }
  }

  /// \brief Check whether the event needs to wait.
  ///
  /// \return Whether the event needs to wait.
  bool NeedWait() const { return need_wait_; }

  /// \brief Require the event to wait.
  void Wait() const {
    auto event = event_;
    if (event != nullptr) {
      event->Wait();
    }
    event_ = nullptr;
  }

  /// \brief Set device event.
  ///
  /// \param[in] device_event The input device event.
  void SetDeviceEvent(const std::shared_ptr<DeviceEvent> &device_event) { device_event_ = device_event; }

  /// \brief Require the device event to wait.
  void WaitDevice() {
    if (device_event_ != nullptr) {
      device_event_->WaitEvent();
    }
  }

  /// \brief Set whether the device needs to wait.
  ///
  /// \return Whether the device needs to wait.
  bool NeedWaitDevice() const {
    if (device_event_ != nullptr) {
      return device_event_->NeedWait();
    }
    return false;
  }

  /// \brief Set synchronization status.
  ///
  /// \param[in] sync_status The input synchronization status.
  void set_sync_status(TensorSyncStatus sync_status) const { sync_status_ = sync_status; }

  /// \brief Get synchronization status.
  ///
  /// \return The synchronization status.
  TensorSyncStatus sync_status() const { return sync_status_; }

  /// \brief Check the value of sync_status_.
  ///
  /// \return Ture if sync_status_ is kNeedSyncDeviceToHostImmediately.
  bool NeedSyncDeviceToHostImmediately() const { return sync_status_ == kNeedSyncDeviceToHostImmediately; }

  /// \brief Check the value of sync_status_.
  ///
  /// \return Ture if sync_status_ is kNeedSyncDeviceToHost.
  bool NeedSyncDeviceToHost() const { return sync_status_ == kNeedSyncDeviceToHost; }

  /// \brief Check the value of sync_status_.
  ///
  /// \return Ture if sync_status_ is kNeedSyncHostToDevice.
  bool NeedSyncHostToDevice() const { return sync_status_ == kNeedSyncHostToDevice; }

  /// \brief Check the value of sync_status_.
  ///
  /// \return Ture if sync_status_ is kNeedSyncHostToDeviceImmediately.
  bool NeedSyncHostToDeviceImmediately() const { return sync_status_ == kNeedSyncHostToDeviceImmediately; }

  /// \brief Check if this Tensor is the output of graph.
  ///
  /// \return Whether this Tensor is the output of graph
  bool IsGraphOutput() const { return graph_output_; }

  /// \brief Set whether this Tensor is the output of graph.
  void SetIsGraphOutput() { graph_output_ = true; }

  /// \brief Get whether this Tensor is updated by the device.
  ///
  /// \return Whether this Tensor is updated by the device.
  bool IsUpdatedByDevice() const { return updated_by_device_; }

  /// \brief Set whether this Tensor is updated by the device.
  void SetIsUpdateByDevice() { updated_by_device_ = true; }

  /// \brief Set lazy callback function to this Tensor
  ///
  /// \param[in] lazy_callback The callback from backend when lazy build is enabled
  void set_lazy_callback(const std::function<void(void)> &lazy_callback) { lazy_callback_ = lazy_callback; }

  /// \brief Get the memory chunk pointer and offset if memory chunk for this tensor exists.
  ///
  /// \return The memory chunk pointer and offset, nullptr and 0 if no memory chunk exists.
  std::pair<void *, size_t> GetChunkOffset() const;

  /// \brief Set user data.
  ///
  /// \param[in] key The key of user data.
  /// \param[in] value The value of user data, nullptr to erase the user data.
  template <typename T>
  void set_user_data(const std::string &key, const std::shared_ptr<T> &value) {
    user_data_.set<T>(key, value);
  }

  /// \brief Get user data.
  ///
  /// \param[in] key The key of user data.
  ///
  /// \return Pointer to user data.
  template <typename T>
  std::shared_ptr<T> user_data(const std::string &key) const {
    return user_data_.get<T>(key);
  }

  /// \brief Check whether there is corresponding user data by the given key.
  ///
  /// \param[in] key The key of user data.
  ///
  /// \return True if it exists, otherwise false.
  bool has_user_data(const std::string &key) const { return user_data_.has(key); }

  /// \brief Reset tensors data so that they are using contiguous memory chunks grouped by data type.
  ///
  /// \param[in] tensors The tensors to be processed.
  /// \param[in] fusion_size Maximum memory chunk size in bytes, 0 for unlimited.
  ///
  /// \return Tensors that data are pointed to each contiguous memory chunks.
  static TensorPtrList FlattenTensors(const TensorPtrList &tensors, size_t fusion_size = 0);

  /// \brief Check if FlattenTensors called for the input tensors.
  ///
  /// \param[in] tensors The tensors to be checked.
  ///
  /// \return True if FlattenTensors called for input tensors, false otherwise.
  static bool IsFlattened(const TensorPtrList &tensors);

  /// \brief Get tensors for each contiguous memory chunks used by the input tensors.
  ///
  /// \param[in] tensors The input tensors.
  ///
  /// \return Tensors that data are pointed to each contiguous memory chunks, empty if failed.
  static TensorPtrList GetFlattenedTensors(const TensorPtrList &tensors);

  /// \brief Get the fusion size for the given flat tensors.
  ///
  /// \param[in] flat_tensors The input flat tensors.
  ///
  /// \return fusion size for the given flat tensors.
  static size_t GetFusionSize(const TensorPtrList &flat_tensors);

  /// \brief Get the tensor compression type.
  ///
  /// \return tensor compression type.
  TensorCompressionType compression_type() const { return compression_type_; }

  /// \brief If tensor use persistent tensor data.
  ///
  /// \return if use persistent tenor data.
  bool is_persistent_data() const;

  /// \brief Set tensor name.
  ///
  /// \param[in] tensor_name The tensor name.
  void set_name(const std::string &tensor_name) { tensor_name_ = tensor_name; }

  /// \brief Get the tensor name.
  ///
  /// \return tensor name.
  const std::string &name() const { return tensor_name_; }

  /// \brief Set tensor quant param.
  ///
  /// \param[in] quant_param The tensor quant param.
  void set_quant_param(const std::vector<std::shared_ptr<QuantizationParam>> &quant_params) {
    quant_params_.assign(quant_params.begin(), quant_params.end());
  }

  /// \brief Get the tensor quant param.
  ///
  /// \return tensor quant param.
  const std::vector<std::shared_ptr<QuantizationParam>> &quant_params() const { return quant_params_; }

 private:
  void ExecuteLazyTask() const;

  bool init_flag_{false};
  bool adapter_flag_{false};
  bool is_forward_output_{false};
  TensorDataPtr data_{nullptr};
  std::string id_{""};
  mutable std::shared_ptr<WaitEvent> event_{nullptr};
  bool need_wait_{false};
  mutable TensorSyncStatus sync_status_{kNeedSyncHostToDevice};
  bool graph_output_{false};
  bool updated_by_device_{false};
  DeviceSyncPtr device_sync_{nullptr};
  // Release device address of graph output tensor or not.
  bool need_release_device_mem_{false};
  bool cache_enable_{false};
  // Tensor base shape which contain dynamic shape info.
  BaseShapePtr base_shape_ptr_{nullptr};
  std::shared_ptr<Tensor> cache_tensor_ptr_{nullptr};
  std::shared_ptr<Tensor> hashmap_tensor_ptr_{nullptr};
  std::string padding_type_{""};
  TypePtr cast_dtype_{nullptr};
  std::shared_ptr<DeviceEvent> device_event_{nullptr};
  std::function<void(void)> lazy_callback_{nullptr};
  UserData user_data_;
  TensorCompressionType compression_type_{kNoCompression};
  std::vector<std::shared_ptr<QuantizationParam>> quant_params_;
  std::string tensor_name_;
};

// CSRTensor entity class
class MS_CORE_API CSRTensor : public MetaSparseTensor {
 public:
  abstract::AbstractBasePtr ToAbstract() override;

  /// \brief Create CSRTensor with given data type from another tensor.
  ///
  /// \param[in] indptr [Tensor] The indices pointer.
  /// \param[in] indices [Tensor] The indices.
  /// \param[in] values [Tensor] The values.
  /// \param[in] shape The shape represented by ShapeVector of the CSRensor.
  CSRTensor(const TensorPtr indptr, const TensorPtr indices, const TensorPtr values, const ShapeVector &shape);

  /// Destructor of CSRTensor.
  ~CSRTensor() override = default;

  MS_DECLARE_PARENT(CSRTensor, MetaSparseTensor)

  /// \brief Gets CSRTensor's indptr.
  ///
  /// \return [TensorPtr] The indices pointer.
  TensorPtr GetIndptr() { return indptr_; }

  /// \brief Gets CSRTensor's indices.
  ///
  /// \return [TensorPtr] The indices.
  TensorPtr GetIndices() { return indices_; }

  /// \brief Gets CSRTensor's values.
  ///
  /// \return [TensorPtr] The values.
  TensorPtr GetValues() { return values_; }

  /// \brief Compare two csrtensor objects to see if they have same data address.
  ///
  /// \param[in] csr_tensor The csrtensor object to be compared.
  /// \return True if having same data address, otherwise false.
  bool operator==(const CSRTensor &csr_tensor) const { return &csr_tensor == this; }

  bool operator==(const Value &other) const override {
    if (other.isa<CSRTensor>()) {
      auto &other_ = static_cast<const CSRTensor &>(other);
      return *this == other_;
    }
    return false;
  }

  const size_t GetSizeAt(size_t index) const;

  TensorPtr GetTensorAt(size_t index) const;

  const size_t GetTensorLength() const { return kShapeIdx + shape().size(); }

  /// \brief Get display information of this Tensor.
  ///
  /// \return The display information of this Tensor.
  std::string ToString() const override;

  static constexpr size_t kIndptrIdx = 0;
  static constexpr size_t kIndicesIdx = 1;
  static constexpr size_t kValuesIdx = 2;
  static constexpr size_t kShapeIdx = 3;

 private:
  TensorPtr indptr_;
  TensorPtr indices_;
  TensorPtr values_;
};
using CSRTensorPtr = std::shared_ptr<CSRTensor>;

// COOTensor entity class
class MS_CORE_API COOTensor : public MetaSparseTensor {
 public:
  abstract::AbstractBasePtr ToAbstract() override;

  /// \brief Create COOTensor with given data type from another tensor.
  ///
  /// \param[in] indices [Tensor] The indices.
  /// \param[in] values [Tensor] The values.
  /// \param[in] shape The shape represented by ShapeVector of the COOTensor.
  COOTensor(const TensorPtr indices, const TensorPtr values, const ShapeVector &shape)
      : MetaSparseTensor(values->data_type(), shape), indices_(indices), values_(values) {}

  /// Destructor of COOTensor.
  ~COOTensor() override = default;

  MS_DECLARE_PARENT(COOTensor, MetaSparseTensor)

  /// \brief Gets COOTensor's indices.
  ///
  /// \return [TensorPtr] The indices.
  TensorPtr GetIndices() { return indices_; }

  /// \brief Gets COOTensor's values.
  ///
  /// \return [TensorPtr] The values.
  TensorPtr GetValues() { return values_; }

  TensorPtr GetTensorAt(size_t index) const;

  const size_t GetTensorLength() const { return kShapeIdx + shape().size(); }

  /// \brief Compare two cootensor objects to see if they have same address.
  ///
  /// \param[in] coo_tensor The cootensor object to be compared.
  /// \return True if having same data address, otherwise false.
  bool operator==(const COOTensor &coo_tensor) const { return &coo_tensor == this; }

  bool operator==(const Value &other) const override {
    if (other.isa<COOTensor>()) {
      auto &other_ = static_cast<const COOTensor &>(other);
      return *this == other_;
    }
    return false;
  }

  /// \brief Get display information of this Tensor.
  ///
  /// \return The display information of this Tensor.
  std::string ToString() const override;

  static constexpr size_t kIndicesIdx = 0;
  static constexpr size_t kValuesIdx = 1;
  static constexpr size_t kShapeIdx = 2;

 private:
  TensorPtr indices_;
  TensorPtr values_;
};
using COOTensorPtr = std::shared_ptr<COOTensor>;

// RowTensor entity class
class MS_CORE_API RowTensor : public MetaSparseTensor {
 public:
  abstract::AbstractBasePtr ToAbstract() override;

  /// \brief Create RowTensor with given data type from another tensor.
  ///
  /// \param[in] indices [Tensor] The indices.
  /// \param[in] values [Tensor] The values.
  /// \param[in] shape The shape represented by ShapeVector of the RowTensor.
  RowTensor(const TensorPtr indices, const TensorPtr values, const ShapeVector &shape)
      : MetaSparseTensor(values->data_type(), shape), indices_(indices), values_(values) {}

  /// Destructor of RowTensor.
  ~RowTensor() override = default;

  /// \brief Gets RowTensor's indices.
  ///
  /// \return [TensorPtr] The indices.
  TensorPtr GetIndices() { return indices_; }

  /// \brief Gets RowTensor's values.
  ///
  /// \return [TensorPtr] The values.
  TensorPtr GetValues() { return values_; }

  /// \brief Compare two rowtensor objects to see if they have same address.
  ///
  /// \param[in] coo_tensor The rowtensor object to be compared.
  /// \return True if having same data address, otherwise false.
  bool operator==(const RowTensor &row_tensor) const { return &row_tensor == this; }

  bool operator==(const Value &other) const override {
    if (other.isa<RowTensor>()) {
      auto &other_ = static_cast<const RowTensor &>(other);
      return *this == other_;
    }
    return false;
  }

  /// \brief Get display information of this Tensor.
  ///
  /// \return The display information of this Tensor.
  std::string ToString() const override;

 private:
  TensorPtr indices_;
  TensorPtr values_;
};
using RowTensorPtr = std::shared_ptr<RowTensor>;

// Convert shape vector to string.
MS_CORE_API std::string ShapeToString(const ShapeVector &shape);
}  // namespace tensor
}  // namespace mindspore

#endif  // MINDSPORE_CORE_IR_TENSOR_H_

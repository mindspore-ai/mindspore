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

#include <future>
#include <memory>
#include <string>
#include <vector>
#include <numeric>
#include <mutex>
#include <condition_variable>
#include <utility>
#include <algorithm>
#include <iomanip>
#include "device_address/device_address.h"
#include "ir/meta_tensor.h"
#include "device_address/device_type.h"
#include "utils/log_adapter.h"
#include "base/bfloat16.h"
#include "utils/os.h"
#include "ir/meta_grad_data.h"
#include "ir/quantization_param.h"
#include "ir/dtype/op_dtype.h"

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

using ShapeValueDType = int64_t;
using ShapeVector = std::vector<ShapeValueDType>;
using ShapeArray = std::vector<ShapeVector>;
class QuantizationParam;

// Pinned memory register interface.
class MS_CORE_API PinnedMemRegister {
 public:
  /// \brief Default constructor for register.
  PinnedMemRegister() = default;

  /// \brief Virtual destructor for register.
  virtual ~PinnedMemRegister() = default;

  /// \brief Register pinned memory.
  ///
  /// \param[in] addr The host address to pin.
  /// \param[in] size The host data size.
  /// \return Void.
  virtual void RegisterPinnedMem(void *addr, size_t size) = 0;

  /// \brief UnRegister pinned memory.
  ///
  /// \param[in] addr The host address to unpin.
  /// \return Void.
  virtual void UnRegisterPinnedMem(void *addr) = 0;
};

// A sub namespace in ME to support tensor related definition.
namespace tensor {
class Tensor;
using TensorPtr = std::shared_ptr<Tensor>;
using TensorPtrList = std::vector<std::shared_ptr<Tensor>>;
using DeviceAddress = device::DeviceAddress;
using DeviceAddressPtr = device::DeviceAddressPtr;

struct Version {
 public:
  Version() : version_counter_(std::make_shared<uint32_t>(0)) {}
  void BumpVersion() { (*version_counter_)++; }
  uint32_t current_version() const {
    MS_EXCEPTION_IF_NULL(version_counter_);
    return *version_counter_;
  }

 private:
  std::shared_ptr<uint32_t> version_counter_;
};

template <typename T>
class FutureData {
 public:
  FutureData(std::shared_ptr<T> data, std::exception_ptr e_ptr) : data_(std::move(data)), e_ptr_(std::move(e_ptr)) {}
  virtual ~FutureData() {}

  virtual std::shared_ptr<T> GetData() const { return data_; }
  const std::exception_ptr &GetException() const { return e_ptr_; }

 private:
  std::shared_ptr<T> data_;
  std::exception_ptr e_ptr_;
};

template <typename T>
class FutureBase {
 public:
  explicit FutureBase(std::future<std::shared_ptr<tensor::FutureData<T>>> future) : future_(std::move(future)) {}
  virtual ~FutureBase() {}
  virtual std::shared_ptr<T> Get() = 0;

 protected:
  std::future<std::shared_ptr<tensor::FutureData<T>>> future_;
  std::shared_ptr<tensor::FutureData<T>> future_data_;
};
// brief Device info of Tensor
//
// Includes the format, data type and host format of a tensor.
struct DeviceInfo {
  explicit DeviceInfo(const std::unique_ptr<DeviceInfo> &device_info) {
    if (device_info == nullptr) {
      return;
    }
    format_ = device_info->format_;
    data_type_ = device_info->data_type_;
    host_format_ = device_info->host_format_;
    device_id_ = device_info->device_id_;
  }
  explicit DeviceInfo(std::string format = "DefaultFormat", TypePtr data_type = nullptr,
                      std::string host_format = "DefaultFormat", int32_t device_id = 0)
      : format_(std::move(format)),
        data_type_(std::move(data_type)),
        host_format_(std::move(host_format)),
        device_id_(device_id) {}
  std::string format_ = "DefaultFormat";
  TypePtr data_type_ = nullptr;
  std::string host_format_ = "DefaultFormat";
  int32_t device_id_ = 0;
};

// Tensor entity class
class MS_CORE_API Tensor : public MetaTensor {
 public:
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

  Tensor(TypeId data_type, const ShapeVector &shape, DeviceAddressPtr data);

  /// \brief Create a lazy allocated tensor.
  ///
  /// \param[in] data_type [TypeId] Data type of the tensor.
  /// \param[in] shape The shape represented by ShapeVector of the tensor.
  Tensor(TypeId data_type, const ShapeVector &shape);

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

  /// \brief Create a tensor with external data buffer.
  ///
  /// \param[in] data_type [TypeId] Data type of the tensor.
  /// \param[in] shape The shape represented by ShapeVector of the tensor.
  /// \param[in] ref_mem The length of data in bytes.
  /// \param[in] data The input data to be referenced by tensor.
  Tensor(TypeId data_type, const ShapeVector &shape, bool ref_mem, void *data);

  Tensor &operator=(const Tensor &tensor);

  /// Destructor of Tensor.
  ~Tensor() override;

  MS_DECLARE_PARENT(Tensor, MetaTensor);

  /// \brief Compare two tensor objects to see if they have same data type, shape and data address.
  ///
  /// \param[in] tensor The Tensor object to be compared.
  /// \return True if having same type, shape and data address, otherwise false.
  bool operator==(const Tensor &tensor) const;

  /// \brief Create Abstract for Tensor.
  ///
  /// \return Abstract of Tensor.
  abstract::AbstractBasePtr ToAbstract() override;

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

  bool operator==(const Value &other) const override;

  /// \brief Gets tensor's dimension.
  ///
  /// \return The number of dimensions of the tensor data.
  int DataDim() const { return static_cast<int>(shape_.size()); }

  /// \brief Get the data type of the tensor for C++
  ///
  /// \return [int] The tensor's data type will be cast to int to return.
  int data_type_c() const { return static_cast<int>(data_type_); }

  /// \brief Get the tensor's shape for C++
  ///
  /// \return [ShapeVector]
  const ShapeVector &shape_c() const;

  /// \brief Get the tensor's format
  ///
  /// \return Get the data format
  Format format() const;

  /// \brief Set the tensor's format
  ///
  /// \param[in] format The Tensor format to be seted
  void set_format(const Format &format);

  /// \brief Get the tensor's implicit copy format
  ///
  /// \return Get the data format
  Format implicit_copy_format() const { return implicit_copy_format_; }

  /// \brief Set the tensor's implicit copy format
  ///
  /// \param[in] format The Tensor format to be seted
  void set_implicit_copy_format(const Format &format) { implicit_copy_format_ = format; }

  /// \brief Get Tensor data pointer for c++ type
  ///
  /// \return The pointer to the object
  void *data_c() const;

  /// \brief Get Tensor data byte-size for c++ type
  ///
  /// \return byte size of Tensor data
  size_t Size() const { return DataNBytes(); }

  /// \brief Copy Tensor data from device and return new Tensor.
  ///
  /// \return Tensor on CPU.
  TensorPtr cpu() const;

  /// \brief Get total number of Tensor elements.
  ///
  /// \return Total number of Tensor elements.
  size_t DataSize() const { return SizeOf(shape_); }

  /// \brief Get byte size of a single element.
  ///
  /// \return Byte size of a single element.
  ssize_t DataItemSize() const;

  /// \brief Get total number of bytes.
  ///
  /// \return Total number of bytes.
  size_t DataNBytes() const { return DataSize() * DataItemSize(); }

  /// \brief Get number of dimensions.
  ///
  /// \return Number of dimensions.
  ssize_t DataNDim() const { return shape_.size(); }

  /// \brief Get display information about data of this Tensor.
  ///
  /// \param[in] use_comma Whether to use comma.
  /// \return The display information.
  std::string DataToString(bool use_comma) const;

  /// \brief Get the internal data ptr. The ptr maybe null if the data is not initialized.
  ///
  /// \return The ptr in device_address of Tensor.
  const void *unsafe_data() const;

  /// \brief Set the data type of this Tensor.
  ///
  /// \param[in] data_type Tensor data type.
  TypeId set_data_type(TypeId data_type) override;

  /// \brief Set the shape of this Tensor.
  ///
  /// \param[in] shape Tensor shape.
  /// \return The shape's size.
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

  /// \brief Check if this Tensor is used in bprop graph.
  ///
  /// \return Whether this Tensor is used in bprop graph.
  bool used_in_bprop_graph() const { return used_in_bprop_graph_; }

  /// \brief Set used in bprop graph flag of this Tensor.
  ///
  /// \param[in] used_in_bprop_graph Whether this Tensor is forward output.
  void set_used_in_bprop_graph(bool used_in_bprop_graph) { used_in_bprop_graph_ = used_in_bprop_graph; }

  /// \brief Get the device address.
  ///
  /// \return The device address.
  const DeviceAddressPtr &device_address() const;

  device::DeviceType device_type() const {
    MS_EXCEPTION_IF_NULL(device_sync_);
    return device_sync_->GetDeviceType();
  }

  /// \brief Set the device address.
  ///
  /// \param[in] device_sync The input Device synchronization.
  /// \param[in] need_update_ref_count If need_update_ref_count is true, the device address cannot be released and
  /// reused, so the feature map should set false when set device address of tensor.
  void set_device_address(const DeviceAddressPtr &device_sync, bool need_update_ref_count = true);

  void set_(DeviceAddressPtr &&device_sync, const TensorStorageInfoPtr &storage_info, const ShapeVector &shape);

  /// \brief Set origin device address for implicit copy.
  ///
  /// \param[in] device_address Origin device address.
  void set_implicit_copy_address(const DeviceAddressPtr &device_address) { implicit_copy_address_ = device_address; }

  /// \brief Get the device address for implicit copy.
  ///
  /// \return The device address.
  const DeviceAddressPtr &implicit_copy_address() const { return implicit_copy_address_; }

  /// \brief Get the id of this Tensor.
  ///
  /// \return The id of this Tensor.
  uint64_t id() const { return id_; }

  /// \brief Set lazy callback function to this Tensor
  ///
  /// \param[in] lazy_callback Wait for async tasks finish before data_sync.
  static void RegisterLazyCallback(const std::function<void(void)> &lazy_callback) { lazy_callback_ = lazy_callback; }

  /// \brief Contiguous callback function to this Tensor
  ///
  /// \return The contiguous callback function
  const std::function<DeviceAddressPtr(const TensorPtr &)> &contiguous_callback() { return contiguous_callback_; }

  /// \brief Set contiguous callback function to this Tensor
  ///
  /// \param[in] contiguous_callback The callback from backend when need to make tensor contiguous.
  void set_contiguous_callback(const std::function<DeviceAddressPtr(const TensorPtr &)> &contiguous_callback) {
    contiguous_callback_ = contiguous_callback;
  }

  /// @brief Get Pynative auto_grad meta data.
  /// @return Auto grad meta data
  const AutoGradMetaInterfacePtr &auto_grad_meta_data() const { return auto_grad_meta_data_; }

  /// @brief Set Pynative auto_grad meta data.
  /// @param auto_grad_meta_data
  void set_auto_grad_meta_data(const AutoGradMetaInterfacePtr &auto_grad_meta_data) {
    auto_grad_meta_data_ = auto_grad_meta_data;
  }

  /// \brief Check whether the tensor is used in auto grad.
  ///
  /// \return Boolean indicate whether the tensor is used in auto grad.
  bool HasAutoGrad() const {
    return auto_grad_meta_data_ != nullptr && auto_grad_meta_data_->input_type() == InputType::kOpOutput;
  }

  /// \brief Get tensor storage info.
  ///
  /// \return Tensor storage info, the value is nullptr default.
  TensorStorageInfoPtr storage_info() const;

  /// \brief Set tensor storage info.
  ///
  void set_storage_info(const TensorStorageInfoPtr &storage_info);

  /// \brief Get tensor's storage info for implicit copy.
  ///
  /// \return Tensor storage info, the value is nullptr default.
  TensorStorageInfoPtr implicit_copy_storage_info() const;

  /// \brief Set tensor's implicit storage info.
  ///
  void set_implicit_copy_storage_info(const TensorStorageInfoPtr &storage_info);

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

  /// \brief Get tensor's BaseShape.
  ///
  /// \return The BaseShape of this tensor.
  const BaseShapePtr &base_shape_ptr() const { return base_shape_ptr_; }

  /// \brief Set tensor's BaseShape.
  ///
  /// \param[in] BaseShapePtr The tensor's BaseShape.
  void set_base_shape(const BaseShapePtr &base_shape) { base_shape_ptr_ = base_shape; }

  /// \brief Determines whether the memory of tensor is contiguous.
  ///
  /// \return True if tensor memory is contiguous, false otherwise.
  bool is_contiguous() const;

  bool NeedContiguous() const;

  /// \brief Get tensor storage stride.
  ///
  /// \return storage stride.
  std::vector<int64_t> stride() const;

  /// \brief Get tensor storage offset.
  ///
  /// \return storage offset.
  size_t storage_offset() const;

  /// \brief Get version.
  ///
  /// \return storage offset.
  const Version &version() const { return version_; }

  /// \brief Set version.
  ///
  void set_version(const Version &version) { version_ = version; }

  /// \brief Bump version.
  ///
  void BumpVersion() { version_.BumpVersion(); }

  void set_need_pipeline_sync(bool need_pipeline_sync) { need_pipeline_sync_ = need_pipeline_sync; }

  bool need_pipeline_sync() const { return need_pipeline_sync_; }

  /// \brief Execute lazy task.
  ///
  void ExecuteLazyTask() const;

  /// \brief Execute contiguous callback.
  ///
  DeviceAddressPtr CallContiguousCallback() const;

  /// \brief To synchronize data with the device without keeping device address, you need to wait for the data to be
  /// valid.
  ///
  void data_sync_directly(const DeviceAddress *const device_sync, bool need_wait = true) const;

  /// \brief Check if this Tensor is initialized.
  ///
  /// \return Whether this Tensor is initialized.
  bool is_init() const { return init_flag_; }

  /// \brief Set the initialization flag of this Tensor.
  ///
  /// \param[in] flag Whether this Tensor is initialized.
  void set_init_flag(bool flag) { init_flag_ = flag; }

  /// \brief Get the cast dtype of this Tensor.
  ///
  /// \return The cast dtype of this Tensor.
  const TypePtr &cast_dtype() { return cast_dtype_; }

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
  const std::shared_ptr<Tensor> &cache_tensor_ptr() const { return cache_tensor_ptr_; }

  /// \brief Set the pointer of cache tensor.
  ///
  /// \param[in] cache_tensor_ptr The input pointer of cache tensor.
  void set_cache_tensor_ptr(const std::shared_ptr<Tensor> &cache_tensor_ptr = nullptr) {
    cache_tensor_ptr_ = cache_tensor_ptr;
  }

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

  /// \brief Get callback need to execute when value is updated of Tensor.
  ///
  /// \return The callback need to execute when value is updated of Tensor.
  const std::function<void(const Tensor *)> &update_value_callback() const { return update_value_callback_; }

  /// \brief Set callback need to execute when value is updated of Tensor.
  ///
  /// \param[in] update_value_callback The callback need to execute when value is updated of Tensor.
  void set_update_value_callback(const std::function<void(const Tensor *)> &update_value_callback) {
    update_value_callback_ = update_value_callback;
  }

  /// \brief Get tensors stub flag.
  ///
  /// \param[in] none.
  ///
  /// \return If compile with backend, return false, else return true.
  static bool CheckStub();

  /// \brief Get the tensor compression type.
  ///
  /// \return tensor compression type.
  TensorCompressionType compression_type() const { return compression_type_; }

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

  /// \brief Offload tensor to file.
  ///
  /// \return offload tensor success.
  bool Offload(const std::string &file_path);

  /// \brief Get tensor offload file path.
  ///
  /// \return offload file path, or empty string if tensor has not offload.
  const std::string &GetOffloadFilePath() const;

  /// \brief pin tensor memory.
  ///
  /// \param[in] register to pin tensor data.
  void PinMemory(PinnedMemRegister *pin_mem_register);

  /// \brief unpin tensor memory.
  void UnPinMemory();

  /// \brief Get tensor's device info.
  ///
  /// \return The device info of this tensor.
  DeviceInfo device_info() const {
    if (device_info_ != nullptr) {
      return *device_info_;
    }
    device_info_ = std::make_unique<DeviceInfo>();
    return *device_info_;
  }

  /// \brief Set tensor's device info.
  ///
  /// \param[in] device_info The tensor's device info.
  void set_device_info(const DeviceInfo &device_info) { device_info_ = std::make_unique<DeviceInfo>(device_info); }

  ops::OP_DTYPE source_type() const { return source_type_; }

  void set_source_type(ops::OP_DTYPE source_type) { source_type_ = source_type; }

  /// \brief Set tensor's device info.
  ///
  /// \param[in] format The input format.
  /// \param[in] data_type The input data type.
  /// \param[in] host_format The input host format.
  void SetDeviceInfo(const std::string &format, const TypePtr &data_type,
                     const std::string &host_format = "DefaultFormat");

  void set_copy_done_flag(bool flag) { copy_done_flag_ = flag; }
  bool get_copy_done_flag() const { return copy_done_flag_; }
  // Grad interface for PyNative

  /// \brief If tensor requires grad.
  /// \return bool
  bool requires_grad();

  /// \brief Set tensor requires grad.
  void set_requires_grad(bool requires_grad);

  /// \brief If tensor need retains grad.
  /// \return whether the tensor retains grad.
  bool retains_grad();

  /// \brief Set tensor need retains grad.
  void retain_grad();

  /// \brief Get grad of tensor.
  /// \return grad of tensor.
  TensorPtr grad();

  /// \brief Set grad of tensor.
  void set_grad(const TensorPtr &grad);

  /// \brief Set tensor is a leaf node.
  /// \return
  bool is_leaf();
  /// \brief output index of operator.
  /// \return output index of the operator.
  size_t output_index();
  /// \brief grad node of operator.
  /// \return grad node.
  BackwardNodePtr grad_node();
  /// \brief Initialize grad interface so we can call grad impl.
  static void InitializeGradImpl(GradHookInterfacePtr grad_impl);
  /// \brief Grad interface of PyNative.
  static const GradHookInterfacePtr &grad_impl();

  void shallow_copy_from(const Tensor &other);

  bool has_fallback() const { return has_fallback_; }

  void set_has_fallback(bool has_fallback) { has_fallback_ = has_fallback; }

 private:
  inline static GradHookInterfacePtr grad_impl_{nullptr};
  // Really execute callback function when host value is updated of Tensor.
  void ExecuteUpdateValueCallback() const;

  // function size 32
  inline static std::function<void(void)> lazy_callback_{nullptr};
  std::function<DeviceAddressPtr(const TensorPtr &)> contiguous_callback_{nullptr};
  std::function<void(const Tensor *)> update_value_callback_{nullptr};

  // string size 32
  uint64_t id_;
  std::string tensor_name_;
  std::string offload_file_;

  // shared_ptr size 16
  Version version_{};

  Format format_{Format::DEFAULT_FORMAT};
  Format implicit_copy_format_{Format::DEFAULT_FORMAT};

  mutable DeviceAddressPtr device_sync_{nullptr};
  TensorStorageInfoPtr storage_info_{nullptr};

  mutable DeviceAddressPtr implicit_copy_address_{nullptr};
  TensorStorageInfoPtr implicit_copy_storage_info_{nullptr};

  AutoGradMetaInterfacePtr auto_grad_meta_data_{nullptr};
  // Tensor base shape which contain dynamic shape info.
  BaseShapePtr base_shape_ptr_{nullptr};
  std::shared_ptr<Tensor> cache_tensor_ptr_{nullptr};
  std::shared_ptr<Tensor> hashmap_tensor_ptr_{nullptr};
  TypePtr cast_dtype_{nullptr};
  std::vector<std::shared_ptr<QuantizationParam>> quant_params_;

  // pointer size 8
  ops::OP_DTYPE source_type_{ops::OP_DTYPE::DT_BEGIN};
  mutable std::unique_ptr<DeviceInfo> device_info_;
  PinnedMemRegister *pin_mem_register_{nullptr};

  // enum size 4
  mutable TensorSyncStatus sync_status_{kNeedSyncHostToDevice};
  TensorCompressionType compression_type_{kNoCompression};

  // bool size 1
  bool used_in_bprop_graph_{true};
  bool need_pipeline_sync_{false};
  bool init_flag_{false};
  bool graph_output_{false};
  bool updated_by_device_{false};
  bool cache_enable_{false};
  bool copy_done_flag_{false};
  bool has_fallback_{false};
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

  bool operator==(const Value &other) const override;

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
  COOTensor(const TensorPtr indices, const TensorPtr values, const ShapeVector &shape);

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

  bool operator==(const Value &other) const override;

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
  RowTensor(const TensorPtr indices, const TensorPtr values, const ShapeVector &shape);

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

  bool operator==(const Value &other) const override;

  /// \brief Get display information of this Tensor.
  ///
  /// \return The display information of this Tensor.
  std::string ToString() const override;

 private:
  TensorPtr indices_;
  TensorPtr values_;
};

// Convert shape vector to string.
MS_CORE_API std::string ShapeToString(const ShapeVector &shape);

using RowTensorPtr = std::shared_ptr<RowTensor>;
}  // namespace tensor
// Tensor copy interface
MS_CORE_API bool SyncCopy(const tensor::TensorPtr &dst, const tensor::TensorPtr &src, size_t stream_id);
MS_CORE_API bool AsyncCopy(const tensor::TensorPtr &dst, const tensor::TensorPtr &src, size_t stream_id,
                           bool keep_src = true);
}  // namespace mindspore

#endif  // MINDSPORE_CORE_IR_TENSOR_H_

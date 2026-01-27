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

#ifndef MINDSPORE_CORE_IR_DEVICE_ADDRESS_H_
#define MINDSPORE_CORE_IR_DEVICE_ADDRESS_H_

#include <vector>
#include <memory>
#include <string>
#include <utility>
#include <map>
#include <unordered_map>
#include <mutex>
#include <optional>
#include "ir/dtype.h"
#include "utils/ms_utils.h"
#include "utils/shape_utils.h"
#include "ir/dtype/type.h"
#include "ir/tensor_data.h"
#include "ir/tensor_storage_info.h"
#include "mindapi/base/format.h"
#include "mindapi/base/types.h"
#include "device_address/device_type.h"
#include "ir/format_utils.h"
#include "device_address/map_memory_allocator.h"

using std::string;

namespace mindspore {
namespace tensor {
class TensorData;
using TensorDataPtr = std::shared_ptr<TensorData>;
}  // namespace tensor
class MS_CORE_API AddressAllocator {
 public:
  /**
   * @brief Allocate memory for device address
   * @param size - The size of memory that needs to be allocated
   * @param stream_id - Stream ID for memory allocation
   * @return Raw pointer to the allocated memory
   */
  virtual void *Alloc(size_t size, uint32_t stream_id) = 0;

  /**
   * @brief Free memory for device address
   * @param address_ptr - Raw pointer in DevicePointer that needs to be freed
   * @return true if free succeeds, false otherwise
   */
  virtual bool Free(void *address_ptr) = 0;

  virtual bool IsPinned() { return false; }

  virtual void *GetHostPtrByDevicePtr(void *devicePtr) { return nullptr; }
};

// DevicePointer encapsulates pointer and reference count-related operations, and supports custom allocator and
// delteter resources. In Ref scenarios, KernelTensor of different DeviceAddress may hold the same DevicePointer
// object.
class MS_CORE_API DevicePointer {
 public:
  // The arguments are pointer and a bool variable that identifies whether pointer is from the memory pool.
  using Deleter = std::function<void(void *, bool)>;

  DevicePointer() = default;
  explicit DevicePointer(void *ptr) : ptr_(ptr) {}
  DevicePointer(void *ptr, const Deleter &deleter, std::shared_ptr<AddressAllocator> allocator = nullptr)
      : ptr_(ptr), deleter_(deleter), allocator_(std::move(allocator)) {}

  DevicePointer(const DevicePointer &) = delete;
  DevicePointer &operator=(const DevicePointer &) = delete;

  ~DevicePointer() {
    try {
      if (map_allocator_) {
        map_allocator_->Free(ptr_);
      } else if (ptr_ != nullptr && allocator_ && from_mem_pool_) {
        allocator_->Free(ptr_);
      } else if (ptr_ != nullptr && deleter_) {
        deleter_(ptr_, from_mem_pool_);
      }
      ptr_ = nullptr;
    } catch (const std::exception &e) {
      MS_LOG(ERROR) << "DevicePointer destructed failed: " << e.what();
    } catch (...) {
      MS_LOG(ERROR) << "DevicePointer destructed failed.";
    }
  }

  std::string ToString() const {
    std::ostringstream ofs;
    ofs << this << " ptr:" << ptr_ << " from mem pool:" << from_mem_pool_ << " deleter:" << (deleter_ != nullptr);
    return ofs.str();
  }

  // Get raw pointer.
  void *ptr() const { return ptr_; }
  // Set raw pointer.
  void set_ptr(void *ptr) { ptr_ = ptr; }

  // Get whether pointer in DevicePointer is allocated from the memory pool.
  bool from_mem_pool() const { return from_mem_pool_; }
  // Set whether pointer in DevicePointer is allocated from the memory pool.
  void set_from_mem_pool(bool from_mem_pool) { from_mem_pool_ = from_mem_pool; }

  // Get pointer resource destructor.
  Deleter deleter() const { return deleter_; }

  // Set pointer resource destructor.
  void set_deleter(const Deleter &deleter) { deleter_ = deleter; }

  std::shared_ptr<AddressAllocator> allocator() const { return allocator_; }

  void set_allocator(std::shared_ptr<AddressAllocator> allocator) { allocator_ = allocator; }

  const std::unique_ptr<MapAllocator> &map_allocator() const { return map_allocator_; }

  void set_map_allocator(std::unique_ptr<MapAllocator> &&map_allocator) { map_allocator_ = std::move(map_allocator); }

 private:
  void *ptr_{nullptr};

  // Whether ptr_  is allocated from the memory pool.
  bool from_mem_pool_{false};

  // The pointer resource destructor.
  Deleter deleter_;

  // The device address allocator that contains allocate memory and delete memory functions.
  std::shared_ptr<AddressAllocator> allocator_;

  std::unique_ptr<MapAllocator> map_allocator_;
};
using DevicePointerPtr = std::shared_ptr<DevicePointer>;

enum class NeedAllocateHeteRes : int64_t { NoNeedHeteRes = 0, NeedHostMem = 1, NeedDiskFile = 2 };
struct HeterogeneousInfo {
  // Address on cpu ddr when the KernelTensor is stored on CPU.
  void *host_ptr_;
  // File name when the KernelTensor is stored on Disk.
  std::string file_name_;
  // Token for unfinished async io.
  std::optional<size_t> aio_token_;
  // Mark which heterogeneous resource should be allocated.
  NeedAllocateHeteRes need_alloc_hete_res_{NeedAllocateHeteRes::NoNeedHeteRes};
  std::string ToString() const {
    std::ostringstream ofs;
    ofs << this << " host ptr:" << host_ptr_ << " file name:" << file_name_
        << " need alloc hete res:" << need_alloc_hete_res_;
    return ofs.str();
  }
};
using HeterogeneousInfoPtr = std::shared_ptr<HeterogeneousInfo>;
using KernelWithIndex = std::pair<AnfNodePtr, size_t>;

enum class StorageType { kDevice, kHost, kFile };
namespace device {
// The flag of device address.
constexpr size_t kDeviceAddressFlagInit = 0;
// Indicates that it is the device address of ref node.
constexpr size_t kDeviceAddressFlagRefNode = 1;
// Indicates that it is the device address of node which has no user.
constexpr size_t kDeviceAddressFlagNotUsed = 2;
// Indicates that it is the device address of node has init arg and do not need device address.
constexpr size_t kDeviceAddressFlagIgnoreDevicePtr = 4;
// Indicates that it is the ptr of device address is nullptr.
constexpr size_t kDeviceAddressFlagNullptr = 8;
// Interface for data synchornize between device and host.
class MS_CORE_API DeviceAddress {
 public:
  using DeviceAddressPtr = std::shared_ptr<DeviceAddress>;
  explicit DeviceAddress(device::DeviceType device_type = device::DeviceType::kUnknown);
  explicit DeviceAddress(void *ptr, size_t size, device::DeviceType device_type, uint32_t stream_id = 0);
  explicit DeviceAddress(const DeviceAddress &other);
  DeviceAddress &operator=(const DeviceAddress &) = delete;
  virtual ~DeviceAddress();

  std::string ToString() const;

  DeviceAddressPtr CloneDeviceAddress();

  const void *GetPtr() const;
  void set_ptr(void *ptr);
  size_t GetSize() const;
  void SetSize(size_t size);

  const std::string &padding_type() const;
  void set_padding_type(const std::string &padding_type);
  bool from_mem_pool() const;
  void set_from_mem_pool(bool from_mem_pool) const;
  virtual void set_communication_ptr(uint8_t *communication_ptr);
  bool from_persistent_mem() const;
  void set_from_persistent_mem(bool from_persistent_mem);
  bool need_recycle() const;
  void set_need_recycle(bool need_recycle);
  void *GetMutablePtr() const;

  device::DeviceType GetDeviceType() const;
  void SetDeviceType(const device::DeviceType &device_type);

  uint32_t device_id() const;

  void set_stream_id(uint32_t stream_id);
  const uint32_t stream_id() const;

  void AddHeldByNode(const std::weak_ptr<ValueNode> &value_node);
  std::vector<std::weak_ptr<ValueNode>> held_by_nodes() const;
  void ClearHeldByNodes();

  void SetNodeIndex(const AnfNodePtr &node, size_t out_index);
  KernelWithIndex GetNodeIndex() const;

  // Return whether DeviceAddress has a valid ptr.
  bool IsPtrValid() const;

  void Swap(DeviceAddress *other);

  std::pair<AnfNodeWeakPtr, size_t> node_index() const;
  void SetDevicePointerDeleter(std::function<void(void *, bool)> &&deleter);

  const DevicePointerPtr &device_pointer() const;
  void set_device_pointer(const DevicePointerPtr &ptr_ref_cnt);

  size_t size() const { return size_; }

  void set_allocator(const std::shared_ptr<AddressAllocator> &allocator) { device_pointer_->set_allocator(allocator); }

  std::shared_ptr<AddressAllocator> allocator() const { return device_pointer_->allocator(); }

  bool remote() const { return remote_; }
  void set_remote(bool remote) { remote_ = remote; }

  void set_map_allocator(std::unique_ptr<MapAllocator> &&map_allocator) {
    device_pointer_->set_map_allocator(std::move(map_allocator));
  }

  const std::unique_ptr<MapAllocator> &map_allocator() const { return device_pointer_->map_allocator(); }

  void set_data(tensor::TensorDataPtr &&data);
  const tensor::TensorDataPtr &data() const;
  bool has_data() const;

  void ClearDeviceMemory();

  void *GetDevicePtr() const { return device_pointer_->ptr(); }
  void SetDevicePtr(void *ptr) const { device_pointer_->set_ptr(ptr); }

 protected:
  // Set a device pointer destructor to kernel tensor, used to release resource reclaiming of the device pointer
  // automatically when DeviceAddress destructed.
  void SetDevicePtrDeleter();

  // {node, out_index}
  std::pair<AnfNodeWeakPtr, size_t> node_index_{AnfNodePtr(nullptr), 0};
  // The DeviceAddress is held by ValueNodes. These ValueNodes are outputs of forward network.
  // We need to release the device memory when the reference count of the device address in bprop graph is 0.
  std::vector<std::weak_ptr<ValueNode>> held_by_nodes_;

  bool from_persistent_mem_{false};
  bool need_recycle_{false};

  // The padding type corresponds to data format.
  std::string padding_type_;

  // the data for numpy object.
  tensor::TensorDataPtr data_;

  DevicePointerPtr device_pointer_;
  uint32_t stream_id_{0};
  size_t size_{0};
  // The device target name, such as "GPU","Ascend".
  device::DeviceType device_type_{device::DeviceType::kUnknown};

  bool remote_{false};
};

using DeviceAddressPtr = std::shared_ptr<DeviceAddress>;
using DeviceAddressPtrList = std::vector<DeviceAddressPtr>;

using DevicePtrDeleterMakerFunc = std::function<void(void *, bool)>;
MS_CORE_API void SetDevicePtrDeleterMaker(device::DeviceType device_type, DevicePtrDeleterMakerFunc &&func);

template <device::DeviceType t>
struct DevicePtrDeleterMakerRegister {
  explicit DevicePtrDeleterMakerRegister(DevicePtrDeleterMakerFunc &&maker) {
    SetDevicePtrDeleterMaker(t, std::move(maker));
  }
};

#define REGISTER_DEVICE_PTR_DELETER_MAKER(t, f)                        \
  namespace {                                                          \
  static DevicePtrDeleterMakerRegister<t> g_deleter_maker_register(f); \
  }
}  // namespace device

struct DeviceAddressExt {
  Format format_{Format::DEFAULT_FORMAT};
  TypeId dtype_id_{kTypeUnknown};
  ShapeVector shape_vector_{};
  TensorStorageInfoPtr tensor_storage_info_{nullptr};
  std::string ToString() const {
    std::ostringstream ofs;
    ofs << this << " format:" << kernel::GetFormatFromEnumToStr(format_) << " type id:" << TypeIdLabel(dtype_id_)
        << " shape: {";
    std::for_each(shape_vector_.begin(), shape_vector_.end(), [&ofs](auto axis) { ofs << axis << " "; });
    ofs << "}"
        << " tensor_storage_info: " << (tensor_storage_info_ != nullptr ? tensor_storage_info_->ToString() : "null");
    return ofs.str();
  }
  DeviceAddressExt(Format format, TypeId dtype_id, const ShapeVector &shape_vector,
                   const TensorStorageInfoPtr &tensor_storage_info = nullptr)
      : format_(format), dtype_id_(dtype_id), shape_vector_(shape_vector), tensor_storage_info_(tensor_storage_info) {}
};
using DeviceAddressExtPtr = std::shared_ptr<DeviceAddressExt>;

using DeviceAddressPtr = device::DeviceAddressPtr;
using SyncCopyFunc = std::function<bool(const DeviceAddressPtr &, const DeviceAddressPtr &, size_t,
                                        const DeviceAddressExtPtr &, const DeviceAddressExtPtr &)>;
using AsyncCopyFunc = std::function<bool(const DeviceAddressPtr &, const DeviceAddressPtr &, size_t, bool,
                                         const DeviceAddressExtPtr &, const DeviceAddressExtPtr &)>;
using SyncPtrFunc = std::function<bool(void *, const void *, uint64_t, size_t)>;

MS_CORE_API void SetCopyFunc(device::DeviceType device_type, SyncCopyFunc &&sync_func, AsyncCopyFunc &&async_func,
                             SyncPtrFunc &&sync_ptr_func);

template <device::DeviceType t>
struct CopyFuncRegister {
  explicit CopyFuncRegister(SyncCopyFunc &&sync_func, AsyncCopyFunc &&async_func, SyncPtrFunc &&sync_ptr_func) {
    SetCopyFunc(t, std::move(sync_func), std::move(async_func), std::move(sync_ptr_func));
  }
};

#define MS_REGISTER_HAL_COPY_FUNC(device_type, sync_func, async_func, sync_ptr_func)           \
  namespace {                                                                                  \
  static CopyFuncRegister<device_type> g_maker_register(sync_func, async_func, sync_ptr_func); \
  }
MS_CORE_API bool CopyToHost(device::DeviceType device_type, void *dst, const void *src, uint64_t size,
                            size_t stream_id);
// DeviceAddressExtPtr record the type shape and format information of the device address. If not provided, the
// copy interface will simply copy the data; otherwise, it will perform conversions for different types and formats.
MS_CORE_API bool SyncCopy(const DeviceAddressPtr &dst_device_address, const DeviceAddressPtr &src_device_address,
                          size_t stream_id, const DeviceAddressExtPtr &src_ext = nullptr,
                          const DeviceAddressExtPtr &dst_ext = nullptr);
MS_CORE_API bool AsyncCopy(const DeviceAddressPtr &dst_device_address, const DeviceAddressPtr &src_device_address,
                           size_t stream_id, bool keep_src = true, const DeviceAddressExtPtr &src_ext = nullptr,
                           const DeviceAddressExtPtr &dst_ext = nullptr);
MS_CORE_API bool HostCopy(const DeviceAddressPtr &dst_device_address, const DeviceAddressPtr &src_device_address,
                          const DeviceAddressExtPtr &src_ext = nullptr, const DeviceAddressExtPtr &dst_ext = nullptr);
}  // namespace mindspore
#endif  // MINDSPORE_CORE_IR_DEVICE_ADDRESS_H_

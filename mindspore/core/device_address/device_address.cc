/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "device_address/device_address.h"
#include <complex>
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <unordered_map>
#include "ir/format_utils.h"
#include "utils/ms_context.h"
#include "device_address/convert_tensor_utils.h"

namespace mindspore {
SyncCopyFunc g_sync_copy_func[static_cast<int>(device::DeviceType::kDeviceEnd)];
AsyncCopyFunc g_async_copy_func[static_cast<int>(device::DeviceType::kDeviceEnd)];
SyncPtrFunc g_sync_ptr_func[static_cast<int>(device::DeviceType::kDeviceEnd)];

void SetCopyFunc(device::DeviceType device_type, SyncCopyFunc &&sync_func, AsyncCopyFunc &&async_func,
                 SyncPtrFunc &&sync_ptr_func) {
  MS_LOG(INFO) << "Resigter copy function for device type:" << device_type;
  g_sync_copy_func[static_cast<int>(device_type)] = sync_func;
  g_async_copy_func[static_cast<int>(device_type)] = async_func;
  g_sync_ptr_func[static_cast<int>(device_type)] = sync_ptr_func;
}

namespace {
bool Copy(void *dst, const void *src, uint64_t size) {
  if (size == 0) {
    return true;
  }
  MS_EXCEPTION_IF_NULL(dst);
  MS_EXCEPTION_IF_NULL(src);
  auto ret_code = memcpy_s(dst, size, src, size);
  if (ret_code == ERANGE) {
    device::ConvertSameType(dst, src, size, kNumberTypeUInt8);
  } else if (ret_code != EOK) {
    MS_LOG(ERROR) << "Failed to copy tensor from ptr:" << src << " to :" << dst << " size:" << size;
    return false;
  }
  return true;
}
}  // namespace

bool CopyToHost(device::DeviceType device_type, void *dst, const void *src, uint64_t size, size_t stream_id) {
  if (device_type == device::DeviceType::kCPU) {
    return Copy(dst, src, size);
  }
  MS_EXCEPTION_IF_NULL(g_sync_ptr_func[static_cast<int>(device_type)]);
  return g_sync_ptr_func[static_cast<int>(device_type)](dst, src, size, stream_id);
}

bool SyncCopy(const DeviceAddressPtr &dst_device_address, const DeviceAddressPtr &src_device_address, size_t stream_id,
              const DeviceAddressExtPtr &src_ext, const DeviceAddressExtPtr &dst_ext) {
  MS_EXCEPTION_IF_NULL(dst_device_address);
  MS_EXCEPTION_IF_NULL(src_device_address);
  if (dst_device_address->GetDeviceType() == device::DeviceType::kUnknown ||
      src_device_address->GetDeviceType() == device::DeviceType::kUnknown) {
    MS_LOG(EXCEPTION) << "Invalid device type for device address:" << dst_device_address
                      << " type:" << dst_device_address->GetDeviceType() << " or device address:" << src_device_address
                      << " type:" << src_device_address->GetDeviceType() << " stream id:" << stream_id;
  }
  if (dst_device_address->GetDeviceType() == device::DeviceType::kCPU &&
      src_device_address->GetDeviceType() == device::DeviceType::kCPU) {
    return HostCopy(dst_device_address, src_device_address, src_ext, dst_ext);
  }
  if (dst_device_address->GetDeviceType() == device::DeviceType::kAscend ||
      src_device_address->GetDeviceType() == device::DeviceType::kAscend) {
    MS_EXCEPTION_IF_NULL(g_sync_copy_func[static_cast<int>(device::DeviceType::kAscend)]);
    return g_sync_copy_func[static_cast<int>(device::DeviceType::kAscend)](dst_device_address, src_device_address,
                                                                           stream_id, src_ext, dst_ext);
  }
  MS_EXCEPTION_IF_NULL(g_sync_copy_func[static_cast<int>(device::DeviceType::kGPU)]);
  return g_sync_copy_func[static_cast<int>(device::DeviceType::kGPU)](dst_device_address, src_device_address, stream_id,
                                                                      src_ext, dst_ext);
}

bool AsyncCopy(const DeviceAddressPtr &dst_device_address, const DeviceAddressPtr &src_device_address, size_t stream_id,
               bool keep_host, const DeviceAddressExtPtr &src_ext, const DeviceAddressExtPtr &dst_ext) {
  MS_EXCEPTION_IF_NULL(dst_device_address);
  MS_EXCEPTION_IF_NULL(src_device_address);
  if (dst_device_address->GetDeviceType() == device::DeviceType::kUnknown ||
      src_device_address->GetDeviceType() == device::DeviceType::kUnknown) {
    MS_LOG(EXCEPTION) << "Invalid device type for device address:" << dst_device_address
                      << " type:" << dst_device_address->GetDeviceType() << " or device address:" << src_device_address
                      << " type:" << src_device_address->GetDeviceType() << " stream id:" << stream_id;
  }
  if (dst_device_address->GetDeviceType() == device::DeviceType::kCPU &&
      src_device_address->GetDeviceType() == device::DeviceType::kCPU) {
    return HostCopy(dst_device_address, src_device_address, src_ext, dst_ext);
  }
  if (dst_device_address->GetDeviceType() == device::DeviceType::kAscend ||
      src_device_address->GetDeviceType() == device::DeviceType::kAscend) {
    MS_EXCEPTION_IF_NULL(g_sync_copy_func[static_cast<int>(device::DeviceType::kAscend)]);
    return g_async_copy_func[static_cast<int>(device::DeviceType::kAscend)](dst_device_address, src_device_address,
                                                                            stream_id, keep_host, src_ext, dst_ext);
  }
  MS_EXCEPTION_IF_NULL(g_sync_copy_func[static_cast<int>(device::DeviceType::kGPU)]);
  return g_async_copy_func[static_cast<int>(device::DeviceType::kGPU)](dst_device_address, src_device_address,
                                                                       stream_id, keep_host, src_ext, dst_ext);
}

void CheckHeterogeneousCopy(const DeviceAddressPtr &dst_device_address, const TensorStorageInfoPtr &dst_storage_info,
                            const DeviceAddressPtr &src_device_address, const TensorStorageInfoPtr &src_storage_info) {
  const auto &dst_device_type = dst_device_address->GetDeviceType();
  const auto &src_device_type = src_device_address->GetDeviceType();
  if (dst_device_type == src_device_type) {
    return;
  }
  if ((dst_device_type == device::DeviceType::kAscend && src_device_type == device::DeviceType::kCPU) &&
      (dst_storage_info != nullptr || src_storage_info != nullptr)) {
    MS_LOG(EXCEPTION) << "Invalid sync host to device for tensor with storage info, src device address: "
                      << src_device_address->ToString() << ", dst device_address: " << dst_device_address->ToString();
  }
  if ((dst_device_type == device::DeviceType::kCPU && src_device_type == device::DeviceType::kAscend) &&
      (dst_storage_info != nullptr || src_storage_info != nullptr)) {
    MS_LOG(WARNING) << "Invalid sync device to host for tensor with storage info, src device address: "
                    << src_device_address->ToString() << ", dst device_address: " << dst_device_address->ToString();
  }
}

bool SyncCopy(const DeviceAddressPtr &dst_device_address, const TensorStorageInfoPtr &dst_storage_info,
              const DeviceAddressPtr &src_device_address, const TensorStorageInfoPtr &src_storage_info,
              size_t stream_id) {
  CheckHeterogeneousCopy(dst_device_address, dst_storage_info, src_device_address, src_storage_info);
  return SyncCopy(dst_device_address, src_device_address, stream_id);
}

bool AsyncCopy(const DeviceAddressPtr &dst_device_address, const TensorStorageInfoPtr &dst_storage_info,
               const DeviceAddressPtr &src_device_address, const TensorStorageInfoPtr &src_storage_info,
               size_t stream_id, bool keep_src) {
  CheckHeterogeneousCopy(dst_device_address, dst_storage_info, src_device_address, src_storage_info);
  return AsyncCopy(dst_device_address, src_device_address, stream_id, keep_src);
}
namespace device {
DevicePtrDeleterMakerFunc g_deleter_func[static_cast<int>(device::DeviceType::kDeviceEnd)];
void SetDevicePtrDeleterMaker(device::DeviceType device_type, DevicePtrDeleterMakerFunc &&func) {
  MS_LOG(DEBUG) << "Resigter device ptr deleter function for device type:" << device::GetDeviceNameByType(device_type);
  g_deleter_func[static_cast<int>(device_type)] = func;
}

DeviceAddress::DeviceAddress(device::DeviceType device_type) {
  device_pointer_ = std::make_shared<DevicePointer>();
  device_type_ = device_type;
  MS_LOG(DEBUG) << "Construct device address:" << ToString();
}

DeviceAddress::DeviceAddress(void *ptr, size_t size, device::DeviceType device_type, uint32_t stream_id)
    : device_pointer_(std::make_shared<DevicePointer>(ptr)),
      stream_id_(stream_id),
      size_(size),
      device_type_(device_type) {
  SetDevicePtrDeleter();
  MS_LOG(DEBUG) << "Construct device address:" << ToString();
}

DeviceAddress::DeviceAddress(const DeviceAddress &other) {
  device_pointer_ = other.device_pointer_ != nullptr
                      ? std::make_shared<DevicePointer>(other.device_pointer_->ptr(), other.device_pointer_->deleter(),
                                                        other.device_pointer_->allocator())
                      : std::make_shared<DevicePointer>();
  stream_id_ = other.stream_id_;
  size_ = other.size_;
  device_type_ = other.device_type_;
  padding_type_ = other.padding_type();
  SetDevicePtrDeleter();
  MS_LOG(DEBUG) << "Construct device address:" << ToString();
}

DeviceAddress::~DeviceAddress() {
  MS_LOG(DEBUG) << "Destruct device address:" << ToString();
  device_pointer_ = nullptr;
}

std::string DeviceAddress::ToString() const {
  std::ostringstream ofs;
  ofs << this << " device type:" << GetDeviceType();
  ofs << " size:" << size_ << " device id:" << device_id()
      << " device name:" << device::GetDeviceNameByType(device_type_) << " device point:";
  if (device_pointer_ == nullptr) {
    ofs << "0";
  } else {
    ofs << device_pointer_->ToString();
  }
  const auto &node_index = GetNodeIndex();
  if (node_index.first != nullptr) {
    ofs << " node:" << node_index.first->fullname_with_scope() << " index:" << node_index.second;
  }
  ofs << " from persist mem:" << from_persistent_mem_ << " need recycle:" << need_recycle_
      << " padding type:" << padding_type_;
  return ofs.str();
}

const void *DeviceAddress::GetPtr() const { return GetDevicePtr(); }

void DeviceAddress::set_ptr(void *ptr) { device_pointer_->set_ptr(ptr); }

size_t DeviceAddress::GetSize() const { return size(); }

void DeviceAddress::SetSize(size_t size) { size_ = size; }

const std::string &DeviceAddress::padding_type() const { return padding_type_; }

void DeviceAddress::set_padding_type(const std::string &padding_type) { padding_type_ = padding_type; }

bool DeviceAddress::from_mem_pool() const { return device_pointer_->from_mem_pool(); }

void DeviceAddress::set_from_mem_pool(bool from_mem_pool) const { device_pointer_->set_from_mem_pool(from_mem_pool); }

void DeviceAddress::set_communication_ptr(uint8_t *communication_ptr) { MS_LOG(EXCEPTION) << "Not implemented error."; }

bool DeviceAddress::from_persistent_mem() const { return from_persistent_mem_; }

void DeviceAddress::set_from_persistent_mem(bool from_persistent_mem) { from_persistent_mem_ = from_persistent_mem; }

bool DeviceAddress::need_recycle() const { return need_recycle_; }

void DeviceAddress::set_need_recycle(bool need_recycle) { need_recycle_ = need_recycle; }

void *DeviceAddress::GetMutablePtr() const { return GetDevicePtr(); }

device::DeviceType DeviceAddress::GetDeviceType() const { return device_type_; }
void DeviceAddress::SetDeviceType(const device::DeviceType &device_type) { device_type_ = device_type; }

uint32_t DeviceAddress::device_id() const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  return device_id;
}

void DeviceAddress::set_stream_id(uint32_t stream_id) { stream_id_ = stream_id; }

const uint32_t DeviceAddress::stream_id() const { return stream_id_; }

void DeviceAddress::AddHeldByNode(const std::weak_ptr<ValueNode> &value_node) {
  (void)held_by_nodes_.emplace_back(value_node);
}

std::vector<std::weak_ptr<ValueNode>> DeviceAddress::held_by_nodes() const { return held_by_nodes_; }

void DeviceAddress::ClearHeldByNodes() { held_by_nodes_.clear(); }

void DeviceAddress::SetNodeIndex(const AnfNodePtr &node, size_t out_index) { node_index_ = {node, out_index}; }

KernelWithIndex DeviceAddress::GetNodeIndex() const {
  return node_index_.first.expired() ? KernelWithIndex{nullptr, node_index_.second}
                                     : KernelWithIndex{node_index_.first.lock(), node_index_.second};
}

bool DeviceAddress::IsPtrValid() const { return GetDevicePtr() != nullptr; }

void DeviceAddress::Swap(DeviceAddress *other) {
  MS_EXCEPTION_IF_NULL(other);
  if (other == this) {
    return;
  }
  other->SetDevicePtr(GetDevicePtr());

  other->set_from_mem_pool(this->from_mem_pool());
  SetDevicePtr(nullptr);
  this->set_from_mem_pool(false);
}

std::pair<AnfNodeWeakPtr, size_t> DeviceAddress::node_index() const { return node_index_; }

void DeviceAddress::SetDevicePointerDeleter(std::function<void(void *, bool)> &&deleter) {
  device_pointer()->set_deleter(deleter);
}

const DevicePointerPtr &DeviceAddress::device_pointer() const { return device_pointer_; }

void DeviceAddress::set_device_pointer(const DevicePointerPtr &device_pointer) {
  MS_EXCEPTION_IF_NULL(device_pointer);
  device_pointer_ = device_pointer;
}

DeviceAddressPtr DeviceAddress::CloneDeviceAddress() { return std::make_shared<DeviceAddress>(*this); }

void DeviceAddress::set_data(tensor::TensorDataPtr &&data) {
  if (GetDeviceType() == device::DeviceType::kCPU) {
    data_ = std::move(data);
  } else {
    MS_LOG(DEBUG) << "Skip device address set_data";
  }
}

const tensor::TensorDataPtr &DeviceAddress::data() const {
  if (GetDeviceType() == device::DeviceType::kCPU) {
    return data_;
  } else {
    MS_LOG(EXCEPTION) << "Not implement exception";
  }
}

bool DeviceAddress::has_data() const {
  if (GetDeviceType() == device::DeviceType::kCPU) {
    return data_ != nullptr;
  } else {
    return false;
  }
}

namespace {
DevicePtrDeleterMakerFunc GetDevicePtrDeleterMaker(device::DeviceType device_type) {
  auto maker = g_deleter_func[static_cast<int>(device_type)];
  return maker;
}
}  // namespace

void DeviceAddress::SetDevicePtrDeleter() {
  if (device_pointer_ == nullptr) {
    return;
  }
  auto deleter = GetDevicePtrDeleterMaker(GetDeviceType());
  if (deleter != nullptr) {
    device_pointer_->set_deleter(deleter);
  } else {
    MS_LOG(INFO) << "Get device ptr deleter function failed, device type: "
                 << device::GetDeviceNameByType(GetDeviceType());
  }
}

void DeviceAddress::ClearDeviceMemory() {
  if (device_pointer_ == nullptr) {
    return;
  }
  auto deleter = device_pointer_->deleter();
  if (GetDevicePtr() != nullptr && from_mem_pool() && deleter) {
    deleter(GetDevicePtr(), from_mem_pool());
    SetDevicePtr(nullptr);
  }
}
}  // namespace device

namespace {

// clang-format off
#define FOR_EACH_TYPE_BASE(M)                    \
  M(kNumberTypeBool, bool)                       \
  M(kNumberTypeUInt8, uint8_t)                   \
  M(kNumberTypeInt4, int8_t)                     \
  M(kNumberTypeInt8, int8_t)                     \
  M(kNumberTypeInt16, int16_t)                   \
  M(kNumberTypeInt32, int32_t)                   \
  M(kNumberTypeInt64, int64_t)                   \
  M(kNumberTypeUInt16, uint16_t)                 \
  M(kNumberTypeUInt32, uint32_t)                 \
  M(kNumberTypeUInt64, uint64_t)                 \
  M(kNumberTypeFloat16, float16)                 \
  M(kNumberTypeFloat32, float)                   \
  M(kNumberTypeFloat64, double)                  \
  M(kNumberTypeFloat8E4M3FN, float8_e4m3fn)      \
  M(kNumberTypeFloat8E5M2, float8_e5m2)          \
  M(kNumberTypeHiFloat8, hifloat8)               \
  M(kNumberTypeComplex64, ComplexStorage<float>) \
  M(kNumberTypeComplex128, ComplexStorage<double>)

#ifndef KERNEL_EXECUTOR_ANDROID
#define FOR_EACH_TYPE_EXTRA(M) M(kNumberTypeBFloat16, bfloat16)
#else
#define FOR_EACH_TYPE_EXTRA(M)
#endif

#define FOR_EACH_TYPE(M) \
  FOR_EACH_TYPE_BASE(M)  \
  FOR_EACH_TYPE_EXTRA(M)

#define REGISTER_SIZE(address_type_id, address_type) { address_type_id, sizeof(address_type) },

static const std::unordered_map<TypeId, size_t> kTypeSizeMap = {
  FOR_EACH_TYPE(REGISTER_SIZE)
};

size_t GetTypeSize(TypeId tid) {
  return kTypeSizeMap.at(tid);
}

template <typename T>
using DstCopyFunc = void (*)(T *src_ptr, void *dst_ptr, size_t size);

template <typename T>
static const std::unordered_map<TypeId, DstCopyFunc<T>> g_dst_copy_map = {
#define REGISTER_DST(dst_type_id, dst_type)                     \
  {dst_type_id, +[](T *src_ptr, void *dst_ptr, size_t size) {   \
    auto buf = static_cast<dst_type *>(dst_ptr);                \
    return tensor::TransDataType<dst_type>(src_ptr, buf, size); \
    }},
  FOR_EACH_TYPE(REGISTER_DST)
#undef REGISTER_DST
};

template <typename T>
void CopyData(T *src_ptr, size_t size, void *dst_ptr, TypeId dst_type_id) {
  auto &m = g_dst_copy_map<T>;
  auto it = m.find(dst_type_id);
  if (it == m.end()) {
    MS_LOG(EXCEPTION) << "Cannot construct Tensor because of unsupported dst data type: " << dst_type_id << ".";
  }
  it->second(src_ptr, dst_ptr, size);
}

using SrcCopyFunc = std::function<void(void *src_ptr, void *dst_ptr, size_t size, TypeId dst_type_id)>;

static const std::unordered_map<TypeId, SrcCopyFunc> g_src_copy_map = {
#define REGISTER_SRC(src_type_id, src_type)                                          \
  {src_type_id, +[](void *src_ptr, void *dst_ptr, size_t size, TypeId dst_type_id) { \
    auto buf = static_cast<src_type *>(src_ptr);                                     \
    return CopyData<src_type>(buf, size, dst_ptr, dst_type_id);                      \
    }},
  FOR_EACH_TYPE(REGISTER_SRC)
#undef REGISTER_SRC
};

#undef FOR_EACH_TYPE
#undef FOR_EACH_TYPE_BASE
#undef FOR_EACH_TYPE_EXTRA
#undef REGISTER_SIZE
// clang-format on

void CopyData(const DeviceAddressPtr &dst_device_address, const DeviceAddressPtr &src_device_address,
              TypeId dst_type_id, TypeId src_type_id) {
  MS_EXCEPTION_IF_NULL(src_device_address);
  MS_EXCEPTION_IF_NULL(dst_device_address);
  auto src_size = src_device_address->GetSize() / GetTypeSize(src_type_id);
  auto dst_size = dst_device_address->GetSize() / GetTypeSize(dst_type_id);
  if (src_size != dst_size) {
    MS_LOG(EXCEPTION) << "Not same shape in device address:" << src_device_address->ToString()
                      << " and:" << dst_device_address->ToString();
  }

  void *src_ptr = src_device_address->GetMutablePtr();
  void *dst_ptr = dst_device_address->GetMutablePtr();
  MS_EXCEPTION_IF_NULL(src_ptr);
  MS_EXCEPTION_IF_NULL(dst_ptr);

  auto it = g_src_copy_map.find(src_type_id);
  if (it == g_src_copy_map.end()) {
    MS_LOG(EXCEPTION) << "Unsupported conversion from " << src_type_id << " to " << dst_type_id;
  }
  it->second(src_ptr, dst_ptr, src_size, dst_type_id);
}

bool DoCopy(const DeviceAddressPtr &dst_device_address, const DeviceAddressPtr &src_device_address,
            const DeviceAddressExtPtr &src_ext) {
  if (src_device_address->GetSize() > dst_device_address->GetSize()) {
    MS_LOG(WARNING) << "Please check whether need sync data, src size: " << src_device_address->GetSize()
                    << ", dst size: " << dst_device_address->GetSize();
    return true;
  }
  auto ret_code = memcpy_s(dst_device_address->GetDevicePtr(), src_device_address->GetSize(),
                           src_device_address->GetDevicePtr(), src_device_address->GetSize());
  // Return ERANGE when the copy size is larger than SECUREC_MEM_MAX_LEN.
  if (ret_code == ERANGE) {
    MS_LOG(DEBUG) << "Copy for same type and return erange from device address:" << src_device_address->ToString()
                  << " to:" << dst_device_address->ToString();
    if (src_ext == nullptr) {
      MS_LOG(EXCEPTION)
        << "For large block memory copy on cpu, the input type id needs to be known, src device address:"
        << src_device_address->ToString() << " dst:" << dst_device_address->ToString();
    }
    device::ConvertSameType(dst_device_address->GetMutablePtr(), src_device_address->GetMutablePtr(),
                            dst_device_address->GetSize(), src_ext->dtype_id_);
    return true;
  } else if (ret_code != EOK) {
    MS_LOG(ERROR) << "Failed to copy tensor from device address:" << src_device_address
                  << " to :" << dst_device_address;
    return false;
  } else {
    return true;
  }
}
}  // namespace

bool HostCopy(const DeviceAddressPtr &dst_device_address, const DeviceAddressPtr &src_device_address,
              const DeviceAddressExtPtr &src_ext, const DeviceAddressExtPtr &dst_ext) {
  MS_EXCEPTION_IF_NULL(dst_device_address);
  MS_EXCEPTION_IF_NULL(src_device_address);
  if (dst_device_address->GetSize() == 0 || src_device_address->GetSize() == 0) {
    MS_LOG(INFO) << "No need sync for dst device address: " << dst_device_address
                 << " and src device address: " << src_device_address;
    return true;
  }

  auto dst_ptr = dst_device_address->GetMutablePtr();
  auto src_ptr = src_device_address->GetMutablePtr();
  MS_EXCEPTION_IF_NULL(src_device_address->GetMutablePtr());
  MS_EXCEPTION_IF_NULL(dst_device_address->GetMutablePtr());
  if (dst_ptr == src_ptr) {
    MS_LOG(DEBUG) << "host_ptr is equal to device ptr, request ignored.";
    return true;
  }
  if (src_ext == nullptr || dst_ext == nullptr) {
    return DoCopy(dst_device_address, src_device_address, src_ext);
  }
  if (src_ext->format_ != dst_ext->format_) {
    MS_LOG(ERROR) << "Format is different, src(format:" << src_ext->format_ << "), dst(format:" << dst_ext->format_
                  << ") for device address:" << dst_device_address;
    return false;
  }
  auto dst_type_id = dst_ext->dtype_id_;
  auto src_type_id = src_ext->dtype_id_;
  if (src_type_id == dst_type_id) {
    return DoCopy(dst_device_address, src_device_address, src_ext);
  }

  if (dst_type_id == kNumberTypeFloat16 && src_type_id == kNumberTypeFloat32) {
    device::FloatToHalf(dst_ptr, src_ptr, dst_device_address->GetSize() >> 1);
  } else if (dst_type_id == kNumberTypeFloat64 && src_type_id == kNumberTypeFloat32) {
    device::FloatToDouble(dst_ptr, src_ptr, dst_device_address->GetSize() / sizeof(double));
  } else if (dst_type_id == kNumberTypeFloat32 && src_type_id == kNumberTypeFloat64) {
    device::DoubleToFloat(dst_ptr, src_ptr, dst_device_address->GetSize() >> 2);
  } else if (dst_type_id == kNumberTypeInt16 && src_type_id == kNumberTypeInt32) {
    device::IntToShort(dst_ptr, src_ptr, dst_device_address->GetSize() >> 1);
  } else if (dst_type_id == kNumberTypeInt64 && src_type_id == kNumberTypeInt32) {
    device::IntToLong(dst_ptr, src_ptr, dst_device_address->GetSize() / sizeof(int64_t));
  } else {
    MS_LOG(DEBUG) << "Types not match. src type: " << TypeIdLabel(src_type_id)
                  << ", dst type: " << TypeIdLabel(dst_type_id) << " device_address:" << dst_device_address << " !";
    CopyData(dst_device_address, src_device_address, dst_type_id, src_type_id);
  }
  return true;
}
}  // namespace mindspore

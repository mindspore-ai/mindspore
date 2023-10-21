/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/hal/device/ascend_kernel_task.h"
#include <functional>
#include <string>
#include "transform/graph_ir/op_adapter_map.h"
#include "plugin/device/ascend/kernel/acl/acl_kernel_mod.h"
#include "plugin/device/ascend/hal/device/ascend_launch_transdata.h"
#include "runtime/device/ms_device_shape_transfer.h"
#include "ops/op_name.h"
#include "ops/view/view_strides_calculator.h"

namespace mindspore::device::ascend {
namespace {
constexpr auto kStorageOffset = "storage_offset";
constexpr auto kDstSize = "dst_size";
constexpr auto kDstStride = "dst_stride";
constexpr auto kSrcSize = "src_size";
constexpr auto kSrcStride = "src_stride";
constexpr auto kDstStorageOffset = "dst_storage_offset";
constexpr auto kSrcStorageOffset = "src_storage_offset";
constexpr size_t kAsStridedSupportMinSize = 32;
}  // namespace
std::vector<int64_t> GetContiguousStrides(const std::vector<int64_t> &shape) {
  if (shape.empty()) {
    return {};
  }

  std::vector<int64_t> ret(shape.size(), 1);
  int64_t strides = 1;
  for (size_t i = shape.size() - 1; i > 0; --i) {
    strides *= shape[i];
    ret[i - 1] = strides;
  }
  return ret;
}

struct AddressAndStorageInfo {
  AddressAndStorageInfo(const DeviceAddressPtr &address, const TensorStorageInfoPtr &storage_info)
      : addr(address), storage(storage_info) {}
  DeviceAddressPtr addr;
  TensorStorageInfoPtr storage;
  bool is_contiguous() { return (storage == nullptr || storage->is_contiguous); }
  size_t storage_offset() { return (storage == nullptr ? 0 : storage->storage_offset); }
  size_t GetSize() {
    MS_EXCEPTION_IF_NULL(addr);
    if (storage == nullptr) {
      return addr->GetSize();
    }

    const auto &shape = storage->shape;
    auto shape_size = LongToSize(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>()));
    return shape_size * GetTypeByte(TypeIdToType(addr->type_id()));
  }

  std::string GetStorageInfoStr() { return (storage == nullptr ? "" : storage->ToString()); }
};
using AddressAndStorageInfoPtr = std::shared_ptr<AddressAndStorageInfo>;

kernel::AddressPtr MallocMemoryForDeviceAddress(const device::DeviceAddressPtr &device_address,
                                                const device::DeviceContext *device_context) {
  if (!device_address) {
    return std::make_shared<kernel::Address>();
  }
  if (device_address->GetPtr() == nullptr) {
    if (!device_context->device_res_manager_->AllocateMemory(device_address.get())) {
      MS_LOG(EXCEPTION) << "Allocate device memory failed!";
    }
  }

  return std::make_shared<kernel::Address>(device_address->GetMutablePtr(), device_address->GetSize());
}

kernel::AddressPtr MallocMemoryForDeviceAddressWithOffset(const AddressAndStorageInfoPtr &addr_info,
                                                          const device::DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(addr_info);
  if (!addr_info->addr) {
    return std::make_shared<kernel::Address>();
  }
  auto device_address = addr_info->addr;
  if (device_address->GetPtr() == nullptr) {
    if (!device_context->device_res_manager_->AllocateMemory(device_address.get())) {
      MS_LOG(EXCEPTION) << "Allocate device memory failed!";
    }
  }

  if (addr_info->storage == nullptr) {
    return std::make_shared<kernel::Address>(device_address->GetMutablePtr(), device_address->GetSize());
  }

  auto offset = addr_info->storage_offset() * GetTypeByte(TypeIdToType(device_address->type_id()));
  if (offset > device_address->GetSize()) {
    MS_LOG(EXCEPTION) << "offset is out of bounds, offset:" << offset << " device size:" << device_address->GetSize();
  }
  return std::make_shared<kernel::Address>(static_cast<uint8_t *>(device_address->GetMutablePtr()) + offset,
                                           device_address->GetSize() - offset);
}

bool IdentityFunc(const DeviceAddressPtr &input_address, const TensorStorageInfoPtr &input_storage_info,
                  const DeviceAddressPtr &output_address, const device::DeviceContext *device_context,
                  void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(input_storage_info);
  MS_EXCEPTION_IF_NULL(input_address);
  MS_EXCEPTION_IF_NULL(output_address);

  auto input = MallocMemoryForDeviceAddress(input_address, device_context);
  auto output = MallocMemoryForDeviceAddress(output_address, device_context);

  auto prim = std::make_shared<Primitive>(transform::kNameIdentity);
  std::vector<std::string> input_device_formats = {input_address->format()};
  auto output_device_formats = {output_address->format()};
  auto input_device_types = {input_address->type_id()};
  auto output_device_types = {output_address->type_id()};
  kernel::AclKernelModPtr identity_kernel = std::make_shared<kernel::AclKernelMod>();

  auto node_idx = input_address->GetNodeIndex();
  int64_t groups = 1;
  if (node_idx.first != nullptr) {
    groups = common::AnfAlgo::GetAttrGroups(node_idx.first, node_idx.second);
  }
  prim->set_attr(kAttrFracZGroup, MakeValue(static_cast<int64_t>(groups)));
  identity_kernel->SetPrimitive(prim);
  identity_kernel->CreateAclConverter();
  identity_kernel->SetDeviceInfo(input_device_formats, output_device_formats, input_device_types, output_device_types);
  identity_kernel->PackageInput(0, "", &input_storage_info->ori_shape);
  identity_kernel->PackageOutput(0, input_storage_info->ori_shape);

  MS_LOG(DEBUG) << "Begin launch kernel: " << prim->name();
  auto ret = identity_kernel->Launch({input}, std::vector<AddressPtr>{}, {output}, stream_ptr);
  MS_LOG(DEBUG) << "End launch kernel: " << prim->name();
  if (!ret) {
    MS_LOG(ERROR) << "Launch kernel failed, kernel full name: " << prim->name();
  }
  return ret;
}

bool AsStridedFunc(const AddressAndStorageInfoPtr &src_addr_info, const AddressAndStorageInfoPtr &dst_addr_info,
                   const device::DeviceContext *device_context, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(src_addr_info);
  MS_EXCEPTION_IF_NULL(dst_addr_info);
  MS_EXCEPTION_IF_NULL(src_addr_info->addr);
  MS_EXCEPTION_IF_NULL(src_addr_info->storage);
  MS_EXCEPTION_IF_NULL(dst_addr_info->addr);
  MS_LOG(DEBUG) << "Start";

  auto input = MallocMemoryForDeviceAddressWithOffset(src_addr_info, device_context);
  auto output = MallocMemoryForDeviceAddressWithOffset(dst_addr_info, device_context);

  auto prim = std::make_shared<Primitive>(transform::kNameAsStrided);
  prim->set_attr(ops::kSize, MakeValue(src_addr_info->storage->shape));
  prim->set_attr(ops::kStride, MakeValue(src_addr_info->storage->strides));
  int64_t offset = 0;
  prim->set_attr(kStorageOffset, MakeValue(offset));

  std::vector<std::string> input_device_formats = {src_addr_info->addr->format(), kOpFormat_DEFAULT, kOpFormat_DEFAULT,
                                                   kOpFormat_DEFAULT};
  auto output_device_formats = {dst_addr_info->addr->format()};
  auto input_device_types = {src_addr_info->addr->type_id(), kNumberTypeInt64, kNumberTypeInt64, kNumberTypeInt64};
  auto output_device_types = {dst_addr_info->addr->type_id()};
  kernel::AclKernelModPtr as_strided_kernel = std::make_shared<kernel::AclKernelMod>();
  as_strided_kernel->SetPrimitive(prim);
  as_strided_kernel->CreateAclConverter();
  as_strided_kernel->SetDeviceInfo(input_device_formats, output_device_formats, input_device_types,
                                   output_device_types);
  ShapeVector size_shape = {static_cast<int64_t>(src_addr_info->storage->shape.size())};
  ShapeVector strides_shape = {static_cast<int64_t>(src_addr_info->storage->strides.size())};
  ShapeVector storage_offset_shape = {1};

  as_strided_kernel->PackageInput(kIndex0, "", &src_addr_info->storage->ori_shape);
  as_strided_kernel->PackageInput(kIndex1, kOpFormat_DEFAULT, &size_shape);
  as_strided_kernel->PackageInput(kIndex2, kOpFormat_DEFAULT, &strides_shape);
  as_strided_kernel->PackageInput(kIndex3, kOpFormat_DEFAULT, &storage_offset_shape);
  as_strided_kernel->PackageOutput(kIndex0, src_addr_info->storage->shape);

  MS_LOG(DEBUG) << "Begin launch kernel: " << prim->name();
  auto ret = as_strided_kernel->Launch({input}, std::vector<AddressPtr>{}, {output}, stream_ptr);
  MS_LOG(DEBUG) << "End launch kernel: " << prim->name();
  if (!ret) {
    MS_LOG(ERROR) << "Launch kernel failed, kernel full name: " << prim->name();
  }
  MS_LOG(DEBUG) << "End";

  return ret;
}

void SetStridesAndShapeForViewCopy(const PrimitivePtr &prim, const AddressAndStorageInfoPtr &src_addr_info,
                                   const AddressAndStorageInfoPtr &dst_addr_info) {
  MS_EXCEPTION_IF_NULL(src_addr_info);
  MS_EXCEPTION_IF_NULL(dst_addr_info);
  MS_EXCEPTION_IF_NULL(prim);
  auto src_storage = src_addr_info->storage;

  if (dst_addr_info->storage != nullptr) {
    auto dst_storage = dst_addr_info->storage;
    prim->set_attr(kDstSize, MakeValue(dst_storage->shape));
    prim->set_attr(kDstStride, MakeValue(dst_storage->strides));
  } else if (src_addr_info->storage != nullptr) {
    // Dst might be null when copy with slice
    prim->set_attr(kDstSize, MakeValue(src_storage->shape));
    prim->set_attr(kDstStride, MakeValue(GetContiguousStrides(src_storage->shape)));
  }
  int64_t storage_offset = 0;
  prim->set_attr(kDstStorageOffset, MakeValue(static_cast<int64_t>(storage_offset)));
  prim->set_attr(kSrcStorageOffset, MakeValue(static_cast<int64_t>(storage_offset)));
  prim->set_attr(kSrcSize, MakeValue(src_storage->shape));
  prim->set_attr(kSrcStride, MakeValue(src_storage->strides));
}

bool ViewCopyFunc(const AddressAndStorageInfoPtr &src_addr_info, const AddressAndStorageInfoPtr &dst_addr_info,
                  const device::DeviceContext *device_context, void *stream_ptr) {
  MS_LOG(DEBUG) << "Start";
  MS_EXCEPTION_IF_NULL(src_addr_info);
  MS_EXCEPTION_IF_NULL(dst_addr_info);

  auto src_storage = src_addr_info->storage;
  MS_EXCEPTION_IF_NULL(src_storage);

  auto input = MallocMemoryForDeviceAddressWithOffset(src_addr_info, device_context);
  auto output = MallocMemoryForDeviceAddressWithOffset(dst_addr_info, device_context);
  if (dst_addr_info->GetSize() == 0) {
    MS_LOG(DEBUG) << "dst_addr_info is 0";
    return true;
  }

  auto prim = std::make_shared<Primitive>(transform::kNameViewCopy);
  SetStridesAndShapeForViewCopy(prim, src_addr_info, dst_addr_info);

  MS_EXCEPTION_IF_NULL(src_addr_info->addr);
  MS_EXCEPTION_IF_NULL(dst_addr_info->addr);

  std::vector<std::string> input_device_formats = {dst_addr_info->addr->format(), src_addr_info->addr->format()};
  auto output_device_formats = {dst_addr_info->addr->format()};
  auto input_device_types = {dst_addr_info->addr->type_id(), src_addr_info->addr->type_id()};
  auto output_device_types = {dst_addr_info->addr->type_id()};
  kernel::AclKernelModPtr view_copy_kernel = std::make_shared<kernel::AclKernelMod>();
  view_copy_kernel->SetPrimitive(prim);
  view_copy_kernel->CreateAclConverter();
  view_copy_kernel->SetDeviceInfo(input_device_formats, output_device_formats, input_device_types, output_device_types);
  auto dst_shape = (dst_addr_info->storage == nullptr ? src_storage->shape : dst_addr_info->storage->ori_shape);
  view_copy_kernel->PackageInput(0, "", &dst_shape);
  view_copy_kernel->PackageInput(1, "", &src_storage->ori_shape);
  view_copy_kernel->PackageOutput(0, dst_shape);

  MS_LOG(DEBUG) << "Begin launch kernel: " << prim->name();
  auto ret = view_copy_kernel->Launch({output, input}, std::vector<AddressPtr>{}, {output}, stream_ptr);
  MS_LOG(DEBUG) << "End launch kernel: " << prim->name();
  if (!ret) {
    MS_LOG(ERROR) << "Launch kernel failed, kernel full name: " << prim->name();
  }
  MS_LOG(DEBUG) << "End";

  return ret;
}

bool LaunchAsyncCopy(const AddressAndStorageInfoPtr &src_addr_info, const AddressAndStorageInfoPtr &dst_addr_info,
                     const size_t &copy_size, const device::DeviceContext *device_context, void *stream_ptr) {
  MS_LOG(DEBUG) << "Start";
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(src_addr_info);
  MS_EXCEPTION_IF_NULL(dst_addr_info);
  auto src_addr = MallocMemoryForDeviceAddress(src_addr_info->addr, device_context);
  auto dst_addr = MallocMemoryForDeviceAddress(dst_addr_info->addr, device_context);

  if (copy_size == 0) {
    MS_LOG(DEBUG) << "copy_size is zero";
    return true;
  }

  MS_EXCEPTION_IF_NULL(src_addr_info->addr);
  auto type_size = GetTypeByte(TypeIdToType(src_addr_info->addr->type_id()));

  auto src_offset = src_addr_info->storage_offset() * type_size;
  auto dst_offset = dst_addr_info->storage_offset() * type_size;

  if (copy_size + src_offset > src_addr->size) {
    MS_LOG(EXCEPTION) << "Src copy out of bounds, copy_size:" << copy_size << ", src_offset:" << src_offset
                      << ", src_addr->size:" << src_addr->size;
  }

  if (copy_size + dst_offset > dst_addr->size) {
    MS_LOG(EXCEPTION) << "Dst copy out of bounds, copy_size:" << copy_size << ", src_offset:" << dst_offset
                      << ", dst_addr->size:" << dst_addr->size;
  }

  aclError status = aclrtMemcpyAsync((static_cast<uint8_t *>(dst_addr->addr) + dst_offset), copy_size,
                                     (static_cast<uint8_t *>(src_addr->addr) + src_offset), copy_size,
                                     ACL_MEMCPY_DEVICE_TO_DEVICE, stream_ptr);
  if (status != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "MemCpyAsync op aclrtMemcpyAsync failed, ret:" << status << " destMax:" << dst_addr->size
                  << " count:" << copy_size;
    return false;
  }
  MS_LOG(DEBUG) << "End";

  return true;
}

bool LaunchTransData(const DeviceAddressPtr &input_address, const TensorStorageInfoPtr &input_storage_info,
                     const DeviceAddressPtr &output_address, const device::DeviceContext *device_context,
                     void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(input_storage_info);
  MS_EXCEPTION_IF_NULL(input_address);
  MS_EXCEPTION_IF_NULL(output_address);
  MS_LOG(DEBUG) << "LaunchTransData begin";

  int64_t groups = 1;
  if (input_address->format() == kOpFormat_FRAC_Z) {
    auto node_idx = input_address->GetNodeIndex();
    if (node_idx.first != nullptr) {
      groups = common::AnfAlgo::GetAttrGroups(node_idx.first, node_idx.second);
    }
  }

  auto src_addr = MallocMemoryForDeviceAddress(input_address, device_context);
  auto dst_addr = MallocMemoryForDeviceAddress(output_address, device_context);

  auto prim = std::make_shared<Primitive>(transform::kNameTransData);
  prim->set_attr(ops::kSrcFormat, MakeValue(input_address->format()));
  prim->set_attr(ops::kDstFormat, MakeValue(output_address->format()));
  prim->set_attr(ops::kGroups, MakeValue(groups));

  auto input_device_formats = {input_address->format()};
  auto output_device_formats = {output_address->format()};
  auto input_device_types = {input_address->type_id()};
  auto output_device_types = {output_address->type_id()};
  kernel::AclKernelModPtr transdata_kernel = std::make_shared<kernel::AclKernelMod>();
  transdata_kernel->SetPrimitive(prim);
  transdata_kernel->CreateAclConverter();
  transdata_kernel->SetDeviceInfo(input_device_formats, output_device_formats, input_device_types, output_device_types);

  transdata_kernel->PackageInput(0, "", &input_storage_info->ori_shape);
  transdata_kernel->PackageOutput(0, input_storage_info->ori_shape);

  MS_LOG(DEBUG) << "Begin launch kernel: " << prim->name();
  auto ret = transdata_kernel->Launch({src_addr}, std::vector<AddressPtr>{}, {dst_addr}, stream_ptr);
  MS_LOG(DEBUG) << "End launch kernel: " << prim->name();
  if (!ret) {
    MS_LOG(ERROR) << "Launch kernel failed, kernel full name: " << prim->name();
  }
  MS_LOG(DEBUG) << "End";

  return ret;
}

bool CopyBaseFormatDataDeviceToDevice(const AddressAndStorageInfoPtr &src_addr_info,
                                      const AddressAndStorageInfoPtr &dst_addr_info,
                                      const device::DeviceContext *device_context, void *stream_ptr);

bool ContiguousViewCopySrcAddr(const AddressAndStorageInfoPtr &src_addr_info,
                               const device::DeviceContext *device_context, void *stream_ptr) {
  MS_LOG(DEBUG) << "Start";
  MS_EXCEPTION_IF_NULL(src_addr_info);
  MS_EXCEPTION_IF_NULL(src_addr_info->storage);
  MS_EXCEPTION_IF_NULL(src_addr_info->addr);

  const auto &dst_shape = src_addr_info->storage->shape;
  auto tensor_size = SizeOf(dst_shape) * GetTypeByte(TypeIdToType(src_addr_info->addr->type_id()));
  auto dst_addr = device_context->device_res_manager_->CreateDeviceAddress(nullptr, tensor_size, kOpFormat_DEFAULT,
                                                                           src_addr_info->addr->type_id(), dst_shape);
  dst_addr->set_device_shape(dst_shape);

  auto dst_addr_info = std::make_shared<AddressAndStorageInfo>(dst_addr, nullptr);
  auto ret = CopyBaseFormatDataDeviceToDevice(src_addr_info, dst_addr_info, device_context, stream_ptr);
  if (!ret) {
    MS_LOG(ERROR) << "CopyBaseFormatDataDeviceToDevice failed.";
    return ret;
  }

  auto get_ori_stride = [](const std::vector<int64_t> &shape) -> std::vector<int64_t> {
    if (shape.empty()) {
      return {};
    }

    std::vector<int64_t> ret{1};
    int64_t strides = 1;
    for (size_t i = shape.size() - 1; i > 0; i--) {
      strides *= shape[i];
      (void)ret.emplace(ret.begin(), strides);
    }
    return ret;
  };

  // Refresh contiguous address for src
  src_addr_info->addr = dst_addr;
  const auto &dst_strides = get_ori_stride(dst_shape);
  src_addr_info->storage = std::make_shared<TensorStorageInfo>(dst_shape, dst_strides, 0, dst_shape, dst_strides, true);

  MS_LOG(DEBUG) << "End";
  return true;
}

bool CopyBaseFormatDataDeviceToDevice(const AddressAndStorageInfoPtr &src_addr_info,
                                      const AddressAndStorageInfoPtr &dst_addr_info,
                                      const device::DeviceContext *device_context, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(src_addr_info);
  MS_EXCEPTION_IF_NULL(dst_addr_info);
  MS_EXCEPTION_IF_NULL(src_addr_info->addr);
  MS_EXCEPTION_IF_NULL(dst_addr_info->addr);

  auto src_size = src_addr_info->GetSize();
  auto dst_size = dst_addr_info->GetSize();

  MS_LOG(DEBUG) << "Src storage info:" << src_addr_info->GetStorageInfoStr()
                << ", dst storage info:" << dst_addr_info->GetStorageInfoStr() << ", input addr size:" << src_size
                << ", output addr size:" << dst_size;
  if (!dst_addr_info->is_contiguous()) {
    if (!src_addr_info->is_contiguous()) {
      if (!ContiguousViewCopySrcAddr(src_addr_info, device_context, stream_ptr)) {
        MS_LOG(ERROR) << "ContiguousViewCopySrcAddr failed.";
        return false;
      }
    }

    auto ret = ViewCopyFunc(src_addr_info, dst_addr_info, device_context, stream_ptr);
    if (!ret) {
      MS_LOG(ERROR) << "ViewCopyFunc failed.";
    }
    return true;
  }

  auto ret = true;
  if (!src_addr_info->is_contiguous()) {
    // AsStrided Op support size >= 32(block), smaller than this will cause accuracy issues.
    if (dst_size < kAsStridedSupportMinSize) {
      ret = ViewCopyFunc(src_addr_info, dst_addr_info, device_context, stream_ptr);
      if (!ret) {
        MS_LOG(ERROR) << "ViewCopyFunc failed.";
      }
    } else {
      ret = AsStridedFunc(src_addr_info, dst_addr_info, device_context, stream_ptr);
      if (!ret) {
        MS_LOG(ERROR) << "AsStridedFunc failed.";
      }
    }
  } else if (src_size == dst_size) {
    ret = LaunchAsyncCopy(src_addr_info, dst_addr_info, src_size, device_context, stream_ptr);
    if (!ret) {
      MS_LOG(ERROR) << "LaunchAsyncCopy failed.";
    }
  } else {
    ret = ViewCopyFunc(src_addr_info, dst_addr_info, device_context, stream_ptr);
    if (!ret) {
      MS_LOG(ERROR) << "ViewCopyFunc failed.";
    }
  }

  return ret;
}

DeviceAddressPtr ConvertAddrToBaseFormat(const DeviceAddressPtr &input_address,
                                         const TensorStorageInfoPtr &input_storage_info,
                                         const device::DeviceContext *device_context, void *stream_ptr) {
  auto baseformat_addr = input_address;

  const auto &ori_format = input_address->format();
  const auto &base_format = trans::FormatHelper::GetInstance().GetBaseFormat(ori_format);
  MS_LOG(DEBUG) << "Input format:" << ori_format << " base_format:" << base_format;

  if (base_format.empty()) {
    MS_LOG(DEBUG) << "Base format is empty, need to transdata first.";
    auto tensor_size = SizeOf(input_storage_info->ori_shape) * GetTypeByte(TypeIdToType(input_address->type_id()));
    baseformat_addr = device_context->device_res_manager_->CreateDeviceAddress(
      nullptr, tensor_size, kOpFormat_NCHW, input_address->type_id(), input_storage_info->ori_shape);
    baseformat_addr->set_device_shape(input_storage_info->ori_shape);
    auto ret = LaunchTransData(input_address, input_storage_info, baseformat_addr, device_context, stream_ptr);
    if (!ret) {
      MS_LOG(EXCEPTION) << "LaunchTransData failed.";
    }
  } else if (ori_format != base_format) {
    MS_LOG(DEBUG) << "Input format is not base format, need convert first.";

    const auto &device_shape =
      trans::TransShapeToDevice(input_storage_info->ori_shape, base_format, input_address->type_id());
    auto tensor_size = SizeOf(device_shape) * GetTypeByte(TypeIdToType(input_address->type_id()));
    baseformat_addr = device_context->device_res_manager_->CreateDeviceAddress(nullptr, tensor_size, base_format,
                                                                               input_address->type_id(), device_shape);
    baseformat_addr->set_device_shape(device_shape);
    auto ret = IdentityFunc(input_address, input_storage_info, baseformat_addr, device_context, stream_ptr);
    if (!ret) {
      MS_LOG(EXCEPTION) << "IdentityFunc failed.";
    }
  }
  return baseformat_addr;
}

bool CopyDataDeviceToDevice(const AddressAndStorageInfoPtr &src_addr_info,
                            const AddressAndStorageInfoPtr &dst_addr_info, const device::DeviceContext *device_context,
                            void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(src_addr_info);
  MS_EXCEPTION_IF_NULL(src_addr_info);

  src_addr_info->addr =
    ConvertAddrToBaseFormat(src_addr_info->addr, src_addr_info->storage, device_context, stream_ptr);
  dst_addr_info->addr =
    ConvertAddrToBaseFormat(dst_addr_info->addr, dst_addr_info->storage, device_context, stream_ptr);

  auto ret = CopyBaseFormatDataDeviceToDevice(src_addr_info, dst_addr_info, device_context, stream_ptr);
  if (!ret) {
    MS_LOG(ERROR) << "CopyBaseFormatDataDeviceToDevice failed.";
  }
  return ret;
}

void RefreshFormat(const DeviceAddressPtr &output_address) {
  if (output_address->device_shape().size() == kSizeFour && output_address->format() == kOpFormat_ND) {
    output_address->set_format(kOpFormat_NCHW);
  } else if (output_address->device_shape().size() != kSizeFour && output_address->format() == kOpFormat_NCHW) {
    output_address->set_format(kOpFormat_ND);
  }
}

bool AscendContiguousKernelTask::RunWithRet() {
  auto device_context = context_->device_context();
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
  device_context->device_res_manager_->BindDeviceToCurrentThread(false);

  auto input_addr_info =
    std::make_shared<AddressAndStorageInfo>(context_->GetInputAddr(0), context_->GetInputStorage(0));
  auto output_addr_info = std::make_shared<AddressAndStorageInfo>(context_->GetOutputAddr(0), nullptr);
  MS_EXCEPTION_IF_NULL(input_addr_info);
  MS_EXCEPTION_IF_NULL(output_addr_info);

  auto stream_ptr = context_->stream();
  MS_EXCEPTION_IF_NULL(stream_ptr);

  if (!input_addr_info->is_contiguous()) {
    auto ret = CopyDataDeviceToDevice(input_addr_info, output_addr_info, device_context, stream_ptr);
    if (!ret) {
      MS_LOG(ERROR) << "CopyDataDeviceToDevice failed.";
      return false;
    }
    return true;
  }

  MS_EXCEPTION_IF_NULL(input_addr_info->addr);
  MS_EXCEPTION_IF_NULL(input_addr_info->storage);
  MS_EXCEPTION_IF_NULL(output_addr_info->addr);
  if (input_addr_info->storage->shape != input_addr_info->storage->ori_shape ||
      input_addr_info->storage->strides != input_addr_info->storage->ori_strides) {
    if (input_addr_info->addr->GetSize() == output_addr_info->addr->GetSize() &&
        !trans::FormatHelper::GetInstance().IsBaseFormatType(input_addr_info->addr->format())) {
      input_addr_info->addr =
        ConvertAddrToBaseFormat(input_addr_info->addr, input_addr_info->storage, device_context, stream_ptr);

      auto ret = IdentityFunc(input_addr_info->addr, input_addr_info->storage, output_addr_info->addr, device_context,
                              stream_ptr);
      if (!ret) {
        MS_LOG(ERROR) << "IdentityFunc failed.";
        return false;
      }
      RefreshFormat(output_addr_info->addr);
    } else {
      auto ret = CopyDataDeviceToDevice(input_addr_info, output_addr_info, device_context, stream_ptr);
      if (!ret) {
        MS_LOG(ERROR) << "CopyDataDeviceToDevice failed.";
        return false;
      }
      RefreshFormat(output_addr_info->addr);
    }
    return true;
  }

  if (trans::FormatHelper::GetInstance().IsPadded(input_addr_info->addr->format()) &&
      input_addr_info->storage_offset() > 0) {
    auto ret = CopyDataDeviceToDevice(input_addr_info, output_addr_info, device_context, stream_ptr);
    if (!ret) {
      MS_LOG(ERROR) << "CopyDataDeviceToDevice failed.";
    }
    RefreshFormat(output_addr_info->addr);
    return ret;
  }

  auto ret = LaunchAsyncCopy(input_addr_info, output_addr_info, input_addr_info->GetSize(), device_context, stream_ptr);
  if (!ret) {
    MS_LOG(ERROR) << "LaunchAsyncCopy failed.";
  }
  return ret;
}

bool AscendCopyWithSliceKernelTask::RunWithRet() {
  MS_LOG(DEBUG) << "Start";
  auto device_context = context_->device_context();
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
  device_context->device_res_manager_->BindDeviceToCurrentThread(false);

  auto dst_addr_info = std::make_shared<AddressAndStorageInfo>(context_->GetInputAddr(0), context_->GetInputStorage(0));
  auto src_addr_info = std::make_shared<AddressAndStorageInfo>(context_->GetInputAddr(1), context_->GetInputStorage(1));

  auto stream = context_->stream();
  MS_EXCEPTION_IF_NULL(stream);

  auto ret = CopyDataDeviceToDevice(src_addr_info, dst_addr_info, device_context, stream);
  if (!ret) {
    MS_LOG(ERROR) << "CopyDataDeviceToDevice failed.";
  }
  MS_LOG(DEBUG) << "End";

  return true;
}
}  // namespace mindspore::device::ascend

/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/hal/device/ascend_device_address.h"
#include <memory>
#include <vector>
#include <unordered_map>
#include <utility>
#include <set>
#include <algorithm>
#include "runtime/mem.h"
#include "acl/acl_rt.h"
#include "runtime/device/kernel_runtime_manager.h"
#include "runtime/device/kernel_runtime.h"
#include "runtime/device/memory_manager.h"
#include "runtime/device/convert_tensor_utils.h"
#include "plugin/device/ascend/hal/device/ascend_launch_transdata.h"
#include "runtime/hardware/device_context_manager.h"
#include "plugin/device/ascend/hal/hardware/ascend_device_context.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "ir/dtype/type.h"
#include "ir/tensor.h"
#include "abstract/utils.h"
#include "include/common/utils/utils.h"
#include "runtime/device/ms_device_shape_transfer.h"
#ifndef ENABLE_SECURITY
#include "debug/data_dump/dump_json_parser.h"
#endif
#ifdef ENABLE_DEBUGGER
#include "debug/tensor_load.h"
#endif

namespace mindspore {
namespace device {
namespace ascend {
const auto kFloat16Bytes = 2;
const auto kFloatBytes = sizeof(float);
const auto kFloat64Bytes = 8;

bool IsUseTransDataTypeFormat(const std::pair<std::string, std::string> &type_format) {
  static const std::set<std::pair<std::string, std::string>> use_trans_data = {
    std::make_pair("float16", mindspore::kOpFormat_NC1HWC0), std::make_pair("float32", mindspore::kOpFormat_NC1HWC0),
    std::make_pair("bool", mindspore::kOpFormat_NC1HWC0),    std::make_pair("float32", mindspore::kOpFormat_FRAC_Z),
    std::make_pair("float16", mindspore::kOpFormat_FRAC_Z),  std::make_pair("float16", mindspore::kOpFormat_FRAC_NZ),
    std::make_pair("float32", mindspore::kOpFormat_FRAC_NZ), std::make_pair("int32", mindspore::kOpFormat_FRAC_NZ),
    std::make_pair("float16", mindspore::kOpFormat_NHWC),    std::make_pair("float32", mindspore::kOpFormat_NHWC),
    std::make_pair("int8", mindspore::kOpFormat_NHWC),       std::make_pair("int16", mindspore::kOpFormat_NHWC),
    std::make_pair("int32", mindspore::kOpFormat_NHWC),      std::make_pair("int64", mindspore::kOpFormat_NHWC),
    std::make_pair("uint8", mindspore::kOpFormat_NHWC),      std::make_pair("uint16", mindspore::kOpFormat_NHWC),
    std::make_pair("uint32", mindspore::kOpFormat_NHWC),     std::make_pair("uint64", mindspore::kOpFormat_NHWC),
    std::make_pair("float16", mindspore::kOpFormat_HWCN),    std::make_pair("float32", mindspore::kOpFormat_HWCN),
    std::make_pair("int8", mindspore::kOpFormat_HWCN),       std::make_pair("int16", mindspore::kOpFormat_HWCN),
    std::make_pair("int32", mindspore::kOpFormat_HWCN),      std::make_pair("int64", mindspore::kOpFormat_HWCN),
    std::make_pair("uint8", mindspore::kOpFormat_HWCN),      std::make_pair("uint16", mindspore::kOpFormat_HWCN),
    std::make_pair("uint32", mindspore::kOpFormat_HWCN),     std::make_pair("uint64", mindspore::kOpFormat_HWCN)};
  return use_trans_data.find(type_format) != use_trans_data.end();
}

bool IsOpNeedTransFormat(const std::string &format) {
  static const std::set<std::string> op_need_trans_format = {
    kOpFormat_NHWC,    kOpFormat_HWCN,        kOpFormat_NC1HWC0,       kOpFormat_FRAC_Z,   kOpFormat_C1HWNCoC0,
    kOpFormat_FRAC_NZ, kOpFormat_NC1HWC0_C04, kOpFormat_FRACTAL_Z_C04, kOpFormat_NDC1HWC0, kOpFormat_FRACTAL_Z_3D};
  return op_need_trans_format.find(format) != op_need_trans_format.end();
}

void SyncMemory(void *dst, const void *src, uint64_t size, aclrtMemcpyKind kind) {
  if (size == 0) {
    return;
  }
  if (dst == nullptr) {
    MS_LOG(EXCEPTION) << "dst ptr is null, please check the address is set correctly.";
  }
  if (src == nullptr) {
    MS_LOG(EXCEPTION) << "src ptr is null, please check the address is set correctly.";
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  auto execution_mode = ms_context->get_param<int>(MS_CTX_EXECUTION_MODE);
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  runtime_instance->SetContext();

  // Only apply asynchronous copy in Pynative && ACL_MEMCPY_HOST_TO_DEVICE mode
  if (execution_mode != kPynativeMode || kind != ACL_MEMCPY_HOST_TO_DEVICE) {
    auto ret_rt_memcpy = aclrtMemcpy(dst, size, src, size, kind);
    if (ret_rt_memcpy != RT_ERROR_NONE) {
      MS_EXCEPTION(DeviceProcessError) << "aclrtMemcpy failed";
    }
  } else {
    auto ret = runtime_instance->MemcpyAsync(dst, src, size, static_cast<int32_t>(RT_MEMCPY_HOST_TO_DEVICE_EX));
    if (!ret) {
      MS_EXCEPTION(DeviceProcessError) << "MemcpyAsync failed";
    }
  }
}

bool FloatToHalfAndSyncHostToDevice(void *dst, size_t dst_size, const void *src, size_t src_size) {
  auto elem_num = src_size / kFloatBytes;
  if (elem_num != (dst_size / kFloat16Bytes)) {
    MS_EXCEPTION(ArgumentError) << "FloatToHalf failed. size not match src_size[" << src_size << "], dst_size["
                                << dst_size << "]";
  }
  std::vector<float16> half_data(elem_num);
  FloatToHalf(half_data.data(), src, elem_num);
  SyncMemory(dst, half_data.data(), dst_size, ACL_MEMCPY_HOST_TO_DEVICE);
  return true;
}

bool Float64ToFloatAndSyncHostToDevice(void *dst, size_t dst_size, const void *src, size_t src_size) {
  if (src_size / kFloat64Bytes != dst_size / kFloatBytes) {
    MS_EXCEPTION(ArgumentError) << "src_size[" << src_size << "], dst_size[" << dst_size << "]";
  }
  size_t elem_num = dst_size / sizeof(float);
  auto host_tmp = std::vector<float>(elem_num);
  DoubleToFloat(host_tmp.data(), src, elem_num);
  SyncMemory(dst, host_tmp.data(), dst_size, ACL_MEMCPY_HOST_TO_DEVICE);
  return true;
}

bool SyncDeviceToHostAndHalfToFloat(void *dst, size_t dst_size, const void *src, size_t src_size) {
  auto elem_num = src_size / kFloat16Bytes;
  if (elem_num != (dst_size / kFloatBytes)) {
    MS_EXCEPTION(ArgumentError) << "HalfToFloat failed. size not match src_size[" << src_size << "], dst_size["
                                << dst_size << "]";
  }
  std::vector<float16> half_data(elem_num);
  SyncMemory(half_data.data(), src, src_size, ACL_MEMCPY_DEVICE_TO_HOST);
  HalfToFloat(dst, half_data.data(), elem_num);
  return true;
}

bool SyncDeviceToHostAndFloatToFloat64(void *dst, size_t dst_size, const void *src, size_t src_size) {
  if (src_size / kFloatBytes != dst_size / kFloat64Bytes) {
    MS_EXCEPTION(ArgumentError) << "src_size[" << src_size << "], dst_size[" << dst_size << "]";
  }
  size_t elem_num = src_size / sizeof(float);
  auto host_tmp = std::vector<float>(elem_num);
  SyncMemory(host_tmp.data(), src, src_size, ACL_MEMCPY_DEVICE_TO_HOST);
  FloatToDouble(dst, host_tmp.data(), elem_num);
  return true;
}

void AscendDeviceAddress::BindDevice() const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (!MsContext::GetInstance()->get_param<bool>(MS_CTX_ENABLE_MINDRT)) {
    return;
  }

  // Bind device by device name and device id on the current thread.
  if (!device_name_.empty()) {
    auto ascend_device_context = GetDeviceContext();
    MS_EXCEPTION_IF_NULL(ascend_device_context);
    if (!ascend_device_context->device_res_manager_->BindDeviceToCurrentThread()) {
      MS_LOG(WARNING) << "Bind device to current thread failed.";
    }
  } else {
    MS_LOG(DEBUG) << "Device name is null.";
  }
}

void AscendDeviceAddress::SyncStream() const {
  MS_LOG(DEBUG) << "SyncStream Start!";
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  auto ret = runtime_instance->SyncStream();
  if (!ret) {
    MS_LOG(EXCEPTION) << "Sync stream error!";
  }
  MS_LOG(DEBUG) << "SyncStream Finish!";
}

bool AscendDeviceAddress::SyncDeviceToHost(size_t size, void *const host_ptr) const {
  MS_EXCEPTION_IF_NULL(host_ptr);
  std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
  BindDevice();
  SyncStream();
  CopyDeviceToHost(host_ptr, size);
  return true;
}

bool AscendDeviceAddress::SyncHostToDevice(size_t size, const void *host_ptr) const {
  MS_EXCEPTION_IF_NULL(host_ptr);
  std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
  BindDevice();
  CopyHostToDevice(host_ptr, size);
  return true;
}

bool AscendDeviceAddress::SyncDeviceToHost(const ShapeVector &shape, size_t size, mindspore::TypeId type,
                                           void *host_ptr) const {
  MS_LOG(DEBUG) << "SyncDeviceToHost, Device(format:" << format_ << ", type_id:" << TypeIdLabel(type_id_)
                << ", size:" << size_ << "), Host(type_id:" << TypeIdLabel(type) << ", size:" << size << ")";
  if (type_id_ > kMonadTypeBegin && type_id_ < kMonadTypeEnd) {
    return true;
  }
  BindDevice();
  SyncStream();
  bool sync_ok = false;
  ShapeVector host_shape = shape;
  if (host_shape.empty()) {
    (void)host_shape.emplace_back(1);
  }
  std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
  if (format_ == kOpFormat_NCHW || format_ == kOpFormat_DEFAULT || format_ == kOpFormat_NCDHW) {
    if (type_id_ == type) {
      CopyDeviceToHost(host_ptr, size);
      sync_ok = true;
    } else if (type_id_ == kNumberTypeFloat32 && type == kNumberTypeFloat64) {
      if (mem_offloaded()) {
        FloatToDouble(host_ptr, offload_ptr_, size_ / sizeof(float));
        sync_ok = true;
      } else {
        sync_ok = SyncDeviceToHostAndFloatToFloat64(host_ptr, size, ptr_, size_);
      }
    } else {
      auto shape_size = abstract::ShapeSize(host_shape);
      auto host = std::vector<uint8_t>(size_);
      CopyDeviceToHost(host.data(), size_);
      const trans::TypeIdArgs type_args{host.data(), shape_size, type_id_, type, size_};
      sync_ok = trans::TransDataType(type_args, host_ptr);
      if (!sync_ok) {
        MS_LOG(ERROR) << "Trans data type failed.";
        return false;
      }
    }
  } else {
    if (IsOpNeedTransFormat(format_)) {
      sync_ok = SyncDeviceToHostAndConvertFormat(shape, size, type, host_ptr);
    } else {
      MS_LOG(INFO) << "Can not find format transfer function for :" << format_;
    }
  }
  if (!sync_ok) {
    MS_LOG(ERROR) << "Unsupported to trans, dev_format:" << format_ << ", dev_type:" << TypeIdLabel(type_id_)
                  << ", host_type:" << TypeIdLabel(type);
    return false;
  }
  return sync_ok;
}

ShapeVector AscendDeviceAddress::GetDeviceShape(ShapeVector *host_shape) const {
  MS_EXCEPTION_IF_NULL(host_shape);
  ShapeVector device_shape;
  auto node_index = GetNodeIndex();
  if (format_ == kOpFormat_FRAC_NZ || format_ == kOpFormat_NCDHW) {
    device_shape = trans::TransShapeToDevice(*host_shape, format_, node_index.first, node_index.second, type_id_);
  } else {
    if (!host_shape_.empty()) {
      host_shape->clear();
      *host_shape = host_shape_;
    }
    *host_shape = trans::PaddingShape(*host_shape, format_);
    device_shape = trans::TransShapeToDevice(*host_shape, format_, node_index.first, node_index.second, type_id_);
  }
  return device_shape;
}

std::shared_ptr<LaunchKernel> AscendDeviceAddress::CreateLaunchTransData(const ShapeVector &host_shape,
                                                                         const std::string &ori_format,
                                                                         const std::string &dst_format) const {
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetCurrentKernelRuntime();
  MS_EXCEPTION_IF_NULL(runtime_instance);
  auto stream = runtime_instance->compute_stream();
  int64_t groups = 1;
  if (format_ == kOpFormat_FRAC_Z) {
    groups = GetGroupsWithCache();
  }
  auto launch_trans_data =
    std::make_shared<AscendLaunchTransData>(stream, type_id_, size_, ori_format, dst_format, host_shape, groups);
  MS_EXCEPTION_IF_NULL(launch_trans_data);
  return launch_trans_data;
}

bool AscendDeviceAddress::SyncDeviceToHostAndConvertFormatBasedOnTransData(const ShapeVector &host_shape, size_t size,
                                                                           mindspore::TypeId type,
                                                                           void *host_ptr) const {
  bool sync_ok = true;
  const std::string dst_format = kOpFormat_NCHW;
  if (launch_transdata_ == nullptr) {
    launch_transdata_ = CreateLaunchTransData(host_shape, format_, dst_format);
    MS_EXCEPTION_IF_NULL(launch_transdata_);
  }
  std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
  // launch transdata
  launch_transdata_->SetInputAddr(static_cast<uint8_t *>(ptr_));
  launch_transdata_->LaunchOpKernel();
  SyncStream();
  auto output_addr_vec = launch_transdata_->GetKernelOutputAddr();
  if (output_addr_vec.size() != 1) {
    launch_transdata_->FreeLaunchDeviceMem();
    MS_LOG(EXCEPTION) << "Launch transdata outputs should have only one output";
  }
  if (type_id_ == type) {
    SyncMemory(host_ptr, output_addr_vec[0], size, ACL_MEMCPY_DEVICE_TO_HOST);
  } else {
    auto host = std::vector<uint8_t>(size);
    SyncMemory(host.data(), output_addr_vec[0], size, ACL_MEMCPY_DEVICE_TO_HOST);
    auto shape_size = abstract::ShapeSize(host_shape);
    const trans::TypeIdArgs type_args{host.data(), shape_size, type_id_, type, size};
    sync_ok = trans::TransDataType(type_args, host_ptr);
    if (!sync_ok) {
      MS_LOG(ERROR) << "Trans data type failed.";
      launch_transdata_->FreeLaunchDeviceMem();
      return false;
    }
  }
  launch_transdata_->FreeLaunchDeviceMem();
  return sync_ok;
}

bool AscendDeviceAddress::SyncDeviceToHostAndConvertFormat(const ShapeVector &shape, size_t size,
                                                           mindspore::TypeId type, void *host_ptr) const {
  MS_LOG(DEBUG) << "SyncDeviceToHostAndConvertFormat, Device(format:" << format_
                << ", type_id:" << TypeIdLabel(type_id_) << ", size:" << size_
                << "), Host(type_id:" << TypeIdLabel(type) << ", size:" << size << ")";
  static const std::unordered_map<mindspore::TypeId, std::string> type_id_name_map = {
    {mindspore::kNumberTypeBool, "bool"},       {mindspore::kNumberTypeInt8, "int8"},
    {mindspore::kNumberTypeInt16, "int16"},     {mindspore::kNumberTypeInt32, "int32"},
    {mindspore::kNumberTypeInt64, "int64"},     {mindspore::kNumberTypeFloat16, "float16"},
    {mindspore::kNumberTypeFloat32, "float32"}, {mindspore::kNumberTypeUInt8, "uint8"},
    {mindspore::kNumberTypeUInt16, "uint16"},   {mindspore::kNumberTypeUInt32, "uint32"},
    {mindspore::kNumberTypeUInt64, "uint64"}};
  bool sync_ok = false;
  ShapeVector host_shape = shape;
  if (host_shape.empty()) {
    (void)host_shape.emplace_back(1);
  }
  auto device_shape = GetDeviceShape(&host_shape);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode &&
      type_id_name_map.find(type_id_) != type_id_name_map.end() && !mem_offloaded()) {
    std::pair<std::string, std::string> type_format = std::make_pair(type_id_name_map.at(type_id_), format_);
    if (IsUseTransDataTypeFormat(type_format)) {
      sync_ok = SyncDeviceToHostAndConvertFormatBasedOnTransData(host_shape, size, type, host_ptr);
      return sync_ok;
    }
  }
  std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
  auto host_tmp = std::vector<uint8_t>(size_);
  CopyDeviceToHost(host_tmp.data(), size_);
  auto node_index = GetNodeIndex();
  if (type_id_ != type) {
    const trans::FormatArgs format_args{host_tmp.data(), size_,        kOpFormat_NCHW, format_,
                                        host_shape,      device_shape, type_id_};
    auto host = std::vector<uint8_t>(size_);
    sync_ok = trans::TransFormatFromDeviceToHost(format_args, host.data(), node_index.first, node_index.second);
    if (!sync_ok) {
      MS_LOG(ERROR) << "Trans format failed.";
      return false;
    }
    auto shape_size = abstract::ShapeSize(host_shape);
    const trans::TypeIdArgs type_args{host.data(), shape_size, type_id_, type, size};
    sync_ok = trans::TransDataType(type_args, host_ptr);
    if (!sync_ok) {
      MS_LOG(ERROR) << "Trans data type failed.";
      return false;
    }
  } else {
    const trans::FormatArgs format_args{host_tmp.data(), size_,        kOpFormat_NCHW, format_,
                                        host_shape,      device_shape, type_id_};
    sync_ok = trans::TransFormatFromDeviceToHost(format_args, host_ptr, node_index.first, node_index.second);
    if (!sync_ok) {
      MS_LOG(ERROR) << "Trans format failed.";
      return false;
    }
  }
  return sync_ok;
}

bool AscendDeviceAddress::SyncHostToDevice(const ShapeVector &shape, size_t size, mindspore::TypeId type,
                                           const void *host_ptr, const std::string &format) const {
  MS_LOG(DEBUG) << "SyncHostToDevice, Device(format:" << format_ << ", type_id:" << TypeIdLabel(type_id_)
                << ", size:" << size_ << "), Host(format:" << format << ", type_id:" << TypeIdLabel(type)
                << ", size:" << size << ")";
  if (type_id_ > kMonadTypeBegin && type_id_ < kMonadTypeEnd) {
    return true;
  }
  BindDevice();
  bool sync_ok = false;
  ShapeVector host_shape = shape;
  if (host_shape.empty()) {
    (void)host_shape.emplace_back(1);
  }
  std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
  if (format_ == kOpFormat_NCHW || format_ == kOpFormat_DEFAULT || format_ == kOpFormat_NCDHW || format_ == format) {
    if (type_id_ == type) {
      CopyHostToDevice(host_ptr, size);
      sync_ok = true;
    } else if (type_id_ == kNumberTypeFloat32 && type == kNumberTypeFloat64) {
      if (mem_offloaded()) {
        DoubleToFloat(offload_ptr_, host_ptr, size_ / sizeof(float));
        sync_ok = true;
      } else {
        sync_ok = Float64ToFloatAndSyncHostToDevice(ptr_, size_, host_ptr, size);
      }
    } else {
      auto shape_size = abstract::ShapeSize(host_shape);
      const trans::TypeIdArgs type_args{host_ptr, shape_size, type, type_id_, size};
      auto host_tmp = std::vector<uint8_t>(size_);
      sync_ok = trans::TransDataType(type_args, host_tmp.data());
      if (!sync_ok) {
        MS_LOG(ERROR) << "Trans data type failed.";
        return false;
      }
      CopyHostToDevice(host_tmp.data(), size_);
    }
  } else {
    if (IsOpNeedTransFormat(format_)) {
      sync_ok = ConvertFormatAndSyncHostToDevice(shape, size, type, host_ptr);
    } else {
      MS_LOG(INFO) << "Can not find format transfer function for :" << format_;
    }
  }
  if (!sync_ok) {
    MS_LOG(ERROR) << "Unsupported trans, dev_format:" << format_ << ", dev_type:" << TypeIdLabel(type_id_)
                  << ", host_type:" << TypeIdLabel(type);
    return false;
  }
  return sync_ok;
}

bool AscendDeviceAddress::SyncDeviceToDeviceWithDiffFormatType(const DeviceSync *src_device_addr) const {
  MS_EXCEPTION_IF_NULL(src_device_addr);
  if (type_id_ > kMonadTypeBegin && type_id_ < kMonadTypeEnd) {
    return true;
  }

  auto src_device_address = dynamic_cast<const AscendDeviceAddress *>(src_device_addr);
  MS_EXCEPTION_IF_NULL(src_device_address);
  BindDevice();
  auto host_shape = src_device_address->host_shape();
  if (host_shape.empty()) {
    MS_LOG(WARNING) << "Host shape of source device address is empty, emplace back shape [1],  device address size: "
                    << src_device_address->GetSize()
                    << ", device address type: " << TypeIdLabel(src_device_address->type_id());
    (void)host_shape.emplace_back(1);
  }
  auto host_tensor = std::make_shared<tensor::Tensor>(src_device_address->type_id(), host_shape);
  MS_EXCEPTION_IF_NULL(host_tensor);
  auto host_tensor_size = LongToSize(host_tensor->data().nbytes());
  auto host_tensor_type = host_tensor->data_type();
  if (!src_device_address->SyncDeviceToHost(host_shape, host_tensor_size, host_tensor_type, host_tensor->data_c())) {
    MS_LOG(ERROR) << "Sync device to device failed at the stage of sync device to intermediate Tensor.";
    return false;
  }
  if (!SyncHostToDevice(host_shape, host_tensor_size, host_tensor_type, host_tensor->data_c(),
                        host_tensor->device_info().host_format_)) {
    MS_LOG(ERROR) << "Sync device to device failed at the stage of sync intermediate tensor to device.";
    return false;
  }
  return true;
}

bool AscendDeviceAddress::SyncDeviceToDevice(const DeviceSync *src_device_addr) const {
  MS_EXCEPTION_IF_NULL(src_device_addr);
  auto src_device_address = dynamic_cast<const AscendDeviceAddress *>(src_device_addr);
  MS_EXCEPTION_IF_NULL(src_device_address);
  if (format_ == src_device_address->format() && type_id_ == src_device_address->type_id()) {
    if (src_device_address->mem_offloaded()) {
      auto device_context = GetDeviceContext();
      MS_EXCEPTION_IF_NULL(device_context);
      void *temp_device_ptr = device_context->device_res_manager_->AllocateMemory(src_device_address->GetSize());
      MS_EXCEPTION_IF_NULL(temp_device_ptr);
      SyncMemory(temp_device_ptr, src_device_address->GetOffloadPtr(), src_device_address->GetSize(),
                 ACL_MEMCPY_HOST_TO_DEVICE);
      const auto ret = SyncDeviceToDevice(ShapeVector(), src_device_address->GetSize(), src_device_address->type_id(),
                                          temp_device_ptr, src_device_address->format());
      device_context->device_res_manager_->FreeMemory(temp_device_ptr);
      return ret;
    }
    return SyncDeviceToDevice(ShapeVector(), src_device_address->GetSize(), src_device_address->type_id(),
                              src_device_address->GetPtr(), src_device_address->format());
  } else {
    MS_LOG(INFO) << "Can not copy from device to device directly, format or type is different, src(format:"
                 << src_device_address->format() << ", type_id:" << TypeIdLabel(src_device_address->type_id())
                 << "), dst(format:" << format_ << ", type_id:" << TypeIdLabel(type_id_)
                 << ", use the intermediate Tensor copy instead.";
    return SyncDeviceToDeviceWithDiffFormatType(src_device_addr);
  }
}

bool AscendDeviceAddress::SyncDeviceToDevice(const ShapeVector &shape, size_t size, TypeId type, const void *src_ptr,
                                             const std::string &format) const {
  bool ret = AsyncDeviceToDevice(shape, size, type, src_ptr, format);
  if (!ret) {
    return ret;
  }
  SyncStream();
  return true;
}

bool AscendDeviceAddress::AsyncDeviceToDevice(const ShapeVector & /* shape */, size_t size, TypeId type,
                                              const void *src_ptr, const std::string &format) const {
  MS_LOG(DEBUG) << "AsyncDeviceToDevice, dst(format:" << format_ << ", type_id:" << TypeIdLabel(type_id_)
                << ", size:" << size_ << "), src(format:" << format << ", type_id:" << TypeIdLabel(type)
                << ", size:" << size << ")";
  if (ptr_ == src_ptr) {
    MS_LOG(INFO) << "Dst addr is same with src addr, no need memcpy data.";
    return true;
  }
  if (type_id_ > kMonadTypeBegin && type_id_ < kMonadTypeEnd) {
    return true;
  }
  if (size_ < size) {
    MS_LOG(ERROR) << "Src size is greater than det size, src size is: " << size << ", dst size is: " << size_;
    return false;
  }
  if (format_ != format || type_id_ != type) {
    MS_LOG(ERROR) << "Format or type is different, src(format:" << format << ", type_id:" << TypeIdLabel(type)
                  << "), dst(format:" << format_ << "), type_id:" << TypeIdLabel(type_id_);
    return false;
  }

  BindDevice();
  std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  bool ret;
  if (mem_offloaded()) {
    ret = runtime_instance->MemcpyAsync(offload_ptr_, src_ptr, size, static_cast<int32_t>(RT_MEMCPY_DEVICE_TO_HOST));
  } else {
    ret = runtime_instance->MemcpyAsync(ptr_, src_ptr, size, static_cast<int32_t>(RT_MEMCPY_DEVICE_TO_DEVICE));
  }
  if (!ret) {
    MS_LOG(ERROR) << "MemcpyAsync failed!";
  }
  return ret;
}

bool AscendDeviceAddress::AsyncHostToDevice(const ShapeVector & /* shape */, size_t size, TypeId /* type */,
                                            const void *host_ptr, size_t stream_id) const {
  MS_ERROR_IF_NULL(host_ptr);
  MS_ERROR_IF_NULL(ptr_);
  auto stream = AscendStreamMng::GetInstance().GetStream(stream_id);
  if (stream == nullptr) {
    stream = AscendStreamMng::GetInstance().GetStream(kDefaultStreamIndex);
  }
  MS_ERROR_IF_NULL(stream);

  BindDevice();
  auto ret = aclrtMemcpyAsync(ptr_, size, host_ptr, size, ACL_MEMCPY_HOST_TO_DEVICE, stream);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "Call aclrtMemcpyAsync host to device failed, the error num[" << ret << "]";
    return false;
  }
  return true;
}

bool AscendDeviceAddress::AsyncDeviceToHost(const ShapeVector & /* shape */, size_t size, TypeId /* type */,
                                            void *host_ptr, size_t stream_id) const {
  MS_ERROR_IF_NULL(host_ptr);
  MS_ERROR_IF_NULL(ptr_);
  const auto stream = AscendStreamMng::GetInstance().GetStream(stream_id);
  MS_ERROR_IF_NULL(stream);

  BindDevice();
  auto ret = aclrtMemcpyAsync(host_ptr, size, ptr_, size, ACL_MEMCPY_DEVICE_TO_HOST, stream);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "Call aclrtMemcpyAsync device to host failed, the error num[" << ret << "]";
    return false;
  }
  return true;
}

bool AscendDeviceAddress::ConvertFormatAndSyncHostToDevice(const ShapeVector &shape, size_t size,
                                                           mindspore::TypeId type, const void *host_ptr) const {
  bool sync_ok = false;
  MS_LOG(DEBUG) << "ConvertFormatAndSyncHostToDevice, Device(format:" << format_
                << ", type_id:" << TypeIdLabel(type_id_) << ", size:" << size_
                << "), Host(type_id:" << TypeIdLabel(type) << ", size:" << size << ")";
  ShapeVector host_shape = shape;
  if (host_shape.empty()) {
    (void)host_shape.emplace_back(1);
  }
  auto node_index = GetNodeIndex();
  std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
  (void)GetGroupsWithCache();
  std::vector<int64_t> device_shape;
  if (format_ == kOpFormat_FRAC_NZ) {
    device_shape = trans::TransShapeToDevice(host_shape, format_, node_index.first, node_index.second, type_id_);
  } else {
    host_shape = trans::PaddingShape(host_shape, format_);
    device_shape = trans::TransShapeToDevice(host_shape, format_, node_index.first, node_index.second, type_id_);
  }
  if (type_id_ != type) {
    auto shape_size = abstract::ShapeSize(host_shape);
    const trans::TypeIdArgs type_args{host_ptr, shape_size, type, type_id_, size};
    auto host_tmp = std::vector<uint8_t>(size_);
    sync_ok = trans::TransDataType(type_args, host_tmp.data());
    if (!sync_ok) {
      MS_LOG(ERROR) << "Trans data type failed.";
      return false;
    }
    const trans::FormatArgs format_args{host_tmp.data(), size_,        kOpFormat_NCHW, format_,
                                        host_shape,      device_shape, type_id_};
    auto dst_tmp = std::vector<uint8_t>(size_);
    sync_ok = trans::TransFormat(format_args, dst_tmp.data(), node_index.first, node_index.second);
    if (!sync_ok) {
      MS_LOG(ERROR) << "Trans format failed.";
      return false;
    }
    CopyHostToDevice(dst_tmp.data(), size_);
  } else {
    const trans::FormatArgs format_args{host_ptr, size_, kOpFormat_NCHW, format_, host_shape, device_shape, type_id_};
    auto host_tmp = std::vector<uint8_t>(size_);
    sync_ok = trans::TransFormat(format_args, host_tmp.data(), node_index.first, node_index.second);
    if (!sync_ok) {
      MS_LOG(ERROR) << "Trans format failed.";
      return false;
    }
    CopyHostToDevice(host_tmp.data(), size_);
  }
  return sync_ok;
}

void AscendDeviceAddress::ClearDeviceMemory() {
  std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
  if (offload_ptr_ != nullptr) {
    auto device_context = GetDeviceContext();
    MS_EXCEPTION_IF_NULL(device_context);
    device_context->device_res_manager_->FreeOffloadMemory(offload_ptr_);
    offload_ptr_ = nullptr;
  }
  if (ptr_ != nullptr && from_mem_pool_) {
    if (communication_ptr_ != nullptr) {
      AscendMemoryPool::GetInstance().FreeTensorMem(communication_ptr_);
      communication_ptr_ = nullptr;
    } else {
      AscendMemoryPool::GetInstance().FreeTensorMem(ptr_);
    }
    ptr_ = nullptr;
  }
}

void AscendDeviceAddress::CopyDeviceToHost(void *dst, uint64_t size) const {
  MS_EXCEPTION_IF_NULL(dst);
  if (mem_offloaded()) {
    MS_EXCEPTION_IF_NULL(offload_ptr_);
    SyncMemory(dst, offload_ptr_, size, ACL_MEMCPY_HOST_TO_HOST);
  } else {
    MS_EXCEPTION_IF_NULL(ptr_);
    SyncMemory(dst, ptr_, size, ACL_MEMCPY_DEVICE_TO_HOST);
  }
}

void AscendDeviceAddress::CopyHostToDevice(const void *src, uint64_t size) const {
  MS_EXCEPTION_IF_NULL(src);
  if (mem_offloaded()) {
    MS_EXCEPTION_IF_NULL(offload_ptr_);
    SyncMemory(offload_ptr_, src, size, ACL_MEMCPY_HOST_TO_HOST);
  } else {
    MS_EXCEPTION_IF_NULL(ptr_);
    SyncMemory(ptr_, src, size, ACL_MEMCPY_HOST_TO_DEVICE);
  }
}

AscendDeviceAddress::~AscendDeviceAddress() {
  try {
    ClearDeviceMemory();
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "AscendDeviceAddress destructor failed: " << e.what();
  } catch (...) {
    MS_LOG(ERROR) << "AscendDeviceAddress destructor failed.";
  }
}

#ifndef ENABLE_SECURITY
/*
 * Feature group: Dump.
 * Target device group: Ascend.
 * Runtime category: Old runtime, MindRT.
 * Description: Dump tensor data to file for e2e dump.
 */
bool AscendDeviceAddress::DumpMemToFile(const std::string &filepath, const std::string &host_fmt,
                                        const ShapeVector &host_shape, TypeId host_type, bool trans_flag) const {
  bool ret = false;
  if (filepath.empty()) {
    MS_LOG(ERROR) << "Dump file path is null!";
    return ret;
  }
  if (trans_flag) {
    std::string path = filepath + '.' + host_fmt;
    MS_LOG(INFO) << "E2E Dump path is " << path;
    if (host_type > TypeId::kNumberTypeEnd || host_type < TypeId::kNumberTypeBegin ||
        host_type == kNumberTypeComplex64) {
      MS_LOG(INFO) << "Cannot create tensor with type: " << TypeIdLabel(host_type);
      return false;
    }
    mindspore::tensor::TensorPtr out_tensor = std::make_shared<tensor::Tensor>(host_type, host_shape);
    MS_EXCEPTION_IF_NULL(out_tensor);
    size_t host_size = LongToSize(out_tensor->data().nbytes());
    ret = SyncDeviceToHost(host_shape, host_size, host_type, out_tensor->data_c());
    if (!ret) {
      MS_LOG(ERROR) << "Copy device mem to host failed";
      return ret;
    }
    ret = DumpJsonParser::DumpToFile(path, out_tensor->data_c(), host_size, host_shape, host_type);
  } else {
    auto host_tmp = std::vector<uint8_t>(size_);
    BindDevice();
    SyncStream();
    auto ret_rt_memcpy = aclrtMemcpy(host_tmp.data(), size_, ptr_, size_, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret_rt_memcpy != RT_ERROR_NONE) {
      MS_LOG(ERROR) << "SyncDeviceToHost: aclrtMemcpy mem size[" << size_ << "] fail, ret[" << ret_rt_memcpy << "]";
      return false;
    }
    std::string path = filepath + '.' + format_;
    MS_LOG(INFO) << "E2E Dump path is " << path;
    ret = DumpJsonParser::DumpToFile(path, host_tmp.data(), size_, host_shape, type_id_);
  }

  return ret;
}
#endif

int64_t AscendDeviceAddress::GetGroupsWithCache() const {
  auto node = GetNodeIndex();
  if (node.first != nullptr) {
    groups_ = common::AnfAlgo::GetAttrGroups(node.first, node.second);
  }
  return groups_;
}

#ifdef ENABLE_DEBUGGER
/*
 * Feature group: Dump, Online debugger.
 * Target device group: Ascend.
 * Runtime category: Old runtime, MindRT.
 * Description: Load tensor to host and create tensor_data object for the loaded tensor.
 */
bool AscendDeviceAddress::LoadMemToHost(const std::string &tensor_name, int execution_order,
                                        const std::string &host_fmt, const ShapeVector &host_shape, TypeId host_type,
                                        size_t slot, bool keep_prev, uint32_t root_graph_id, bool force_update,
                                        bool trans_flag) const {
  bool ret = false;
  auto debugger = Debugger::GetInstance();
  MS_EXCEPTION_IF_NULL(debugger);
  if (debugger->TensorExistsInCurrent(tensor_name) && !force_update) {
    MS_LOG(INFO) << tensor_name << " already loaded for this step so not loading it again.";
    return true;
  }
  // TensorData is freed up in AscendSession class
  auto tensor_data = std::make_shared<mindspore::TensorData>();
  MS_EXCEPTION_IF_NULL(tensor_data);
  tensor_data->SetName(tensor_name);
  tensor_data->SetExecutionOrder(execution_order);
  tensor_data->SetSlot(slot);

  if (host_type > TypeId::kNumberTypeEnd || host_type < TypeId::kNumberTypeBegin || host_type == kNumberTypeComplex64) {
    MS_LOG(INFO) << "Cannot create tensor with type: " << TypeIdLabel(host_type);
    return false;
  }
  mindspore::tensor::TensorPtr out_tensor = std::make_shared<tensor::Tensor>(host_type, host_shape);
  MS_EXCEPTION_IF_NULL(out_tensor);
  size_t host_size = LongToSize(out_tensor->data().nbytes());
  if (host_size == 0) {
    MS_LOG(INFO) << "Tensor size is 0 for tensor: " << tensor_name;
    return true;
  }
  bool ret_sync = false;
  if (trans_flag) {
    ret_sync = SyncDeviceToHost(host_shape, host_size, host_type, out_tensor->data_c());
  } else {
    ret_sync = SyncDeviceToHost(host_size, out_tensor->data_c());
  }
  if (!ret_sync) {
    MS_LOG(ERROR) << "Convert format or Copy device mem to host failed";
    return ret;
  }
  MS_LOG(INFO) << "E2E tensor name is " << tensor_name;
  tensor_data->SetTensor(out_tensor);
  tensor_data->SetDataPtr(static_cast<char *>(out_tensor->data_c()));
  tensor_data->SetByteSize(LongToSize(out_tensor->data().nbytes()));
  tensor_data->SetType(host_type);
  tensor_data->SetShape(out_tensor->shape());
  tensor_data->SetRootGraphId(root_graph_id);
  std::string tensor_format = trans_flag ? host_fmt : format_;
  tensor_data->SetFormat(tensor_format);
  ret = debugger->LoadNewTensor(tensor_data, keep_prev);
  MS_LOG(INFO) << "Load tensor '" << tensor_name << "' into debugger tensor loader successfully: format("
               << tensor_format << ")";
  return ret;
}
#endif
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

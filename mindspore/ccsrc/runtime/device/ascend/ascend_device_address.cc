/**
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
#include "runtime/device/ascend/ascend_device_address.h"
#include <memory>
#include <vector>
#include <unordered_map>
#include <utility>
#include <set>
#include <algorithm>
#include "runtime/mem.h"
#include "runtime/device/kernel_runtime_manager.h"
#include "runtime/device/kernel_runtime.h"
#include "runtime/device/memory_manager.h"
#include "runtime/device/convert_tensor_utils.h"
#include "runtime/device/ascend/ascend_launch_transdata.h"
#include "ir/dtype/type.h"
#include "ir/tensor.h"
#include "abstract/utils.h"
#include "utils/utils.h"
#include "common/trans.h"
#ifndef ENABLE_SECURITY
#include "debug/data_dump/dump_json_parser.h"
#endif
#ifdef ENABLE_DEBUGGER
#include "debug/tensor_load.h"
#endif

namespace {
const std::unordered_map<mindspore::TypeId, std::string> type_id_name_map = {
  {mindspore::kNumberTypeBool, "bool"},       {mindspore::kNumberTypeInt8, "int8"},
  {mindspore::kNumberTypeInt16, "int16"},     {mindspore::kNumberTypeInt32, "int32"},
  {mindspore::kNumberTypeInt64, "int64"},     {mindspore::kNumberTypeFloat16, "float16"},
  {mindspore::kNumberTypeFloat32, "float32"}, {mindspore::kNumberTypeUInt8, "uint8"},
  {mindspore::kNumberTypeUInt16, "uint16"},   {mindspore::kNumberTypeUInt32, "uint32"},
  {mindspore::kNumberTypeUInt64, "uint64"}};
const std::set<std::pair<std::string, std::string>> use_trans_data = {
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
}  // namespace

namespace mindspore {
namespace device {
namespace ascend {
const int FLOAT_LEN = sizeof(float);
const int FLOAT16_LEN = 2;
const std::set<std::string> kOpNeedTransFormat = {
  kOpFormat_NHWC,    kOpFormat_HWCN,        kOpFormat_NC1HWC0,       kOpFormat_FRAC_Z,   kOpFormat_C1HWNCoC0,
  kOpFormat_FRAC_NZ, kOpFormat_NC1HWC0_C04, kOpFormat_FRACTAL_Z_C04, kOpFormat_NDC1HWC0, kOpFormat_FRACTAL_Z_3D};

void SyncMemory(void *dst, const void *src, uint64_t size, rtMemcpyKind_t kind) {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  auto execution_mode = ms_context->get_param<int>(MS_CTX_EXECUTION_MODE);
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  runtime_instance->SetContext();

  // Only apply asynchronous copy in Pynative && RT_MEMCPY_HOST_TO_DEVICE mode
  if (execution_mode != kPynativeMode || kind != RT_MEMCPY_HOST_TO_DEVICE) {
    auto ret_rt_memcpy = rtMemcpy(dst, size, src, size, kind);
    if (ret_rt_memcpy != RT_ERROR_NONE) {
      MS_EXCEPTION(DeviceProcessError) << "rtMemcpy failed";
    }
  } else {
    auto ret = runtime_instance->MemcpyAsync(dst, src, size, static_cast<int32_t>(RT_MEMCPY_HOST_TO_DEVICE_EX));
    if (!ret) {
      MS_EXCEPTION(DeviceProcessError) << "MemcpyAsync failed";
    }
  }
}

bool DataSync(void *dst, const void *src, uint64_t src_size) {
  if (dst == src) {
    MS_LOG(INFO) << "dst addr is same with src addr, no need memcpy data.";
    return true;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  auto ret = runtime_instance->MemcpyAsync(dst, src, src_size, static_cast<int32_t>(RT_MEMCPY_DEVICE_TO_DEVICE));
  if (!ret) {
    MS_LOG(ERROR) << "rtMemcpyAsync failed!";
    return false;
  }
  return true;
}

bool FloatToHalfAndSyncHostToDevice(void *dst, size_t dst_size, const void *src, size_t src_size) {
  auto elem_num = src_size / FLOAT_LEN;
  if (elem_num != (dst_size / FLOAT16_LEN)) {
    MS_EXCEPTION(ArgumentError) << "FloatToHalf failed. size not match src_size[" << src_size << "], dst_size["
                                << dst_size << "]";
  }
  std::vector<float16> half_data(elem_num);
  FloatToHalf(half_data.data(), src, elem_num);
  SyncMemory(dst, half_data.data(), dst_size, RT_MEMCPY_HOST_TO_DEVICE);
  return true;
}

bool Float64ToFloatAndSyncHostToDevice(void *dst, size_t dst_size, const void *src, size_t src_size) {
  if (src_size / 2 != dst_size) {
    MS_EXCEPTION(ArgumentError) << "src_size[" << src_size << "], dst_size[" << dst_size << "]";
  }
  size_t elem_num = dst_size / sizeof(float);
  auto host_tmp = std::vector<float>(elem_num);
  DoubleToFloat(host_tmp.data(), src, elem_num);
  SyncMemory(dst, host_tmp.data(), dst_size, RT_MEMCPY_HOST_TO_DEVICE);
  return true;
}

bool SyncDeviceToHostAndHalfToFloat(void *dst, size_t dst_size, const void *src, size_t src_size) {
  auto elem_num = src_size / FLOAT16_LEN;
  if (elem_num != (dst_size / FLOAT_LEN)) {
    MS_EXCEPTION(ArgumentError) << "HalfToFloat failed. size not match src_size[" << src_size << "], dst_size["
                                << dst_size << "]";
  }
  std::vector<float16> half_data(elem_num);
  SyncMemory(half_data.data(), src, src_size, RT_MEMCPY_DEVICE_TO_HOST);
  HalfToFloat(dst, half_data.data(), elem_num);
  return true;
}

bool SyncDeviceToHostAndFloatToFloat64(void *dst, size_t dst_size, const void *src, size_t src_size) {
  if (src_size != dst_size / 2) {
    MS_EXCEPTION(ArgumentError) << "src_size[" << src_size << "], dst_size[" << dst_size << "]";
  }
  size_t elem_num = src_size / sizeof(float);
  auto host_tmp = std::vector<float>(elem_num);
  SyncMemory(host_tmp.data(), src, src_size, RT_MEMCPY_DEVICE_TO_HOST);
  FloatToDouble(dst, host_tmp.data(), elem_num);
  return true;
}

void AscendDeviceAddress::SyncStream() const {
  MS_LOG(DEBUG) << "SyncStream Start!";
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) != kPynativeMode &&
      !ms_context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER)) {
    MS_LOG(DEBUG) << "Finish!";
    return;
  }
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
  SyncStream();
  SyncMemory(host_ptr, ptr_, size, RT_MEMCPY_DEVICE_TO_HOST);
  return true;
}

bool AscendDeviceAddress::SyncHostToDevice(size_t size, const void *host_ptr) const {
  MS_EXCEPTION_IF_NULL(host_ptr);
  SyncMemory(ptr_, host_ptr, size, RT_MEMCPY_HOST_TO_DEVICE);
  return true;
}

bool AscendDeviceAddress::SyncDeviceToHost(const ShapeVector &shape, size_t size, mindspore::TypeId type,
                                           void *host_ptr) const {
  MS_LOG(INFO) << "SyncDeviceToHost, Device(format:" << format_ << ", type_id:" << TypeIdLabel(type_id_)
               << ", size:" << size_ << "), Host(type_id:" << TypeIdLabel(type) << ", size:" << size << ")";
  if (type_id_ > kMonadTypeBegin && type_id_ < kMonadTypeEnd) {
    return true;
  }
  SyncStream();
  bool sync_ok = false;
  std::vector<size_t> host_shape;
  (void)std::transform(shape.begin(), shape.end(), std::back_inserter(host_shape), LongToSize);
  if (host_shape.empty()) {
    host_shape.emplace_back(1);
  }
  if (format_ == kOpFormat_NCHW || format_ == kOpFormat_DEFAULT || format_ == kOpFormat_NCDHW) {
    if (type_id_ == type) {
      SyncMemory(host_ptr, ptr_, size, RT_MEMCPY_DEVICE_TO_HOST);
      sync_ok = true;
    } else if (type_id_ == kNumberTypeFloat32 && type == kNumberTypeFloat64) {
      sync_ok = SyncDeviceToHostAndFloatToFloat64(host_ptr, size, ptr_, size_);
    } else {
      auto shape_size = abstract::ShapeSize(host_shape);
      auto host = std::vector<uint8_t>(size_);
      SyncMemory(host.data(), ptr_, size_, RT_MEMCPY_DEVICE_TO_HOST);
      const trans::TypeIdArgs type_args{host.data(), shape_size, type_id_, type, size_};
      sync_ok = trans::TransDataType(type_args, host_ptr);
      if (!sync_ok) {
        MS_LOG(ERROR) << "Trans data type failed.";
        return false;
      }
    }
  } else {
    auto iter = kOpNeedTransFormat.find(format_);
    if (iter != kOpNeedTransFormat.end()) {
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

std::vector<size_t> AscendDeviceAddress::GetDeviceShape(std::vector<size_t> *host_shape) const {
  MS_EXCEPTION_IF_NULL(host_shape);
  std::vector<size_t> device_shape;
  auto node_index = GetNodeIndex();
  if (format_ == kOpFormat_FRAC_NZ || format_ == kOpFormat_NCDHW) {
    device_shape = trans::TransShapeToDevice(*host_shape, format_, node_index.first, node_index.second);
  } else {
    if (host_shape_.empty()) {
      *host_shape = trans::PaddingShape(*host_shape, format_);
    } else {
      host_shape->clear();
      (void)std::transform(host_shape_.begin(), host_shape_.end(), std::back_inserter(*host_shape), LongToSize);
    }
    device_shape = trans::TransShapeToDevice(*host_shape, format_, node_index.first, node_index.second);
  }
  return device_shape;
}

std::shared_ptr<LaunchKernel> AscendDeviceAddress::CreateLaunchTransData(const std::vector<size_t> &host_shape,
                                                                         const std::string &ori_format,
                                                                         const std::string &dst_format) const {
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetCurrentKernelRuntime();
  MS_EXCEPTION_IF_NULL(runtime_instance);
  auto stream = runtime_instance->compute_stream();
  auto launch_trans_data =
    std::make_shared<AscendLaunchTransData>(stream, type_id_, size_, ori_format, dst_format, host_shape);
  MS_EXCEPTION_IF_NULL(launch_trans_data);
  return launch_trans_data;
}

bool AscendDeviceAddress::SyncDeviceToHostAndConvertFormatBasedOnTransData(const std::vector<size_t> &host_shape,
                                                                           size_t size, mindspore::TypeId type,
                                                                           void *host_ptr) const {
  bool sync_ok = true;
  const std::string dst_format = kOpFormat_NCHW;
  if (launch_transdata_ == nullptr) {
    launch_transdata_ = CreateLaunchTransData(host_shape, format_, dst_format);
    MS_EXCEPTION_IF_NULL(launch_transdata_);
  }
  // launch transdata
  launch_transdata_->SetInputAddr(static_cast<uint8_t *>(ptr_));
  launch_transdata_->LaunchOpKernel();
  SyncStream();
  auto output_addr_vec = launch_transdata_->GetKernelOutputAddr();
  if (output_addr_vec.size() != 1) {
    launch_transdata_->FreeLaunchDeviceMem();
    MS_LOG(EXCEPTION) << "Launch transdata outputs should have only one output";
    return false;
  }
  if (type_id_ == type) {
    SyncMemory(host_ptr, output_addr_vec[0], size, RT_MEMCPY_DEVICE_TO_HOST);
  } else {
    auto host = std::vector<uint8_t>(size);
    SyncMemory(host.data(), output_addr_vec[0], size, RT_MEMCPY_DEVICE_TO_HOST);
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
  MS_LOG(INFO) << "SyncDeviceToHostAndConvertFormat, Device(format:" << format_ << ", type_id:" << TypeIdLabel(type_id_)
               << ", size:" << size_ << "), Host(type_id:" << TypeIdLabel(type) << ", size:" << size << ")";
  bool sync_ok = false;
  std::vector<size_t> host_shape;
  (void)std::transform(shape.begin(), shape.end(), std::back_inserter(host_shape), LongToSize);
  if (host_shape.empty()) {
    host_shape.emplace_back(1);
  }
  std::vector<size_t> device_shape = GetDeviceShape(&host_shape);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode &&
      type_id_name_map.find(type_id_) != type_id_name_map.end()) {
    std::pair<std::string, std::string> type_format = std::make_pair(type_id_name_map.at(type_id_), format_);
    if (use_trans_data.find(type_format) != use_trans_data.end()) {
      sync_ok = SyncDeviceToHostAndConvertFormatBasedOnTransData(host_shape, size, type, host_ptr);
      return sync_ok;
    }
  }
  auto host_tmp = std::vector<uint8_t>(size_);
  SyncMemory(host_tmp.data(), ptr_, size_, RT_MEMCPY_DEVICE_TO_HOST);
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
  MS_LOG(INFO) << "SyncHostToDevice, Device(format:" << format_ << ", type_id:" << TypeIdLabel(type_id_)
               << ", size:" << size_ << "), Host(format:" << format << ", type_id:" << TypeIdLabel(type)
               << ", size:" << size << ")";
  if (type_id_ > kMonadTypeBegin && type_id_ < kMonadTypeEnd) {
    return true;
  }

  bool sync_ok = false;
  std::vector<size_t> host_shape;
  (void)std::transform(shape.begin(), shape.end(), std::back_inserter(host_shape), LongToSize);
  if (host_shape.empty()) {
    host_shape.emplace_back(1);
  }
  if (format_ == kOpFormat_NCHW || format_ == kOpFormat_DEFAULT || format_ == kOpFormat_NCDHW || format_ == format) {
    if (type_id_ == type) {
      SyncMemory(ptr_, host_ptr, size, RT_MEMCPY_HOST_TO_DEVICE);
      sync_ok = true;
    } else if (type_id_ == kNumberTypeFloat32 && type == kNumberTypeFloat64) {
      sync_ok = Float64ToFloatAndSyncHostToDevice(ptr_, size_, host_ptr, size);
    } else {
      auto shape_size = abstract::ShapeSize(host_shape);
      const trans::TypeIdArgs type_args{host_ptr, shape_size, type, type_id_, size};
      auto host_tmp = std::vector<uint8_t>(size_);
      sync_ok = trans::TransDataType(type_args, host_tmp.data());
      if (!sync_ok) {
        MS_LOG(ERROR) << "Trans data type failed.";
        return false;
      }
      SyncMemory(ptr_, host_tmp.data(), size_, RT_MEMCPY_HOST_TO_DEVICE);
    }
  } else {
    auto iter = kOpNeedTransFormat.find(format_);
    if (iter != kOpNeedTransFormat.end()) {
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

bool AscendDeviceAddress::SyncDeviceToDevice(const ShapeVector &, size_t size, TypeId type, const void *src_ptr,
                                             const std::string &format) const {
  MS_LOG(INFO) << "SyncDeviceToDevice, dst(format:" << format_ << ", type_id:" << TypeIdLabel(type_id_)
               << ", size:" << size_ << "), src(format:" << format << ", type_id:" << TypeIdLabel(type)
               << ", size:" << size << ")";
  if (type_id_ > kMonadTypeBegin && type_id_ < kMonadTypeEnd) {
    return true;
  }
  bool sync_ok = false;
  if (format_ == format && type_id_ == type) {
    if (!DataSync(ptr_, src_ptr, size)) {
      MS_LOG(ERROR) << "DataSync failed!";
      return false;
    }
    sync_ok = true;
  } else {
    MS_LOG(ERROR) << "format or type is different.";
    sync_ok = false;
  }
  return sync_ok;
}

bool AscendDeviceAddress::ConvertFormatAndSyncHostToDevice(const ShapeVector &shape, size_t size,
                                                           mindspore::TypeId type, const void *host_ptr) const {
  bool sync_ok = false;
  MS_LOG(INFO) << "ConvertFormatAndSyncHostToDevice, Device(format:" << format_ << ", type_id:" << TypeIdLabel(type_id_)
               << ", size:" << size_ << "), Host(type_id:" << TypeIdLabel(type) << ", size:" << size << ")";
  std::vector<size_t> host_shape;
  (void)std::transform(shape.begin(), shape.end(), std::back_inserter(host_shape), LongToSize);
  if (host_shape.empty()) {
    host_shape.emplace_back(1);
  }
  auto node_index = GetNodeIndex();
  std::vector<size_t> device_shape;
  if (format_ == kOpFormat_FRAC_NZ) {
    device_shape = trans::TransShapeToDevice(host_shape, format_, node_index.first, node_index.second);
  } else {
    host_shape = trans::PaddingShape(host_shape, format_);
    device_shape = trans::TransShapeToDevice(host_shape, format_, node_index.first, node_index.second);
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
    SyncMemory(ptr_, dst_tmp.data(), size_, RT_MEMCPY_HOST_TO_DEVICE);
  } else {
    const trans::FormatArgs format_args{host_ptr, size_, kOpFormat_NCHW, format_, host_shape, device_shape, type_id_};
    auto host_tmp = std::vector<uint8_t>(size_);
    sync_ok = trans::TransFormat(format_args, host_tmp.data(), node_index.first, node_index.second);
    if (!sync_ok) {
      MS_LOG(ERROR) << "Trans format failed.";
      return false;
    }
    SyncMemory(ptr_, host_tmp.data(), size_, RT_MEMCPY_HOST_TO_DEVICE);
  }
  return sync_ok;
}

void AscendDeviceAddress::ClearDeviceMemory() {
  if (ptr_ == nullptr) {
    return;
  }
  if (from_mem_pool_) {
    if (communication_ptr_ != nullptr) {
      AscendMemoryPool::GetInstance().FreeTensorMem(communication_ptr_);
      communication_ptr_ = nullptr;
    } else {
      AscendMemoryPool::GetInstance().FreeTensorMem(ptr_);
    }
    ptr_ = nullptr;
  }
}

AscendDeviceAddress::~AscendDeviceAddress() {
  try {
    ClearDeviceMemory();
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "AscendDeviceAddress destructor failed: " << e.what();
  }
}

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
    mindspore::tensor::TensorPtr out_tensor = std::make_shared<tensor::Tensor>(host_type, host_shape);
    MS_EXCEPTION_IF_NULL(out_tensor);
    size_t host_size = out_tensor->data().nbytes();
    ret = SyncDeviceToHost(host_shape, host_size, host_type, out_tensor->data_c());
    if (!ret) {
      MS_LOG(ERROR) << "Copy device mem to host failed";
      return ret;
    }
#ifndef ENABLE_SECURITY
    ret = DumpJsonParser::DumpToFile(path, out_tensor->data_c(), host_size, host_shape, host_type);
#endif
  } else {
    auto host_tmp = std::vector<uint8_t>(size_);
    auto ret_rt_memcpy = rtMemcpy(host_tmp.data(), size_, ptr_, size_, RT_MEMCPY_DEVICE_TO_HOST);
    if (ret_rt_memcpy != RT_ERROR_NONE) {
      MS_LOG(ERROR) << "SyncDeviceToHost: rtMemcpy mem size[" << size_ << "] fail, ret[" << ret_rt_memcpy << "]";
    }
    std::string path = filepath + '.' + format_;
    MS_LOG(INFO) << "E2E Dump path is " << path;
#ifndef ENABLE_SECURITY
    ret = DumpJsonParser::DumpToFile(path, host_tmp.data(), size_, host_shape, type_id_);
#endif
  }

  return ret;
}

#ifdef ENABLE_DEBUGGER
bool AscendDeviceAddress::LoadMemToHost(const std::string &tensor_name, int execution_order, const std::string &,
                                        const ShapeVector &host_shape, TypeId host_type, size_t slot,
                                        bool keep_prev) const {
  bool ret = false;
  auto debugger = Debugger::GetInstance();
  MS_EXCEPTION_IF_NULL(debugger);
  if (debugger->TensorExistsInCurrent(tensor_name)) {
    MS_LOG(INFO) << tensor_name << " already loaded for this step so not loading it again.";
    return true;
  }
  // TensorData is freed up in AscendSession class
  auto tensor_data = std::make_shared<mindspore::TensorData>();
  MS_EXCEPTION_IF_NULL(tensor_data);
  tensor_data->SetName(tensor_name);
  tensor_data->SetExecutionOrder(execution_order);
  tensor_data->SetSlot(slot);

  mindspore::tensor::TensorPtr out_tensor = std::make_shared<tensor::Tensor>(host_type, host_shape);
  MS_EXCEPTION_IF_NULL(out_tensor);
  size_t host_size = out_tensor->data().nbytes();
  auto ret_sync = SyncDeviceToHost(host_shape, host_size, host_type, out_tensor->data_c());
  if (!ret_sync) {
    MS_LOG(ERROR) << "Copy device mem to host failed";
    return ret;
  }
  MS_LOG(INFO) << "E2E tensor name is " << tensor_name;
  tensor_data->SetTensor(out_tensor);
  tensor_data->SetDataPtr(static_cast<char *>(out_tensor->data_c()));
  tensor_data->SetByteSize(LongToSize(out_tensor->data().nbytes()));
  tensor_data->SetType((unsigned int)host_type);
  tensor_data->SetShape(out_tensor->shape());
  ret = debugger->LoadNewTensor(tensor_data, keep_prev);
  return ret;
}
#endif
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

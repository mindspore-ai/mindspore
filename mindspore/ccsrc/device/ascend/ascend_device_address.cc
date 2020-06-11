/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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
#include "device/ascend/ascend_device_address.h"
#include <memory>
#include <vector>
#include <set>
#include <algorithm>
#include "runtime/mem.h"
#include "device/kernel_runtime_manager.h"
#include "device/convert_tensor_utils.h"
#include "ir/dtype/type.h"
#include "ir/tensor.h"
#include "kernel/common_utils.h"
#include "utils/utils.h"
#include "common/utils.h"
#include "common/trans.h"
#ifdef ENABLE_DUMP_E2E
#include "debug/e2e_dump.h"
#endif
#ifdef ENABLE_DEBUGGER
#include "debug/tensor_load.h"
#endif

namespace mindspore {
namespace device {
namespace ascend {
const int FLOAT_LEN = sizeof(float);
const int FLOAT16_LEN = 2;  // sizeof(float16);
const std::set<std::string> kOpNeedTransFormat = {kOpFormat_NHWC,        kOpFormat_HWCN,         kOpFormat_NC1HWC0,
                                                  kOpFormat_FRAC_Z,      kOpFormat_C1HWNCoC0,    kOpFormat_FRAC_NZ,
                                                  kOpFormat_NC1HWC0_C04, kOpFormat_FRACTAL_Z_C04};

void SyncMemory(void *dst, const void *src, uint64_t size, rtMemcpyKind_t kind) {
  auto ret_rt_memcpy = rtMemcpy(dst, size, src, size, kind);
  if (ret_rt_memcpy != RT_ERROR_NONE) {
    MS_EXCEPTION(DeviceProcessError) << "rtMemcpy failed";
  }
}

bool FloatToHalfAndSyncHostToDevice(void *dst, size_t dst_size, const void *src, size_t src_size) {
  auto elem_num = src_size / FLOAT_LEN;
  if (elem_num != (dst_size / FLOAT16_LEN)) {
    MS_EXCEPTION(ArgumentError) << "FloatToHalf failed. size not match src_size[" << src_size << "], dst_size["
                                << dst_size << "]";
  }
  std::vector<Eigen::half> half_data(elem_num);
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
  std::vector<Eigen::half> half_data(elem_num);
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
  MS_LOG(INFO) << "Start!";
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->execution_mode() != kPynativeMode) {
    MS_LOG(INFO) << "Finish!";
    return;
  }
  auto device_id = ms_context->device_id();
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  auto ret = runtime_instance->SyncStream();
  if (!ret) {
    MS_LOG(EXCEPTION) << "Sync stream error!";
  }
  MS_LOG(INFO) << "Finish!";
}

bool AscendDeviceAddress::SyncDeviceToHost(const std::vector<int> &shape, size_t size, mindspore::TypeId type,
                                           void *host_ptr) const {
  MS_LOG(INFO) << "SyncDeviceToHost, Device(format:" << format_ << ", type_id:" << TypeIdLabel(type_id_)
               << ", size:" << size_ << "), Host(type_id:" << TypeIdLabel(type) << ", size:" << size << ")";
  SyncStream();
  bool sync_ok = false;
  std::vector<size_t> host_shape;
  (void)std::transform(shape.begin(), shape.end(), std::back_inserter(host_shape), IntToSize);
  if (host_shape.empty()) {
    host_shape.emplace_back(1);
  }
  if (format_ == kOpFormat_NCHW || format_ == kOpFormat_DEFAULT || format_ == kOpFormat_NDHWC) {
    if (type_id_ == type) {
      SyncMemory(host_ptr, ptr_, size, RT_MEMCPY_DEVICE_TO_HOST);
      sync_ok = true;
    } else if (type_id_ == kNumberTypeFloat32 && type == kNumberTypeFloat64) {
      sync_ok = SyncDeviceToHostAndFloatToFloat64(host_ptr, size, ptr_, size_);
    } else {
      auto shape_size = trans::ShapeSize(host_shape);
      auto host = std::vector<uint8_t>(size_);
      SyncMemory(host.data(), ptr_, size_, RT_MEMCPY_DEVICE_TO_HOST);
      const trans::TypeIdArgs type_args{host.data(), shape_size, type_id_, type, size};
      sync_ok = trans::TransDataType(type_args, host_ptr);
      if (!sync_ok) {
        MS_LOG(ERROR) << "trans data type failed.";
        return false;
      }
    }
  } else {
    auto iter = kOpNeedTransFormat.find(format_);
    if (iter != kOpNeedTransFormat.end()) {
      sync_ok = SyncDeviceToHostAndConvertFormat(shape, size, type, host_ptr);
    } else {
      MS_LOG(INFO) << "Can not find format transfer for :" << format_;
    }
  }
  if (!sync_ok) {
    MS_LOG(ERROR) << "Not support to trans, dev_format:" << format_ << ", dev_type:" << TypeIdLabel(type_id_)
                  << ", host_type:" << TypeIdLabel(type);
    return false;
  }
  return sync_ok;
}

bool AscendDeviceAddress::SyncDeviceToHostAndConvertFormat(const std::vector<int> &shape, size_t size,
                                                           mindspore::TypeId type, void *host_ptr) const {
  MS_LOG(INFO) << "SyncDeviceToHostAndConvertFormat, Device(format:" << format_ << ", type_id:" << TypeIdLabel(type_id_)
               << ", size:" << size_ << "), Host(type_id:" << TypeIdLabel(type) << ", size:" << size << ")";
  bool sync_ok = false;
  auto host_tmp = std::vector<uint8_t>(size_);
  SyncMemory(host_tmp.data(), ptr_, size_, RT_MEMCPY_DEVICE_TO_HOST);
  std::vector<size_t> host_shape;
  (void)std::transform(shape.begin(), shape.end(), std::back_inserter(host_shape), IntToSize);
  std::vector<size_t> device_shape;
  if (host_shape.empty()) {
    host_shape.emplace_back(1);
  }
  if (format_ == kOpFormat_FRAC_NZ || format_ == kOpFormat_NDHWC) {
    device_shape = trans::TransShapeToDevice(host_shape, format_);
  } else {
    host_shape = trans::PaddingShapeTo4d(host_shape);
    device_shape = trans::TransShapeToDevice(host_shape, format_);
  }
  if (type_id_ != type) {
    const trans::FormatArgs format_args{host_tmp.data(), size_,        kOpFormat_NCHW, format_,
                                        host_shape,      device_shape, type_id_};
    auto host = std::vector<uint8_t>(size_);
    sync_ok = trans::TransFormatFromDeviceToHost(format_args, host.data());
    if (!sync_ok) {
      MS_LOG(ERROR) << "Trans format failed.";
      return false;
    }
    auto shape_size = trans::ShapeSize(host_shape);
    const trans::TypeIdArgs type_args{host.data(), shape_size, type_id_, type, size};
    sync_ok = trans::TransDataType(type_args, host_ptr);
    if (!sync_ok) {
      MS_LOG(ERROR) << "Trans format failed.";
      return false;
    }
  } else {
    const trans::FormatArgs format_args{host_tmp.data(), size_,        kOpFormat_NCHW, format_,
                                        host_shape,      device_shape, type_id_};
    sync_ok = trans::TransFormatFromDeviceToHost(format_args, host_ptr);
    if (!sync_ok) {
      MS_LOG(ERROR) << "Trans format failed.";
      return false;
    }
  }
  return sync_ok;
}

bool AscendDeviceAddress::SyncHostToDevice(const std::vector<int> &shape, size_t size, mindspore::TypeId type,
                                           const void *host_ptr) const {
  MS_LOG(INFO) << "SyncHostToDevice, Device(format:" << format_ << ", type_id:" << TypeIdLabel(type_id_)
               << ", size:" << size_ << "), Host(type_id:" << TypeIdLabel(type) << ", size:" << size << ")";
  SyncStream();
  bool sync_ok = false;
  std::vector<size_t> host_shape;
  (void)std::transform(shape.begin(), shape.end(), std::back_inserter(host_shape), IntToSize);
  if (host_shape.empty()) {
    host_shape.emplace_back(1);
  }
  if (format_ == kOpFormat_NCHW || format_ == kOpFormat_DEFAULT || format_ == kOpFormat_NDHWC) {
    if (type_id_ == type) {
      SyncMemory(ptr_, host_ptr, size_, RT_MEMCPY_HOST_TO_DEVICE);
      sync_ok = true;
    } else if (type_id_ == kNumberTypeFloat32 && type == kNumberTypeFloat64) {
      sync_ok = Float64ToFloatAndSyncHostToDevice(ptr_, size_, host_ptr, size);
    } else {
      auto shape_size = trans::ShapeSize(host_shape);
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
      MS_LOG(INFO) << "Can not find format transfer for :" << format_;
    }
  }
  if (!sync_ok) {
    MS_LOG(ERROR) << "Not support to trans, dev_format:" << format_ << ", dev_type:" << TypeIdLabel(type_id_)
                  << ", host_type:" << TypeIdLabel(type);
    return false;
  }
  return sync_ok;
}

bool AscendDeviceAddress::ConvertFormatAndSyncHostToDevice(const std::vector<int> &shape, size_t size,
                                                           mindspore::TypeId type, const void *host_ptr) const {
  bool sync_ok = false;
  MS_LOG(INFO) << "ConvertFormatAndSyncHostToDevice, Device(format:" << format_ << ", type_id:" << TypeIdLabel(type_id_)
               << ", size:" << size_ << "), Host(type_id:" << TypeIdLabel(type) << ", size:" << size << ")";
  std::vector<size_t> host_shape;
  (void)std::transform(shape.begin(), shape.end(), std::back_inserter(host_shape), IntToSize);
  if (host_shape.empty()) {
    host_shape.emplace_back(1);
  }
  std::vector<size_t> device_shape;
  if (format_ == kOpFormat_FRAC_NZ || format_ == kOpFormat_NDHWC) {
    device_shape = trans::TransShapeToDevice(host_shape, format_);
  } else {
    host_shape = trans::PaddingShapeTo4d(host_shape);
    device_shape = trans::TransShapeToDevice(host_shape, format_);
  }
  if (type_id_ != type) {
    auto shape_size = trans::ShapeSize(host_shape);
    const trans::TypeIdArgs type_args{host_ptr, shape_size, type, type_id_, size};
    auto host_tmp = std::vector<uint8_t>(size_);
    sync_ok = trans::TransDataType(type_args, host_tmp.data());
    if (!sync_ok) {
      MS_LOG(ERROR) << "Trans datatype failed.";
      return false;
    }
    const trans::FormatArgs format_args{host_tmp.data(), size_,        kOpFormat_NCHW, format_,
                                        host_shape,      device_shape, type_id_};
    auto dst_tmp = std::vector<uint8_t>(size_);
    sync_ok = trans::TransFormat(format_args, dst_tmp.data());
    if (!sync_ok) {
      MS_LOG(ERROR) << "Trans format failed.";
      return false;
    }
    SyncMemory(ptr_, dst_tmp.data(), size_, RT_MEMCPY_HOST_TO_DEVICE);
  } else {
    const trans::FormatArgs format_args{host_ptr, size_, kOpFormat_NCHW, format_, host_shape, device_shape, type_id_};
    auto host_tmp = std::vector<uint8_t>(size_);
    sync_ok = trans::TransFormat(format_args, host_tmp.data());
    if (!sync_ok) {
      MS_LOG(ERROR) << "Trans format failed.";
      return false;
    }
    SyncMemory(ptr_, host_tmp.data(), size_, RT_MEMCPY_HOST_TO_DEVICE);
  }
  return sync_ok;
}

AscendDeviceAddress::~AscendDeviceAddress() {
  if (ptr_ == nullptr) {
    return;
  }
  if (from_mem_pool_) {
    AscendMemoryPool::GetInstance().FreeTensorMem(ptr_);
    ptr_ = nullptr;
  }
}

#ifdef ENABLE_DUMP_E2E
bool AscendDeviceAddress::DumpMemToFile(bool trans_flag, const std::string &filepath, const std::string &host_fmt,
                                        const std::vector<int> &host_shape, TypeId host_type) const {
  bool ret = false;
  if (filepath.empty()) {
    MS_LOG(ERROR) << "Dump file path is null!";
    return ret;
  }
  std::string shape = "shape";
  if (host_shape.size()) {
    for (auto &value : host_shape) {
      shape = shape + '_' + std::to_string(value);
    }
  } else {
    shape = shape + "_0";
  }
  std::string file_extension = ".bin";
  if (trans_flag) {
    std::string path = filepath + '_' + shape + '_' + TypeIdLabel(host_type) + '_' + host_fmt + file_extension;
    MS_LOG(INFO) << "E2E Dump path is " << path;
    mindspore::tensor::TensorPtr out_tensor = std::make_shared<tensor::Tensor>(host_type, host_shape);
    size_t host_size = out_tensor->data().nbytes();
    ret = SyncDeviceToHost(host_shape, host_size, host_type, out_tensor->data_c(true));
    if (!ret) {
      MS_LOG(ERROR) << "Copy device mem to host failed";
      return ret;
    }
    ret = mindspore::Dump::DumpToFile(path, out_tensor->data_c(false), host_size);
  } else {
    auto host_tmp = std::vector<uint8_t>(size_);
    auto ret_rt_memcpy = rtMemcpy(host_tmp.data(), size_, ptr_, size_, RT_MEMCPY_DEVICE_TO_HOST);
    if (ret_rt_memcpy != RT_ERROR_NONE) {
      MS_LOG(ERROR) << "SyncDeviceToHost: rtMemcpy mem size[" << size_ << "] fail, ret[" << ret_rt_memcpy << "]";
    }
    std::string path =
      filepath + '_' + shape + '_' + TypeIdToType(type_id_)->ToString() + '_' + format_ + file_extension;
    MS_LOG(INFO) << "E2E Dump path is " << path;
    ret = mindspore::Dump::DumpToFile(path, host_tmp.data(), size_);
  }

  return ret;
}
#endif

#ifdef ENABLE_DEBUGGER
bool AscendDeviceAddress::LoadMemToHost(bool trans_flag, const std::string &tensor_name, int execution_order,
                                        const std::string &host_fmt, const std::vector<int> &host_shape,
                                        TypeId host_type, size_t slot, Debugger *debugger) const {
  bool ret = false;

  DebugServices *debug_services = debugger->get_debug_services();
  TensorLoader *tensor_loader = debug_services->get_tensor_loader();

  if (trans_flag) {
    MS_LOG(INFO) << "E2E tensor name is " << tensor_name;
    mindspore::tensor::TensorPtr out_tensor = std::make_shared<tensor::Tensor>(host_type, host_shape);
    size_t host_size = out_tensor->data().nbytes();
    ret = SyncDeviceToHost(host_shape, host_size, host_type, out_tensor->data_c(true));
    if (!ret) {
      MS_LOG(ERROR) << "Copy device mem to host failed";
      return ret;
    }
    auto tensor_data = std::make_shared<mindspore::TensorData>();
    tensor_data->SetName(tensor_name);
    tensor_data->SetExecutionOrder(execution_order);
    tensor_data->SetTensor(out_tensor);
    tensor_data->SetSlot(slot);
    ret = tensor_loader->LoadNewTensor(tensor_data);

  } else {
    mindspore::tensor::TensorPtr out_tensor = std::make_shared<tensor::Tensor>(type_id_, host_shape);
    size_t host_size = out_tensor->data().nbytes();
    auto ret_rt_memcpy = rtMemcpy(out_tensor->data_c(true), host_size, ptr_, host_size, RT_MEMCPY_DEVICE_TO_HOST);

    auto tensor_data = std::make_shared<mindspore::TensorData>();
    tensor_data->SetName(tensor_name);
    tensor_data->SetExecutionOrder(execution_order);
    tensor_data->SetTensor(out_tensor);
    tensor_data->SetSlot(slot);
    ret = tensor_loader->LoadNewTensor(tensor_data);
    if (ret_rt_memcpy != RT_ERROR_NONE) {
      MS_LOG(ERROR) << "SyncDeviceToHost: rtMemcpy mem size[" << size_ << "] fail, ret[" << ret_rt_memcpy << "]";
    }
    MS_LOG(INFO) << "E2E tensor name is " << tensor_name;
  }
  return ret;
}
#endif

}  // namespace ascend
}  // namespace device
}  // namespace mindspore

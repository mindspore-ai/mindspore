/**
 * Copyright 2019-2024 Huawei Technologies Co., Ltd
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
#include "graph/types.h"
#include "pybind_api/gil_scoped_long_running.h"
#include "runtime/device/kernel_runtime_manager.h"
#include "runtime/device/memory_manager.h"
#include "runtime/device/convert_tensor_utils.h"
#include "plugin/device/ascend/hal/device/ascend_event.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "ir/dtype/type.h"
#include "ir/tensor.h"
#include "abstract/utils.h"
#include "include/common/utils/utils.h"
#include "runtime/device/ms_device_shape_transfer.h"
#include "plugin/device/ascend/hal/device/ascend_device_synchronizer.h"
#ifndef ENABLE_SECURITY
#include "include/backend/debug/data_dump/dump_json_parser.h"
#endif
#ifdef ENABLE_DEBUGGER
#include "debug/tensor_load.h"
#endif
#include "transform/symbol/acl_rt_symbol.h"
#include "transform/symbol/symbol_utils.h"

namespace py = pybind11;
namespace mindspore {
namespace device {
namespace ascend {
const auto kFloat16Bytes = 2;
const auto kFloatBytes = sizeof(float);
const auto kFloat64Bytes = 8;
static std::recursive_mutex transdata_mutx;

#if defined(RT_MEMORY_P2PDMA)
static std::mutex dma_lock;
#endif

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

static const std::set<std::string> basic_format = {kOpFormat_NCHW, kOpFormat_DEFAULT, kOpFormat_NCDHW, kOpFormat_ND};

bool IsOpNeedTransFormat(const std::string &format) {
  static const std::set<std::string> op_need_trans_format = {
    kOpFormat_NHWC,    kOpFormat_HWCN,        kOpFormat_NC1HWC0,       kOpFormat_FRAC_Z,   kOpFormat_C1HWNCoC0,
    kOpFormat_FRAC_NZ, kOpFormat_NC1HWC0_C04, kOpFormat_FRACTAL_Z_C04, kOpFormat_NDC1HWC0, kOpFormat_FRACTAL_Z_3D};
  return op_need_trans_format.find(format) != op_need_trans_format.end();
}

void AscendDeviceAddress::DeviceSynchronizerInit() {
  set_device_synchronizer(std::make_shared<AscendDeviceSynchronizer>());
}

void AscendDeviceAddress::SyncHostMemoryToDeviceWithCopySrc(void *dst, const void *src, uint64_t size,
                                                            aclrtMemcpyKind kind,
                                                            KernelRuntime *runtime_instance) const {
  MS_EXCEPTION_IF_NULL(runtime_instance);

  MS_LOG(DEBUG) << "Begin, size:" << size;
  std::shared_ptr<uint8_t[]> buffer(new (std::nothrow) uint8_t[size]);
  MS_EXCEPTION_IF_NULL(buffer);
  auto ret_code = memcpy_s(buffer.get(), size, src, size);
  // Return ERANGE when the copy size is larger than SECUREC_MEM_MAX_LEN.
  if (ret_code == ERANGE) {
    ConvertSameType(buffer.get(), src, size, type_id());
  }

  const auto stream = AscendStreamMng::GetInstance().GetStream(this->stream_id());
  auto ret = runtime_instance->MemcpyAsync(dst, buffer.get(), size, static_cast<int32_t>(kind), stream);
  if (!ret) {
    MS_LOG(EXCEPTION) << "MemcpyAsync failed!";
  }

  device::CallbackFunc callback_func = [buffer]() {
    // Clear buffer automatically.
    MS_LOG(DEBUG) << "callback_func exec, buffer cnt:" << buffer.use_count();
  };
  auto device_context = GetDeviceContext();
  MS_EXCEPTION_IF_NULL(device_context);
  auto callback_ret = device_context->GetKernelExecutor(false)->LaunchCallback(callback_func, this->stream_id());
  if (!callback_ret) {
    MS_LOG(EXCEPTION) << "LaunchCallback failed";
  }
}

void AscendDeviceAddress::SyncHostMemoryToDeviceForTensorFromNumpy(void *dst, const void *src, uint64_t size,
                                                                   aclrtMemcpyKind kind,
                                                                   KernelRuntime *runtime_instance) const {
  MS_EXCEPTION_IF_NULL(runtime_instance);
  MS_LOG(DEBUG) << "Begin, size:" << size;

  runtime_instance->SetContextForce();
  // Memcpy needs to be synchronized firstm, if tensor data is from numpy.
  const auto stream = AscendStreamMng::GetInstance().GetStream(this->stream_id());
  // cppcheck-suppress unreadVariable
  auto lock = device::KernelRuntime::LockRuntime(stream);
  if (!AscendStreamMng::GetInstance().SyncStream(stream)) {
    MS_EXCEPTION(DeviceProcessError) << "Sync stream error!";
  }

  auto ret_rt_memcpy = CALL_ASCEND_API(aclrtMemcpy, dst, size, src, size, kind);
  MS_LOG(DEBUG) << "tensor is_from_numpy, sync it first";
  if (ret_rt_memcpy != ACL_ERROR_NONE) {
    MS_EXCEPTION(DeviceProcessError) << "aclrtMemcpy failed";
  }
}

void AscendDeviceAddress::SyncHostMemoryToDeviceWithTensorData(void *dst, const void *src, uint64_t size,
                                                               aclrtMemcpyKind kind,
                                                               const tensor::TensorDataPtr &tensor_data,
                                                               KernelRuntime *runtime_instance) const {
  MS_EXCEPTION_IF_NULL(runtime_instance);

  MS_LOG(DEBUG) << "Begin, size:" << size;
  const auto stream = AscendStreamMng::GetInstance().GetStream(this->stream_id());
  auto ret = runtime_instance->MemcpyAsync(dst, src, size, static_cast<int32_t>(kind), stream);
  if (!ret) {
    MS_LOG(EXCEPTION) << "MemcpyAsync failed!";
  }
  device::CallbackFunc callback_func = [tensor_data]() {
    // Clear tensor_data automatically.
    MS_LOG(DEBUG) << "callback_func exec, tensor_data cnt:" << tensor_data.use_count();
  };
  auto device_context = GetDeviceContext();
  MS_EXCEPTION_IF_NULL(device_context);
  auto callback_ret = device_context->GetKernelExecutor(false)->LaunchCallback(callback_func, this->stream_id());
  if (!callback_ret) {
    MS_LOG(EXCEPTION) << "LaunchCallback failed";
  }
}

void AscendDeviceAddress::SyncMemory(void *dst, const void *src, uint64_t size, aclrtMemcpyKind kind,
                                     const tensor::TensorDataPtr &tensor_data) const {
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
    auto ret = runtime_instance->SyncStream();
    if (!ret) {
      MS_LOG(EXCEPTION) << "Sync stream error!";
    }
    if (!common::IsNeedProfileMemory()) {
      auto ret_rt_memcpy = CALL_ASCEND_API(aclrtMemcpy, dst, size, src, size, kind);
      if (ret_rt_memcpy != ACL_ERROR_NONE) {
        MS_EXCEPTION(DeviceProcessError) << "aclrtMemcpy failed";
      }
    }
  } else {
    if (tensor_data == nullptr) {
      // tensor_data is nullptr. Need to copy host first, then dispatch callbacks.
      SyncHostMemoryToDeviceWithCopySrc(dst, src, size, kind, runtime_instance);
      return;
    }
    if (tensor_data->is_from_numpy()) {
      SyncHostMemoryToDeviceForTensorFromNumpy(dst, src, size, kind, runtime_instance);
    } else {
      SyncHostMemoryToDeviceWithTensorData(dst, src, size, kind, tensor_data, runtime_instance);
    }
  }
}

bool AscendDeviceAddress::Float64ToFloatAndSyncHostToDevice(void *dst, size_t dst_size, const void *src,
                                                            size_t src_size,
                                                            const tensor::TensorDataPtr &tensor_data) const {
  if (src_size / kFloat64Bytes != dst_size / kFloatBytes) {
    MS_INTERNAL_EXCEPTION(ArgumentError) << "src_size[" << src_size << "], dst_size[" << dst_size << "]";
  }
  size_t elem_num = dst_size / sizeof(float);
  auto host_tmp = std::vector<float>(elem_num);
  DoubleToFloat(host_tmp.data(), src, elem_num);
  SyncMemory(dst, host_tmp.data(), dst_size, ACL_MEMCPY_HOST_TO_DEVICE, tensor_data);
  return true;
}

bool AscendDeviceAddress::SyncDeviceToHostAndFloatToFloat64(void *dst, size_t dst_size, const void *src,
                                                            size_t src_size) const {
  if (src_size / kFloatBytes != dst_size / kFloat64Bytes) {
    MS_INTERNAL_EXCEPTION(ArgumentError) << "src_size[" << src_size << "], dst_size[" << dst_size << "]";
  }
  size_t elem_num = src_size / sizeof(float);
  auto host_tmp = std::vector<float>(elem_num);
  SyncMemory(host_tmp.data(), src, src_size, ACL_MEMCPY_DEVICE_TO_HOST);
  FloatToDouble(dst, host_tmp.data(), elem_num);
  return true;
}

void AscendDeviceAddress::SetDevicePtrDeleter() {
  if (!address_common_) {
    return;
  }

  address_common_->pointer_ref_count_->set_deleter(
    [communication_ptr = this->communication_ptr_](void *ptr, bool from_mem_pool) {
      if (ptr == nullptr || !from_mem_pool) {
        return;
      }

      if (communication_ptr != nullptr) {
        AscendMemoryPool::GetInstance().FreeTensorMem(communication_ptr);
      } else {
        AscendMemoryPool::GetInstance().FreeTensorMem(ptr);
      }
    });
}

void AscendDeviceAddress::BindDevice() const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (!MsContext::GetInstance()->get_param<bool>(MS_CTX_ENABLE_MINDRT)) {
    return;
  }

  // Bind device by device name and device id on the current thread.
  if (!device_name().empty()) {
    auto ascend_device_context = GetDeviceContext();
    MS_EXCEPTION_IF_NULL(ascend_device_context);
    if (!ascend_device_context->device_res_manager_->BindDeviceToCurrentThread(false)) {
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

bool AscendDeviceAddress::SyncStream(size_t stream_id) const {
  const auto stream = AscendStreamMng::GetInstance().GetStream(stream_id);
  MS_EXCEPTION_IF_NULL(stream);
  BindDevice();
  if (!AscendStreamMng::GetInstance().SyncStream(stream)) {
    MS_LOG(ERROR) << "Sync default stream failed.";
    return false;
  }
  return true;
}

bool AscendDeviceAddress::CopyDeviceToHost(void *dst, const void *src, size_t size, bool async,
                                           size_t stream_id) const {
  return CopyBetweenHostDevice(dst, src, size, async, stream_id, false);
}

bool AscendDeviceAddress::CopyHostToDevice(void *dst, const void *src, size_t size, bool async,
                                           size_t stream_id) const {
  return CopyBetweenHostDevice(dst, src, size, async, stream_id, true);
}

bool AscendDeviceAddress::DeviceToFileDirectly(void *ptr, size_t size, const std::string &file_name,
                                               size_t stream_id) const {
  return CopyBetweenFileDeviceDirectly(ptr, file_name, size, stream_id, false);
}

bool AscendDeviceAddress::FileToDeviceDirectly(void *ptr, size_t size, const std::string &file_name,
                                               size_t stream_id) const {
  return CopyBetweenFileDeviceDirectly(ptr, file_name, size, stream_id, true);
}

bool AscendDeviceAddress::CopyBetweenFileDeviceDirectly(void *ptr, const std::string &file_name, size_t size,
                                                        size_t stream_id, bool file_to_device) const {
#if defined(RT_MEMORY_P2PDMA)
  void *dargs = AscendDmaHandle::GetInstance().GetDargs();
  void *buf = AscendDmaHandle::GetInstance().GetBuf();
  if (dargs == nullptr || buf == nullptr) {
    return false;
  }
  std::lock_guard<std::mutex> lock(dma_lock);
  auto open_flag = file_to_device ? (O_RDWR | O_DIRECT) : (O_RDWR | O_CREAT | O_DIRECT);
  auto nvme_fd = open(file_name.c_str(), open_flag, S_IRUSR | S_IWUSR);
  if (nvme_fd < 0) {
    MS_LOG(ERROR) << "Open file failed, file name:" << file_name;
    return false;
  }
  size_t buf_size = AscendDmaHandle::GetInstance().GetSize();
  size_t count = (size + buf_size - 1) / buf_size;
  for (size_t i = 0; i < count; i++) {
    size_t ptr_offset = i * buf_size;
    size_t cur_size = (i == count - 1) ? (size - ptr_offset) : buf_size;
    if (file_to_device) {
      size_t ret_size = read(nvme_fd, buf, cur_size);
      if (ret_size != cur_size || !SyncStream(stream_id)) {
        MS_LOG(ERROR) << "Read file failed, file name:" << file_name << ", size:" << size;
        close(nvme_fd);
        return false;
      }
      DeviceToDevice(static_cast<uint8_t *>(ptr) + ptr_offset, dargs, cur_size, stream_id);
    } else {
      DeviceToDevice(dargs, static_cast<uint8_t *>(ptr) + ptr_offset, cur_size, stream_id);
      size_t ret_size = write(nvme_fd, buf, cur_size);
      if (ret_size != cur_size || !SyncStream(stream_id)) {
        MS_LOG(ERROR) << "Write file failed, file name:" << file_name << ", size:" << size;
        close(nvme_fd);
        return false;
      }
    }
  }
  close(nvme_fd);
  return true;
#else
  return false;
#endif
}

void AscendDeviceAddress::DeviceToDevice(void *dst, void *src, size_t size, size_t stream_id) const {
  MS_EXCEPTION_IF_NULL(dst);
  MS_EXCEPTION_IF_NULL(src);
  const auto stream = AscendStreamMng::GetInstance().GetStream(stream_id);
  MS_EXCEPTION_IF_NULL(stream);
  BindDevice();
  auto ret = CALL_ASCEND_API(aclrtMemcpyAsync, dst, size, src, size, ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Call aclrtMemcpyAsync device to device failed, the error num[" << ret << "].";
  }
  if (!AscendStreamMng::GetInstance().SyncStream(stream_id)) {
    MS_LOG(EXCEPTION) << "Sync default failed.";
  }
}

bool AscendDeviceAddress::SyncDeviceToHost(size_t size, void *const host_ptr) const {
  MS_EXCEPTION_IF_NULL(host_ptr);
  std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
  BindDevice();
  SyncStream();
  if (!MoveToDevice(false)) {
    MS_LOG(WARNING) << "Move data to device failed, check previous log for details.";
  }
  CopyDeviceToHost(host_ptr, size);
  return true;
}

bool AscendDeviceAddress::SyncHostToDevice(size_t size, const void *host_ptr) const {
  MS_EXCEPTION_IF_NULL(host_ptr);
  std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
  BindDevice();
  if (!MoveToDevice(false)) {
    MS_LOG(WARNING) << "Move data to device failed, check previous log for details.";
  }
  CopyHostToDevice(host_ptr, size, nullptr);
  return true;
}

bool AscendDeviceAddress::SyncDeviceToHost(const ShapeVector &shape, size_t size, mindspore::TypeId type,
                                           void *host_ptr) const {
  MS_LOG(DEBUG) << "SyncDeviceToHost, Device(format:" << format() << ", type_id:" << TypeIdLabel(type_id())
                << ", size:" << GetSize() << "), Host(type_id:" << TypeIdLabel(type) << ", size:" << size << ")";
  if (type_id() > kMonadTypeBegin && type_id() < kMonadTypeEnd) {
    return true;
  }
  BindDevice();
  SyncStream();
  if (!MoveToDevice(false)) {
    MS_LOG(WARNING) << "Move data to device failed, check previous log for details.";
  }
  bool sync_ok = false;
  ShapeVector host_shape = shape;
  if (host_shape.empty()) {
    (void)host_shape.emplace_back(1);
  }
  std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
  if (basic_format.find(format()) != basic_format.end()) {
    if (type_id() == type) {
      CopyDeviceToHost(host_ptr, size);
      sync_ok = true;
    } else if (type_id() == kNumberTypeFloat32 && type == kNumberTypeFloat64) {
      if (mem_offloaded()) {
        FloatToDouble(host_ptr, loadable_mem_->offload_ptr_, GetSize() / sizeof(float));
        sync_ok = true;
      } else {
        sync_ok = SyncDeviceToHostAndFloatToFloat64(host_ptr, size, GetDevicePtr(), GetSize());
      }
    } else {
      auto shape_size = abstract::ShapeSize(host_shape);
      auto host = std::vector<uint8_t>(GetSize());
      CopyDeviceToHost(host.data(), GetSize());
      const trans::TypeIdArgs type_args{host.data(), shape_size, type_id(), type, GetSize()};
      sync_ok = trans::TransDataType(type_args, host_ptr);
      if (!sync_ok) {
        MS_LOG(ERROR) << "Trans data type failed.";
        return false;
      }
    }
  } else {
    if (IsOpNeedTransFormat(format())) {
      sync_ok = SyncDeviceToHostAndConvertFormat(shape, size, type, host_ptr);
    } else {
      MS_LOG(INFO) << "Can not find format transfer function for :" << format();
    }
  }
  if (!sync_ok) {
    MS_LOG(ERROR) << "Unsupported to trans, dev_format:" << format() << ", dev_type:" << TypeIdLabel(type_id())
                  << ", host_type:" << TypeIdLabel(type);
    return false;
  }
  return sync_ok;
}

ShapeVector AscendDeviceAddress::GetDeviceShape(ShapeVector *host_shape) const {
  MS_EXCEPTION_IF_NULL(host_shape);
  ShapeVector device_shape;
  auto node_index = GetNodeIndex();
  if (format() == kOpFormat_FRAC_NZ || format() == kOpFormat_NCDHW) {
    device_shape = trans::TransShapeToDevice(*host_shape, format(), node_index.first, node_index.second, type_id());
  } else {
    if (!DeviceAddress::host_shape().empty()) {
      host_shape->clear();
      *host_shape = DeviceAddress::host_shape();
    }
    *host_shape = trans::PaddingShape(*host_shape, format());
    device_shape = trans::TransShapeToDevice(*host_shape, format(), node_index.first, node_index.second, type_id());
  }
  return device_shape;
}

std::shared_ptr<LaunchTransData> AscendDeviceAddress::CreateLaunchTransData(const ShapeVector &host_shape,
                                                                            const std::string &ori_format,
                                                                            const std::string &dst_format) const {
  int64_t groups = 1;
  if (format() == kOpFormat_FRAC_Z) {
    groups = GetGroupsWithCache();
  }
  auto launch_trans_data = std::make_shared<LaunchTransData>(this->stream_id(), type_id(), GetSize(), ori_format,
                                                             dst_format, host_shape, groups);
  MS_EXCEPTION_IF_NULL(launch_trans_data);
  return launch_trans_data;
}

bool AscendDeviceAddress::SyncDeviceToHostAndConvertFormatBasedOnTransData(const ShapeVector &host_shape, size_t size,
                                                                           mindspore::TypeId type,
                                                                           void *host_ptr) const {
  bool sync_ok = true;
  const std::string dst_format = kOpFormat_NCHW;
  if (launch_transdata_ == nullptr) {
    launch_transdata_ = CreateLaunchTransData(host_shape, format(), dst_format);
    MS_EXCEPTION_IF_NULL(launch_transdata_);
  }
  std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
  // launch transdata
  GilReleaseWithCheck release_gil;
  launch_transdata_->SetInputAddr(GetMutablePtr());
  {
    std::lock_guard<std::recursive_mutex> lock_launch(transdata_mutx);
    launch_transdata_->LaunchOpKernel();
  }

  SyncStream();
  auto output_addr_vec = launch_transdata_->GetKernelOutputAddr();
  if (output_addr_vec.size() != 1) {
    launch_transdata_->FreeDeviceMem();
    MS_LOG(EXCEPTION) << "Launch transdata outputs should have only one output, actual output size: "
                      << output_addr_vec.size();
  }
  if (type_id() == type) {
    SyncMemory(host_ptr, output_addr_vec[0], size, ACL_MEMCPY_DEVICE_TO_HOST);
  } else {
    auto host = std::vector<uint8_t>(size);
    SyncMemory(host.data(), output_addr_vec[0], size, ACL_MEMCPY_DEVICE_TO_HOST);
    auto shape_size = abstract::ShapeSize(host_shape);
    const trans::TypeIdArgs type_args{host.data(), shape_size, type_id(), type, size};
    sync_ok = trans::TransDataType(type_args, host_ptr);
    if (!sync_ok) {
      MS_LOG(ERROR) << "Trans data type failed.";
      launch_transdata_->FreeDeviceMem();
      return false;
    }
  }
  launch_transdata_->FreeDeviceMem();
  return sync_ok;
}

bool AscendDeviceAddress::SyncDeviceToHostAndConvertFormat(const ShapeVector &shape, size_t size,
                                                           mindspore::TypeId type, void *host_ptr) const {
  MS_LOG(DEBUG) << "SyncDeviceToHostAndConvertFormat, Device(format:" << format()
                << ", type_id:" << TypeIdLabel(type_id()) << ", size:" << GetSize()
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
      type_id_name_map.find(type_id()) != type_id_name_map.end() && !mem_offloaded()) {
    std::pair<std::string, std::string> type_format = std::make_pair(type_id_name_map.at(type_id()), format());
    if (IsUseTransDataTypeFormat(type_format)) {
      sync_ok = SyncDeviceToHostAndConvertFormatBasedOnTransData(host_shape, size, type, host_ptr);
      return sync_ok;
    }
  }
  std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
  auto host_tmp = std::vector<uint8_t>(GetSize());
  CopyDeviceToHost(host_tmp.data(), GetSize());
  auto node_index = GetNodeIndex();
  if (type_id() != type) {
    const trans::FormatArgs format_args{host_tmp.data(), GetSize(),    kOpFormat_NCHW, format(),
                                        host_shape,      device_shape, type_id()};
    auto host = std::vector<uint8_t>(GetSize());
    sync_ok = trans::TransFormatFromDeviceToHost(format_args, host.data(), node_index.first, node_index.second);
    if (!sync_ok) {
      MS_LOG(ERROR) << "Trans format failed.";
      return false;
    }
    auto shape_size = abstract::ShapeSize(host_shape);
    const trans::TypeIdArgs type_args{host.data(), shape_size, type_id(), type, size};
    sync_ok = trans::TransDataType(type_args, host_ptr);
    if (!sync_ok) {
      MS_LOG(ERROR) << "Trans data type failed.";
      return false;
    }
  } else {
    const trans::FormatArgs format_args{host_tmp.data(), GetSize(),    kOpFormat_NCHW, format(),
                                        host_shape,      device_shape, type_id()};
    sync_ok = trans::TransFormatFromDeviceToHost(format_args, host_ptr, node_index.first, node_index.second);
    if (!sync_ok) {
      MS_LOG(ERROR) << "Trans format failed.";
      return false;
    }
  }
  return sync_ok;
}

bool AscendDeviceAddress::SyncHostToDeviceImpl(const ShapeVector &shape, size_t size, mindspore::TypeId type,
                                               const void *host_ptr, const std::string &format,
                                               const tensor::TensorDataPtr &tensor_data) const {
  MS_LOG(DEBUG) << "SyncHostToDevice, Device(format:" << DeviceAddress::format()
                << ", type_id:" << TypeIdLabel(type_id()) << ", size:" << GetSize() << "), Host(format:" << format
                << ", type_id:" << TypeIdLabel(type) << ", size:" << size << ")";
  if (type_id() > kMonadTypeBegin && type_id() < kMonadTypeEnd) {
    return true;
  }
  BindDevice();
  if (!MoveToDevice(false)) {
    MS_LOG(WARNING) << "Move data to device failed, check previous log for details.";
  }
  bool sync_ok = false;
  ShapeVector host_shape = shape;
  if (host_shape.empty()) {
    (void)host_shape.emplace_back(1);
  }
  std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
  if (DeviceAddress::format() == format || basic_format.find(DeviceAddress::format()) != basic_format.end()) {
    if (type_id() == type) {
      CopyHostToDevice(host_ptr, size, tensor_data);
      sync_ok = true;
    } else if (type_id() == kNumberTypeFloat32 && type == kNumberTypeFloat64) {
      sync_ok = Float64ToFloatAndSyncHostToDevice(GetDevicePtr(), GetSize(), host_ptr, size, tensor_data);
    } else {
      auto shape_size = abstract::ShapeSize(host_shape);
      const trans::TypeIdArgs type_args{host_ptr, shape_size, type, type_id(), size};
      auto host_tmp = std::vector<uint8_t>(GetSize());
      sync_ok = trans::TransDataType(type_args, host_tmp.data());
      if (!sync_ok) {
        MS_LOG(ERROR) << "Trans data type failed.";
        return false;
      }
      CopyHostToDevice(host_tmp.data(), GetSize(), tensor_data);
    }
  } else {
    if (IsOpNeedTransFormat(DeviceAddress::format())) {
      sync_ok = ConvertFormatAndSyncHostToDevice(shape, size, type, host_ptr, tensor_data);
    } else {
      MS_LOG(INFO) << "Can not find format transfer function for :" << DeviceAddress::format();
    }
  }
  if (!sync_ok) {
    MS_LOG(ERROR) << "Unsupported trans, dev_format:" << DeviceAddress::format()
                  << ", dev_type:" << TypeIdLabel(type_id()) << ", host_type:" << TypeIdLabel(type);
    return false;
  }
  return sync_ok;
}

bool AscendDeviceAddress::SyncHostToDevice(const ShapeVector &shape, size_t size, mindspore::TypeId type,
                                           const void *host_ptr, const std::string &format) const {
  return SyncHostToDeviceImpl(shape, size, type, host_ptr, format);
}

bool AscendDeviceAddress::SyncHostToDevice(const ShapeVector &shape, size_t size, TypeId type,
                                           const std::string &format, const tensor::TensorDataPtr &tensor_data) const {
  MS_EXCEPTION_IF_NULL(tensor_data);
  return SyncHostToDeviceImpl(shape, size, type, tensor_data->data(), format, tensor_data);
}

bool AscendDeviceAddress::SyncDeviceToDeviceWithDiffFormatType(const DeviceSync *src_device_addr) const {
  MS_EXCEPTION_IF_NULL(src_device_addr);
  if (type_id() > kMonadTypeBegin && type_id() < kMonadTypeEnd) {
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
  if (!src_device_address->MoveToDevice(false)) {
    MS_LOG(WARNING) << "Move data to device failed, check previous log for details.";
  }
  if (format() == src_device_address->format() && type_id() == src_device_address->type_id()) {
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
                 << "), dst(format:" << format() << ", type_id:" << TypeIdLabel(type_id())
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
  MS_LOG(DEBUG) << "AsyncDeviceToDevice, dst(format:" << DeviceAddress::format()
                << ", type_id:" << TypeIdLabel(type_id()) << ", size:" << GetSize() << "), src(format:" << format
                << ", type_id:" << TypeIdLabel(type) << ", size:" << size << ")";
  if (GetDevicePtr() == src_ptr) {
    MS_LOG(INFO) << "Dst addr is same with src addr, no need memcpy data.";
    return true;
  }
  if (type_id() > kMonadTypeBegin && type_id() < kMonadTypeEnd) {
    return true;
  }
  if (GetSize() < size) {
    MS_LOG(ERROR) << "Src size is greater than det size, src size is: " << size << ", dst size is: " << GetSize();
    return false;
  }
  if (DeviceAddress::format() != format || type_id() != type) {
    MS_LOG(ERROR) << "Format or type is different, src(format:" << format << ", type_id:" << TypeIdLabel(type)
                  << "), dst(format:" << DeviceAddress::format() << "), type_id:" << TypeIdLabel(type_id());
    return false;
  }

  BindDevice();
  if (!MoveToDevice(false)) {
    MS_LOG(WARNING) << "Move data to device failed, check previous log for details.";
  }
  std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  bool ret;
  if (mem_offloaded()) {
    ret = runtime_instance->MemcpyAsync(loadable_mem_->offload_ptr_, src_ptr, size,
                                        static_cast<int32_t>(ACL_MEMCPY_DEVICE_TO_HOST),
                                        runtime_instance->compute_stream());
  } else {
    ret =
      runtime_instance->MemcpyAsync(GetDevicePtr(), src_ptr, size, static_cast<int32_t>(ACL_MEMCPY_DEVICE_TO_DEVICE),
                                    runtime_instance->compute_stream());
  }
  if (!ret) {
    MS_LOG(ERROR) << "MemcpyAsync failed!";
  }
  return ret;
}

bool AscendDeviceAddress::AsyncHostToDevice(size_t size, TypeId /* type */, const void *host_ptr) const {
  MS_ERROR_IF_NULL(host_ptr);
  BindDevice();
  if (!MoveToDevice(false)) {
    MS_LOG(WARNING) << "Move data to device failed, check previous log for details.";
  }
  MS_ERROR_IF_NULL(GetDevicePtr());

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id);
  MS_EXCEPTION_IF_NULL(runtime_instance);

  auto ret = CALL_ASCEND_API(aclrtMemcpyAsync, GetDevicePtr(), size, host_ptr, size, ACL_MEMCPY_HOST_TO_DEVICE,
                             runtime_instance->compute_stream());
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Call aclrtMemcpyAsync host to device failed, the error num[" << ret << "]";
    return false;
  }
  return true;
}

bool AscendDeviceAddress::AsyncHostToDevice(const ShapeVector & /* shape */, size_t size, TypeId /* type */,
                                            const void *host_ptr, size_t stream_id) const {
  MS_ERROR_IF_NULL(host_ptr);
  BindDevice();
  if (!MoveToDevice(false)) {
    MS_LOG(WARNING) << "Move data to device failed, check previous log for details.";
  }
  MS_ERROR_IF_NULL(GetDevicePtr());
  auto stream = AscendStreamMng::GetInstance().GetStream(stream_id);
  if (stream == nullptr) {
    stream = AscendStreamMng::GetInstance().GetStream(kDefaultStreamIndex);
  }
  MS_ERROR_IF_NULL(stream);

  auto ret = CALL_ASCEND_API(aclrtMemcpyAsync, GetDevicePtr(), size, host_ptr, size, ACL_MEMCPY_HOST_TO_DEVICE, stream);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Call aclrtMemcpyAsync host to device failed, the error num[" << ret << "]";
    return false;
  }
  return true;
}

bool AscendDeviceAddress::AsyncDeviceToHost(const ShapeVector & /* shape */, size_t size, TypeId /* type */,
                                            void *host_ptr, size_t stream_id) const {
  MS_ERROR_IF_NULL(host_ptr);
  BindDevice();
  if (!MoveToDevice(false)) {
    MS_LOG(ERROR) << "Move data to device failed, check previous log for details.";
    return false;
  }
  MS_ERROR_IF_NULL(GetDevicePtr());
  const auto stream = AscendStreamMng::GetInstance().GetStream(stream_id);
  MS_ERROR_IF_NULL(stream);
  auto ret = CALL_ASCEND_API(aclrtMemcpyAsync, host_ptr, size, GetDevicePtr(), size, ACL_MEMCPY_DEVICE_TO_HOST, stream);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Call aclrtMemcpyAsync device to host failed, the error num[" << ret << "]";
    return false;
  }
  return true;
}

bool AscendDeviceAddress::ConvertFormatAndSyncHostToDevice(const ShapeVector &shape, size_t size,
                                                           mindspore::TypeId type, const void *host_ptr,
                                                           const tensor::TensorDataPtr &tensor_data) const {
  bool sync_ok = false;
  MS_LOG(DEBUG) << "ConvertFormatAndSyncHostToDevice, Device(format:" << format()
                << ", type_id:" << TypeIdLabel(type_id()) << ", size:" << GetSize()
                << "), Host(type_id:" << TypeIdLabel(type) << ", size:" << size << ")";
  ShapeVector host_shape = shape;
  if (host_shape.empty()) {
    (void)host_shape.emplace_back(1);
  }
  auto node_index = GetNodeIndex();
  std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
  (void)GetGroupsWithCache();
  std::vector<int64_t> device_shape;
  if (format() == kOpFormat_FRAC_NZ) {
    device_shape = trans::TransShapeToDevice(host_shape, format(), node_index.first, node_index.second, type_id());
  } else {
    host_shape = trans::PaddingShape(host_shape, format());
    device_shape = trans::TransShapeToDevice(host_shape, format(), node_index.first, node_index.second, type_id());
  }
  if (type_id() != type) {
    auto shape_size = abstract::ShapeSize(host_shape);
    const trans::TypeIdArgs type_args{host_ptr, shape_size, type, type_id(), size};
    auto host_tmp = std::vector<uint8_t>(GetSize());
    sync_ok = trans::TransDataType(type_args, host_tmp.data());
    if (!sync_ok) {
      MS_LOG(ERROR) << "Trans data type failed.";
      return false;
    }
    const trans::FormatArgs format_args{host_tmp.data(), GetSize(),    kOpFormat_NCHW, format(),
                                        host_shape,      device_shape, type_id()};
    auto dst_tmp = std::vector<uint8_t>(GetSize());
    sync_ok = trans::TransFormat(format_args, dst_tmp.data(), node_index.first, node_index.second);
    if (!sync_ok) {
      MS_LOG(ERROR) << "Trans format failed.";
      return false;
    }
    CopyHostToDevice(dst_tmp.data(), GetSize(), tensor_data);
  } else {
    const trans::FormatArgs format_args{host_ptr,   GetSize(),    kOpFormat_NCHW, format(),
                                        host_shape, device_shape, type_id()};
    auto host_tmp = std::vector<uint8_t>(GetSize());
    sync_ok = trans::TransFormat(format_args, host_tmp.data(), node_index.first, node_index.second);
    if (!sync_ok) {
      MS_LOG(ERROR) << "Trans format failed.";
      return false;
    }
    CopyHostToDevice(host_tmp.data(), GetSize(), tensor_data);
  }
  return sync_ok;
}

void AscendDeviceAddress::ClearDeviceMemory() {
  std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
  (void)Wait();
  if (loadable_mem_ != nullptr && loadable_mem_->offload_ptr_ != nullptr) {
    auto device_context = GetDeviceContext();
    MS_EXCEPTION_IF_NULL(device_context);
    device_context->device_res_manager_->FreeOffloadMemory(loadable_mem_->offload_ptr_);
    loadable_mem_->offload_ptr_ = nullptr;
  }
  if (GetDevicePtr() != nullptr && from_mem_pool()) {
    if (communication_ptr_ != nullptr) {
      AscendMemoryPool::GetInstance().FreeTensorMem(communication_ptr_);
      communication_ptr_ = nullptr;
    } else {
      AscendMemoryPool::GetInstance().FreeTensorMem(GetDevicePtr());
    }
    SetDevicePtr(nullptr);
  }
}

void AscendDeviceAddress::CopyDeviceToHost(void *dst, uint64_t size) const {
  MS_EXCEPTION_IF_NULL(dst);
  if (mem_offloaded()) {
    MS_EXCEPTION_IF_NULL(loadable_mem_->offload_ptr_);
    SyncMemory(dst, loadable_mem_->offload_ptr_, size, ACL_MEMCPY_HOST_TO_HOST);
  } else {
    MS_EXCEPTION_IF_NULL(GetDevicePtr());
    SyncMemory(dst, GetDevicePtr(), size, ACL_MEMCPY_DEVICE_TO_HOST);
  }
}

void AscendDeviceAddress::CopyHostToDevice(const void *src, uint64_t size,
                                           const tensor::TensorDataPtr &tensor_data) const {
  MS_EXCEPTION_IF_NULL(src);

  if (mem_offloaded()) {
    MS_EXCEPTION_IF_NULL(loadable_mem_->offload_ptr_);
    SyncMemory(loadable_mem_->offload_ptr_, src, size, ACL_MEMCPY_HOST_TO_HOST, tensor_data);
  } else {
    MS_EXCEPTION_IF_NULL(GetDevicePtr());
    if (type_id() == kObjectTypeString) {
      // NOTE: For string type, ge::StringHead.len does not include '\0', since kernel_tensor allocated size including
      // '\0', see method `CreateDeviceAddressForScalarAndString` defined in `device_address_utils.cc`, and method
      // `PrepareDataForStringValue` defined in `data_prepare_actor.cc`, so here pass `size - 1` to `head.len`.
      // NOTE: method `CopyHostToDevice` can be triggered from the two scenarios as below:
      // 1. method `CopyNoneTensorDataToDevice` in `device_address_utils.cc` passes a kernel tensor, the parameter
      // `size` include `ge::StringHead`
      // 2. method `PrepareDataForStringValue` in `data_prepare_actor.cc` passes a raw string, the parameter `size` does
      // not include `ge::StringHead`
      if (size == GetSize() && size >= sizeof(ge::StringHead)) {
        size -= sizeof(ge::StringHead);
      }
      ge::StringHead head{.addr = sizeof(ge::StringHead), .len = static_cast<int64_t>(size) - 1};
      // sync string head info from device to host
      SyncMemory(GetDevicePtr(), &head, sizeof(ge::StringHead), ACL_MEMCPY_HOST_TO_DEVICE, nullptr);
      // sync string body (real contents) from device to host
      SyncMemory(static_cast<void *>(static_cast<char *>(GetDevicePtr()) + sizeof(ge::StringHead)), src, size,
                 ACL_MEMCPY_HOST_TO_DEVICE, tensor_data);
      MS_LOG(DEBUG) << "Copy string info to device, ge::StringHead.len=" << head.len
                    << ", text=" << std::string(static_cast<const char *>(src), head.len)
                    << ", device_addr=" << GetDevicePtr();
    } else {
      SyncMemory(GetDevicePtr(), src, size, ACL_MEMCPY_HOST_TO_DEVICE, tensor_data);
    }
  }
}

bool AscendDeviceAddress::CopyBetweenHostDevice(void *dst, const void *src, size_t size, bool async, size_t stream_id,
                                                bool host_to_device) const {
  MS_EXCEPTION_IF_NULL(dst);
  MS_EXCEPTION_IF_NULL(src);
  auto copy_kind = host_to_device ? ACL_MEMCPY_HOST_TO_DEVICE : ACL_MEMCPY_DEVICE_TO_HOST;
  const auto stream = AscendStreamMng::GetInstance().GetStream(stream_id);
  MS_EXCEPTION_IF_NULL(stream);
  BindDevice();
  auto ret = CALL_ASCEND_API(aclrtMemcpyAsync, dst, size, src, size, copy_kind, stream);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Call aclrtMemcpyAsync device to host failed, the error num[" << ret << "]";
    return false;
  }
  if (async) {
    auto record_event = std::make_shared<AscendEvent>();
    record_event->set_record_stream(stream);
    record_event->RecordEvent();
    if (loadable_mem_ == nullptr) {
      loadable_mem_ = std::make_unique<LoadableMember>();
    }
    loadable_mem_->swap_event_.device_event_ = record_event;
  } else {
    if (!AscendStreamMng::GetInstance().SyncStream(stream)) {
      MS_LOG(ERROR) << "Sync default stream failed.";
      return false;
    }
  }
  return true;
}

bool AscendDeviceAddress::CopyDeviceToHost(void *dst, const void *src, const size_t &size) const {
  SyncMemory(dst, src, size, ACL_MEMCPY_DEVICE_TO_HOST);
  return true;
}

bool AscendDeviceAddress::CopyHostToDevice(void *dst, const void *src, const size_t &size) const {
  SyncMemory(dst, src, size, ACL_MEMCPY_HOST_TO_DEVICE);
  return true;
}

bool AscendDeviceAddress::AsyncDeviceToHost(size_t size, void *host_ptr) const {
  MS_EXCEPTION_IF_NULL(host_ptr);
  if (GetDevicePtr() == host_ptr) {
    MS_LOG(INFO) << "Dst addr is same with src addr, no need copy data.";
    return true;
  }
  BindDevice();
  MS_EXCEPTION_IF_NULL(GetDevicePtr());
  auto device_context = GetDeviceContext();
  MS_EXCEPTION_IF_NULL(device_context);
  auto stream_id = device_context->device_res_manager_->GetCurrentStreamId();
  auto stream = device_context->device_res_manager_->GetStream(stream_id);
  if (stream == nullptr) {
    stream = device_context->device_res_manager_->GetStream(kDefaultStreamIndex);
  }
  MS_ERROR_IF_NULL(stream);
  auto ret = CALL_ASCEND_API(aclrtMemcpyAsync, host_ptr, size, GetDevicePtr(), size, ACL_MEMCPY_DEVICE_TO_HOST, stream);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Call aclrtMemcpyAsync host to device failed, the error num[" << ret << "]";
    return false;
  }
  return true;
}

bool AscendDeviceAddress::AsyncHostToDevice(size_t size, const void *host_ptr) const {
  MS_EXCEPTION_IF_NULL(host_ptr);
  if (GetDevicePtr() == host_ptr) {
    MS_LOG(INFO) << "Dst addr is same with src addr, no need copy data.";
    return true;
  }
  BindDevice();
  auto device_context = GetDeviceContext();
  MS_EXCEPTION_IF_NULL(device_context);
  auto stream_id = device_context->device_res_manager_->GetCurrentStreamId();
  auto stream = device_context->device_res_manager_->GetStream(stream_id);
  if (stream == nullptr) {
    stream = device_context->device_res_manager_->GetStream(kDefaultStreamIndex);
    stream_id = kDefaultStreamIndex;
  }
  MS_ERROR_IF_NULL(stream);
  if (GetDevicePtr() == nullptr) {
    auto ptr = device_context->device_res_manager_->AllocateMemory(size, stream_id);
    MS_EXCEPTION_IF_NULL(ptr);
    SetDevicePtr(ptr);
  }
  auto device_id = MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  runtime_instance->SetContext();
  SyncHostMemoryToDeviceWithCopySrc(GetDevicePtr(), host_ptr, size, ACL_MEMCPY_HOST_TO_DEVICE, runtime_instance);
  return true;
}

AscendDeviceAddress::~AscendDeviceAddress() {
  try {
    // Only release offload memory, release device memory when `kernel_tensor_` in base class destroyed, because maybe
    // multi GPUDeviceAddress objects use same device pointer in ref case.
    std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
    (void)Wait();
    if (loadable_mem_ != nullptr && loadable_mem_->offload_ptr_ != nullptr) {
      auto device_context = GetDeviceContext();
      MS_EXCEPTION_IF_NULL(device_context);
      device_context->device_res_manager_->FreeOffloadMemory(loadable_mem_->offload_ptr_);
      loadable_mem_->offload_ptr_ = nullptr;
    }
    LoadableDeviceAddress::ReleaseResource();
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
  if (GetSize() == 0) {
    MS_LOG(INFO) << "the operator in filepath: " << filepath << ", size == 0";
    return true;
  }
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
    auto host_tmp = std::vector<uint8_t>(GetSize());
    BindDevice();
    SyncStream();
    auto ret_rt_memcpy =
      CALL_ASCEND_API(aclrtMemcpy, host_tmp.data(), GetSize(), GetDevicePtr(), GetSize(), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret_rt_memcpy != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "SyncDeviceToHost: aclrtMemcpy mem size[" << GetSize() << "] fail, ret[" << ret_rt_memcpy << "]";
      return false;
    }
    std::string path = filepath + '.' + format();
    MS_LOG(INFO) << "E2E Dump path is " << path;
    ret = DumpJsonParser::DumpToFile(path, host_tmp.data(), GetSize(), host_shape, type_id());
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
                                        bool trans_flag, bool async_copy) const {
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
  if (async_copy) {
    if (trans_flag) {
      ret_sync = SyncDeviceToHost(host_shape, host_size, host_type, out_tensor->data_c());
    } else {
      ret_sync = SyncDeviceToHost(host_size, out_tensor->data_c());
    }
  } else {
    // copy device to host using sync mode
    auto ret_rt_memcpy = CALL_ASCEND_API(aclrtMemcpy, out_tensor->data_c(), host_size, GetDevicePtr(), GetSize(),
                                         ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret_rt_memcpy != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "SyncDeviceToHost: aclrtMemcpy mem size[" << GetSize() << "] fail, ret[" << ret_rt_memcpy << "]";
      return false;
    } else {
      ret_sync = true;
    }
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
  std::string tensor_format = trans_flag ? host_fmt : format();
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

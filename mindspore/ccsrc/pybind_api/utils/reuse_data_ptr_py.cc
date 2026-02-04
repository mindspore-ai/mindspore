/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include <string>
#include <memory>
#include "ir/tensor.h"
#include "utils/ms_context.h"
#include "include/utils/tensor_py.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "utils/log_adapter.h"
#include "mindapi/base/format.h"

#include "include/runtime/hardware_abstract/stream/multi_stream_controller.h"
#include "include/pynative/utils/pynative_utils.h"
#include "include/utils/convert_utils_py.h"
#include "pybind_api/utils/utils_api.h"

namespace mindspore {

void Synchronize() { runtime::Pipeline::Get().WaitForward(); }

// Reuse src tensor's device address by dst tensor. For internal usage only.
void ReuseDataPtr(const py::object &dst_, const py::object &src_, size_t offset) {
  // get context meta
  Synchronize();
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  const auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  const auto device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);

  // get device
  device::DeviceContextKey host_key = {device::GetDeviceTypeByName(device_name), device_id};
  auto device_ctx = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
  MS_EXCEPTION_IF_NULL(device_ctx);
  device_ctx->Initialize();

  auto stream_id = device_ctx->device_res_manager_->DefaultStream();

  // create src device address if null
  auto src = tensor::ConvertToTensor(src_);
  MS_EXCEPTION_IF_NULL(src);
  MS_EXCEPTION_IF_NULL(src->device_address());
  if (src->device_address()->GetDeviceType() != device_ctx->GetDeviceType()) {
    auto device_ptr = device_ctx->device_res_manager_->AllocateMemory(src->Size(), stream_id);
    auto src_device_address = std::make_shared<device::DeviceAddress>(
      reinterpret_cast<void *>(device_ptr), src->Size(), device::GetDeviceTypeByName(device_name), stream_id);

    MS_LOG(DEBUG) << "Create DeviceAddress, ptr:" << reinterpret_cast<void *>(device_ptr) << ", size:" << src->Size()
                  << ", shape:" << src->shape() << ", data_type:" << TypeIdToString(src->data_type());
    MS_EXCEPTION_IF_NULL(src_device_address);
    device::DeviceContextKey new_host_key = {src_device_address->GetDeviceType(), src_device_address->device_id()};
    device::DeviceContext *host_context =
      device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(new_host_key);
    MS_EXCEPTION_IF_NULL(host_context);
    MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);

    if (!host_context->device_res_manager_->SyncAllStreams()) {
      MS_LOG(ERROR) << "Sync stream failed.";
    }
    DeviceAddressExtPtr src_ext =
      std::make_shared<DeviceAddressExt>(src->format(), src->data_type(), src->shape(), src->storage_info());
    DeviceAddressExtPtr dst_ext =
      std::make_shared<DeviceAddressExt>(Format::DEFAULT_FORMAT, src->data_type(), src->shape());
    SyncCopy(src_device_address, src->device_address(), src_device_address->stream_id(), src_ext, dst_ext);
    src->set_device_address(src_device_address);
    src->set_format(dst_ext->format_);
    src->set_storage_info(nullptr);
  }

  // create device address with src ptr
  uint8_t *ptr = reinterpret_cast<uint8_t *>(src->device_address()->GetMutablePtr());
  auto dst = tensor::ConvertToTensor(dst_);
  MS_EXCEPTION_IF_NULL(dst);
  auto offset_size = offset * UnitSizeInBytes(dst->data_type());
  if (offset_size >= src->Size()) {
    MS_EXCEPTION(ValueError) << "Offset overflow. Expect offset in bytes less than " << src->Size() << ", got "
                             << offset_size;
  }

  auto dst_device_address = std::make_shared<device::DeviceAddress>(
    reinterpret_cast<void *>(ptr + offset_size), dst->Size(), device::GetDeviceTypeByName(device_name), stream_id);

  MS_LOG(DEBUG) << "Create DeviceAddress, ptr:" << reinterpret_cast<void *>(ptr) << ", size:" << dst->Size()
                << ", shape:" << dst->shape() << ", data_type:" << TypeIdToString(dst->data_type());
  MS_EXCEPTION_IF_NULL(dst_device_address);

  // set device address to dst
  dst->set_device_address(dst_device_address);
  dst->set_format(Format::DEFAULT_FORMAT);
  dst->set_storage_info(nullptr);
}

void RegReuseDataPtr(py::module *m) {
  (void)m->def("_reuse_data_ptr", &mindspore::ReuseDataPtr, "Reuse tensor device address.");
}
}  // namespace mindspore

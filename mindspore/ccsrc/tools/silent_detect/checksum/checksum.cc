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

#include "tools/silent_detect/checksum/checksum.h"
#include <memory>
#include <string>
#include <vector>
#include "ir/tensor_new.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "runtime/hardware_abstract/utils.h"
#include "include/runtime/hardware_abstract/stream/multi_stream_controller.h"
#include "tools/silent_detect/checksum/checksum_kernel.h"
#include "tools/silent_detect/checksum/checksum_mgr.h"

namespace mindspore {
namespace checksum {
using kernel::KernelTensorPtr;
using tensor::Tensor;
using tensor::TensorPtr;

constexpr auto kMatMul = "MatMul";
constexpr float kCheckSumThreshold = 1e-20;

inline TensorPtr KernelTensor2Tensor(KernelTensorPtr kernel_tensor) {
  if (!kernel_tensor) {
    return nullptr;
  }
  const void *src = kernel_tensor->device_ptr();
  auto host_type = kernel_tensor->dtype_id();
  auto host_shape = kernel_tensor->GetShapeVector();
  auto device_tensor = kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_tensor);

  auto out_tensor = tensor::from_spec(host_type, host_shape, device::DeviceType::kCPU);
  MS_EXCEPTION_IF_NULL(out_tensor);
  size_t host_size = out_tensor->DataNBytes();
  if (host_size == 0) {
    MS_LOG(WARNING) << "kernel tensor size is 0, skip it.";
    return out_tensor;
  }

  device::DeviceContextKey host_key = {device_tensor->GetDeviceType(), device_tensor->device_id()};
  device::DeviceContext *host_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
  MS_EXCEPTION_IF_NULL(host_context);
  MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);
  if (!host_context->device_res_manager_->CopyDirectly(out_tensor->data_c(), host_size, src, host_size,
                                                       device::CopyType::kD2H)) {
    MS_LOG(EXCEPTION) << "Copy D2H failed";
  }
  return out_tensor;
}

void CheckSumViaCallback(const CNodePtr &cnode, const std::vector<KernelTensor *> &input_kernel_tensors,
                         const std::vector<KernelTensor *> &output_kernel_tensors,
                         const DeviceContext *device_context) {
  // check dimension and dtype
  if (!CheckSumKernel::IsCheckSumSupported(input_kernel_tensors, output_kernel_tensors)) {
    MS_VLOG(VL_ASCEND_SILENT_CHECK) << "CheckSum is not supported for MatMul node: " << cnode->fullname_with_scope();
    return;
  }

  // multi stream protect
  auto stream_id = AnfAlgo::GetStreamId(cnode);
  auto &multi_stream_controller = device::DeviceContextManager::GetInstance().GetMultiStreamController(
    device_context->device_context_key().device_type_);
  if (stream_id != kDefaultStreamIndex) {
    multi_stream_controller->DispatchRecordWaitEvent(stream_id, kDefaultStreamIndex);
  }

  // launch CheckSum kernel
  MS_VLOG(VL_ASCEND_SILENT_CHECK) << "Begin CheckSum for MatMul node: " << cnode->fullname_with_scope();
  auto kernel = CheckSumKernel(device_context);
  KernelTensorPtr result = kernel.LaunchKernelAsync(input_kernel_tensors, output_kernel_tensors, stream_id);

  // callback
  device::CallbackFunc callback_func = [result, cnode]() mutable {
    auto tensor = KernelTensor2Tensor(result);
    if (tensor == nullptr || tensor->data_type() != kNumberTypeFloat32 || tensor->DataSize() != 1) {
      MS_LOG_WITH_NODE(EXCEPTION, cnode) << "CheckSum result is invalid";
    }
    float result_value = static_cast<float *>(tensor->data_c())[0];
    if (result_value < kCheckSumThreshold) {
      MS_VLOG(VL_ASCEND_SILENT_CHECK) << "End CheckSum for MatMul node: " << cnode->fullname_with_scope()
                                      << ", result: " << result_value << ", threshold: " << kCheckSumThreshold;
    } else {
      MS_LOG(WARNING) << "CheckSum result " << result_value << " exceeds threshold " << kCheckSumThreshold
                      << ", cnode: " << cnode->fullname_with_scope();
      CheckSumMgr::GetInstance().SetCheckSumResult(true);
    }
  };
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  const auto &device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  device::DeviceContextKey host_key = {device::GetDeviceTypeByName(device_name), device_id};
  device::DeviceContext *host_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
  MS_EXCEPTION_IF_NULL(host_context);
  MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);
  auto callback_ret = host_context->device_res_manager_->LaunchCallback(callback_func, stream_id);
  if (!callback_ret) {
    MS_LOG(ERROR) << "Async check sum callback launch fail.";
  }

  // multi stream protect
  if (stream_id != kDefaultStreamIndex) {
    multi_stream_controller->DispatchRecordWaitEvent(kDefaultStreamIndex, stream_id);
  }
}

void AscendCheckSum(const CNodePtr &cnode, const std::vector<KernelTensor *> &input_kernel_tensors,
                    const std::vector<KernelTensor *> &output_kernel_tensors, const DeviceContext *device_context) {
  if (!CheckSumMgr::GetInstance().IsCheckSumEnable()) {
    return;
  }
  auto prim = GetCNodePrimitive(cnode);
  if (prim != nullptr && prim->name() == kMatMul) {
    CheckSumViaCallback(cnode, input_kernel_tensors, output_kernel_tensors, device_context);
  }
}
}  // namespace checksum
}  // namespace mindspore

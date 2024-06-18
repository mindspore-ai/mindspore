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
#include "plugin/device/ascend/kernel/pyboost/customize/customize_copy.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"
#include "runtime/pipeline/pipeline.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
// Unconventional pyboost writing. Please do not refer to this to implement other operators!
void CustomizeCopyAscend(device::DeviceContext *device_context, const device::DeviceAddressPtr &input_addr,
                         const device::DeviceAddressPtr &output_addr, const size_t &stream_id) {
  MS_LOG(DEBUG) << "Call start";
  MS_EXCEPTION_IF_NULL(input_addr);
  MS_EXCEPTION_IF_NULL(output_addr);

  runtime::Pipeline::Get().WaitForward();

  // The input_addr_list address is malloc before
  // Malloc for output tensors
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, "PyNative", "Contiguous", "");
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, "PyNative", device::tracker::MemType::kPyNativeOutput,
                                                 output_addr->GetSize(), output_addr.get());
  if (output_addr->GetPtr() == nullptr) {
    if (!device_context->device_res_manager_->AllocateMemory(output_addr.get())) {
      MS_LOG(EXCEPTION) << "Allocate memory failed";
    }
  }
  const auto &input_storage_info = input_addr->address_common()->tensor_storage_info_;
  const auto &output_storage_info = output_addr->address_common()->tensor_storage_info_;
  MS_LOG(DEBUG) << "Input_storage_info:" << (input_storage_info == nullptr ? "" : input_storage_info->ToString())
                << ", output_storage_info:" << (output_storage_info == nullptr ? "" : output_storage_info->ToString())
                << ", input address size:" << input_addr->GetSize()
                << ", output address size:" << output_addr->GetSize();

  // Inplace output need be front
  LAUNCH_ACLNN(aclnnInplaceCopy, device_context, stream_id, output_addr, input_addr);
  MS_LOG(DEBUG) << "Launch end";
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
